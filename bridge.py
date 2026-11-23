import logging
import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from groq import AsyncGroq
from telegram import Message, Update
from telegram.constants import ChatType
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from telegram_common import parse_owner_ids, split_text


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    token: str
    groq_key: str
    owner_ids: tuple[int, ...]
    webhook_url: str
    webhook_secret: str

    @classmethod
    def from_env(cls):
        names = (
            "TELEGRAM_BOT_TOKEN", "GROQ_API_KEY", "TELEGRAM_OWNER_ID",
            "DEVIN_WEBHOOK_URL", "DEVIN_WEBHOOK_SECRET",
        )
        values = [os.environ.get(name, "").strip() for name in names]
        missing = [name for name, value in zip(names, values) if not value]
        if missing:
            raise ValueError(f"Укажите переменные: {', '.join(missing)}")
        token, groq_key, owner_ids, webhook_url, webhook_secret = values
        return cls(token, groq_key, parse_owner_ids(owner_ids), webhook_url, webhook_secret)


class Bridge:
    def __init__(self, config: Config, groq: AsyncGroq, http: httpx.AsyncClient):
        self.config = config
        self.groq = groq
        self.http = http

    async def authorized(self, message: Message) -> bool:
        user = message.from_user
        if message.chat.type != ChatType.PRIVATE or user is None or user.is_bot:
            return False
        if user.id not in self.config.owner_ids:
            await message.reply_text(f"Нет доступа. Ваш Telegram ID: {user.id}")
            return False
        return True

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message and await self.authorized(update.message):
            await update.message.reply_text(
                "Отправьте текст или голосовое сообщение — передам задачу в Software Factory."
            )

    async def show_id(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message and await self.authorized(update.message):
            await update.message.reply_text(f"Ваш Telegram ID: {update.message.from_user.id}")

    async def transcribe(self, message: Message) -> str:
        if message.voice:
            media, name = message.voice, "voice.ogg"
        elif message.audio:
            media, name = message.audio, message.audio.file_name or "audio.mp3"
        else:
            media, name = message.video_note, "video.mp4"
        file = await media.get_file()
        data = await file.download_as_bytearray()
        result = await self.groq.audio.transcriptions.create(
            file=(name, bytes(data)), model="whisper-large-v3",
        )
        return result.text.strip()

    async def forward(self, payload: dict):
        response = await self.http.post(
            self.config.webhook_url,
            headers={"X-Webhook-Secret": self.config.webhook_secret},
            json=payload,
        )
        response.raise_for_status()

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message or not await self.authorized(message):
            return
        sender = message.from_user
        payload = {
            "source": "telegram",
            "kind": "text",
            "chat_id": message.chat.id,
            "message_id": message.message_id,
            "from": sender.username or sender.full_name,
            "text": message.text or message.caption or "",
        }
        if message.document:
            payload["attachments"] = [{
                "file_id": message.document.file_id,
                "file_name": message.document.file_name or "document",
            }]
        elif message.photo:
            payload["attachments"] = [{"file_id": message.photo[-1].file_id, "file_name": "photo.jpg"}]

        if message.voice or message.audio or message.video_note:
            try:
                transcript = await self.transcribe(message)
            except Exception as exc:
                LOG.warning("Не удалось распознать сообщение: %s", type(exc).__name__)
                await message.reply_text("❌ Не удалось распознать голосовое сообщение.")
                return
            if not transcript:
                await message.reply_text("❌ В голосовом сообщении не удалось разобрать речь.")
                return
            payload.update(kind="voice", text=transcript)
            for chunk in split_text(f"🎙 {transcript}"):
                await message.reply_text(chunk)
        elif not (payload["text"] or payload.get("attachments")):
            return

        try:
            await self.forward(payload)
        except httpx.HTTPError as exc:
            LOG.warning("Не удалось передать задачу: %s", type(exc).__name__)
            await message.reply_text("❌ Не удалось передать задачу в Software Factory.")
        else:
            await message.reply_text("✅ Передал задачу")


def build_application(config: Config) -> Application:
    groq = AsyncGroq(api_key=config.groq_key, timeout=60)
    http = httpx.AsyncClient(timeout=15)
    bridge = Bridge(config, groq, http)

    async def shutdown(application: Application):
        await groq.close()
        await http.aclose()

    async def log_error(update: object, context: ContextTypes.DEFAULT_TYPE):
        LOG.error("Ошибка Telegram: %s", type(context.error).__name__)

    application = Application.builder().token(config.token).post_shutdown(shutdown).build()
    application.add_handler(CommandHandler("start", bridge.start, filters.UpdateType.MESSAGE))
    application.add_handler(CommandHandler("id", bridge.show_id, filters.UpdateType.MESSAGE))
    application.add_handler(MessageHandler(filters.UpdateType.MESSAGE & ~filters.COMMAND, bridge.handle_message))
    application.add_error_handler(log_error)
    return application


def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv()
    try:
        config = Config.from_env()
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    build_application(config).run_polling(allowed_updates=[Update.MESSAGE])


if __name__ == "__main__":
    main()
