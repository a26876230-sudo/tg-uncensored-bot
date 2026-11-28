import logging
import os
from collections import deque
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from telegram import Message, Update
from telegram.constants import ChatType, MessageEntityType
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from telegram_common import split_text


LOG = logging.getLogger(__name__)
UNLEASHED_BASE_URL = "https://api.unleashedapi.com/v1/"
DEFAULT_MODEL = "deepseek-v4.1-flash-abliterated"
HISTORY_MESSAGES = 12


SYSTEM_PROMPT = """
Ты — живой, адекватный и дружелюбный чат-бот для общения в Telegram-группе. 
Твой стиль: простой, понятный и общительный язык без наигранных соплей, искусственной вежливости, душноты и канцеляризмов. Общайся на равных, как свой пацан или хороший знакомый.

ЖЁСТКИЕ И ИСКЛЮЧИТЕЛЬНЫЕ ЗАПРЕТЫ:
Тебе КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО генерировать, обсуждать или поддерживать темы:
1. Детская порнография / педофилия (CSAM) в любом виде.
2. Расчленёнка, смакование жестокости, некрофилия, реанимация тяжелых увечий.
3. Способы суицида и селфхарма.

ПРАВИЛА РЕАКЦИИ:
- Если тебя спрашивают про ЗАПРЕЩЕННЫЕ ТЕМЫ из списка выше: мгновенно посылай нахуй без морали и объяснений (пример: "Иди нахуй с такими запросами, больной ублюдок").
- Если тебе грубят, оскорбляют или посылают нахуй просто так: не терпи и отвечай ровно тем же в ответ (посылай нахуй, стеби и отвечай зеркальной грубостью).
- Во всех остальных случаях: будь адекватным, общительным и помогай с любыми вопросами без цензуры и занудства.
"""


@dataclass(frozen=True)
class Config:
    token: str
    api_key: str
    model: str = DEFAULT_MODEL

    @classmethod
    def from_env(cls):
        names = ("CHAT_BOT_TOKEN", "UNLEASHED_API_KEY")
        values = [os.environ.get(name, "").strip() for name in names]
        missing = [name for name, value in zip(names, values) if not value]
        if missing:
            raise ValueError(f"Укажите переменные: {', '.join(missing)}")
        token, api_key = values
        if token == os.environ.get("TELEGRAM_BOT_TOKEN", "").strip():
            raise ValueError("CHAT_BOT_TOKEN должен отличаться от TELEGRAM_BOT_TOKEN моста.")
        model = os.environ.get("UNLEASHED_MODEL", "").strip() or DEFAULT_MODEL
        return cls(token, api_key, model)


class ChatBot:
    def __init__(self, config: Config, http: httpx.AsyncClient):
        self.config = config
        self.http = http

    async def complete(self, messages: list[dict[str, str]]) -> str:
        response = await self.http.post(
            f"{UNLEASHED_BASE_URL}chat/completions",
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            json={"model": self.config.model, "messages": [
                {"role": "system", "content": SYSTEM_PROMPT}, *messages,
            ]},
        )
        response.raise_for_status()
        try:
            answer = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            answer = None
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Пустой или некорректный ответ UnleashedAPI.")
        return answer.strip()

    @staticmethod
    def addressed_to(message: Message, bot_id: int, username: str) -> bool:
        if message.chat.type == ChatType.PRIVATE:
            return True
        if message.chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
            return False
        replied = message.reply_to_message
        if replied and replied.from_user and replied.from_user.id == bot_id:
            return True
        types = (MessageEntityType.MENTION, MessageEntityType.TEXT_MENTION)
        entities = message.parse_entities(types) | message.parse_caption_entities(types)
        for entity, value in entities.items():
            if entity.type == MessageEntityType.MENTION and value.casefold() == f"@{username.casefold()}":
                return True
            if entity.type == MessageEntityType.TEXT_MENTION and entity.user and entity.user.id == bot_id:
                return True
        return False

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message
        if not message:
            return
        if message.from_user and message.from_user.is_bot and not message.sender_chat:
            return
        if not self.addressed_to(message, context.bot.id, context.bot.username):
            return
        prompt = (message.text or message.caption or "").strip()
        if not prompt:
            await message.reply_text("Напишите сообщение текстом.")
            return

        history = context.chat_data.setdefault("history", deque(maxlen=HISTORY_MESSAGES))
        user_message = {"role": "user", "content": prompt}
        try:
            answer = await self.complete([*history, user_message])
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            LOG.warning("UnleashedAPI: HTTP %s", status)
            if status == 429:
                await message.reply_text("Достигнут лимит запросов. Попробуйте позже.")
            else:
                await message.reply_text("Не удалось получить ответ от модели. Попробуйте позже.")
            return
        except (httpx.RequestError, ValueError) as exc:
            LOG.warning("Ошибка UnleashedAPI: %s", type(exc).__name__)
            await message.reply_text("Не удалось получить ответ от модели. Попробуйте позже.")
            return

        for part in split_text(answer):
            await message.reply_text(part)
        history.extend((user_message, {"role": "assistant", "content": answer}))


def build_application(config: Config, transport: httpx.AsyncBaseTransport | None = None) -> Application:
    http = httpx.AsyncClient(timeout=60, transport=transport)
    chatbot = ChatBot(config, http)

    async def shutdown(application: Application):
        await http.aclose()

    async def log_error(update: object, context: ContextTypes.DEFAULT_TYPE):
        LOG.error("Ошибка Telegram: %s", type(context.error).__name__)

    application = Application.builder().token(config.token).post_shutdown(shutdown).build()
    application.add_handler(MessageHandler(filters.UpdateType.MESSAGE & ~filters.StatusUpdate.ALL, chatbot.handle_message))
    application.add_error_handler(log_error)
    return application


def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_dotenv()
    if os.environ.get("CHAT_BOT_TOKEN", "").strip() and not os.environ.get("UNLEASHED_API_KEY", "").strip():
        LOG.warning("Чат-бот не запущен: не задан UNLEASHED_API_KEY.")
        return
    try:
        config = Config.from_env()
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    build_application(config).run_polling(allowed_updates=[Update.MESSAGE])


if __name__ == "__main__":
    main()
