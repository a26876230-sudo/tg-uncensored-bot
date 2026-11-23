import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from telegram import Update
from telegram.request import HTTPXRequest

from bridge import Bridge, Config, build_application


def make_message(user_id=111, **fields):
    return SimpleNamespace(
        chat=SimpleNamespace(id=user_id, type="private"),
        from_user=SimpleNamespace(id=user_id, is_bot=False, username="owner", full_name="Owner Name"),
        message_id=42, reply_text=AsyncMock(),
        **dict.fromkeys(("text", "caption", "voice", "audio", "video_note", "document", "photo")) | fields,
    )


def make_bridge(transport, groq=None):
    config = Config("123:fake", "fake", (111, 222), "https://devin.example/hook", "fake-secret")
    return Bridge(config, groq, httpx.AsyncClient(transport=httpx.MockTransport(transport)))


def route(bridge, message):
    async def run():
        async with bridge.http:
            await bridge.handle_message(SimpleNamespace(message=message), None)
    asyncio.run(run())


def test_unauthorized_never_forwards_or_downloads():
    media = SimpleNamespace(get_file=AsyncMock())
    message = make_message(333, voice=media)
    route(make_bridge(lambda request: pytest.fail("unexpected webhook")), message)
    media.get_file.assert_not_awaited()
    message.reply_text.assert_awaited_once_with("Нет доступа. Ваш Telegram ID: 333")


def test_groups_are_ignored_even_for_owner():
    message = make_message(text="private task")
    message.chat.type = "group"
    route(make_bridge(lambda request: pytest.fail("unexpected webhook")), message)
    message.reply_text.assert_not_awaited()


@pytest.mark.parametrize(
    ("field", "file_name"),
    [("voice", "voice.ogg"), ("audio", "recording.wav"), ("video_note", "video.mp4")],
)
def test_transcribes_and_echoes_before_forwarding(field, file_name):
    file = SimpleNamespace(download_as_bytearray=AsyncMock(return_value=bytearray(b"audio")))
    media = SimpleNamespace(get_file=AsyncMock(return_value=file), file_name="recording.wav")
    message = make_message(222, **{field: media})
    create = AsyncMock(return_value=SimpleNamespace(text="  Собери APK  "))
    groq = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))

    def post(request):
        message.reply_text.assert_awaited_once_with("🎙 Собери APK")
        assert request.method == "POST"
        assert request.headers["X-Webhook-Secret"] == "fake-secret"
        assert json.loads(request.content) == {
            "source": "telegram", "kind": "voice", "chat_id": 222,
            "message_id": 42, "from": "owner", "text": "Собери APK",
        }
        return httpx.Response(204)

    route(make_bridge(post, groq), message)
    create.assert_awaited_once_with(file=(file_name, b"audio"), model="whisper-large-v3")
    assert message.reply_text.await_count == 2
    assert message.reply_text.await_args.args == ("✅ Передал задачу",)


def test_document_metadata_and_failed_webhook():
    document = SimpleNamespace(file_id="document-id", file_name="task.txt")
    message = make_message(document=document, caption="Смотри файл")
    message.from_user.username = None

    def post(request):
        assert json.loads(request.content) == {
            "source": "telegram", "kind": "text", "chat_id": 111,
            "message_id": 42, "from": "Owner Name", "text": "Смотри файл",
            "attachments": [{"file_id": "document-id", "file_name": "task.txt"}],
        }
        return httpx.Response(403)

    route(make_bridge(post), message)
    message.reply_text.assert_awaited_once_with("❌ Не удалось передать задачу в Software Factory.")


@pytest.mark.parametrize(
    ("text", "reply", "forwarded"),
    [
        ("Собери APK", "✅ Передал задачу", True),
        ("/health check APK", "✅ Передал задачу", True),
        ("/id", "Ваш Telegram ID: 111", False),
        ("/start", "Отправьте текст или голосовое сообщение — передам задачу в Software Factory.", False),
    ],
)
def test_application_dispatches_text_to_local_webhook(monkeypatch, text, reply, forwarded):
    received = []
    telegram_calls = []

    class Webhook(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, self.headers["X-Webhook-Secret"], payload))
            self.send_response(204)
            self.end_headers()

    sender = {"id": 111, "is_bot": False, "first_name": "Owner", "username": "owner"}
    bot = {"id": 123, "is_bot": True, "first_name": "Bridge", "username": "bridge_bot"}
    message = {
        "message_id": 42, "date": 1, "chat": {"id": 111, "type": "private"}, "from": sender,
    }

    async def telegram_api(self, url, method, request_data=None, **kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        if endpoint == "getMe":
            result = bot
        elif endpoint == "sendMessage":
            telegram_calls.append(request_data.parameters)
            result = {**message, "from": bot, "text": request_data.parameters["text"]}
        else:
            pytest.fail(f"unexpected Telegram call: {endpoint}")
        return 200, json.dumps({"ok": True, "result": result}).encode()

    monkeypatch.setattr(HTTPXRequest, "do_request", telegram_api)
    with ThreadingHTTPServer(("127.0.0.1", 0), Webhook) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()

        async def run():
            config = Config(
                "123:fake", "fake", (111,),
                f"http://127.0.0.1:{server.server_port}/hook", "fake-secret",
            )
            application = build_application(config)
            try:
                async with application:
                    await application.start()
                    try:
                        update = Update.de_json({
                            "update_id": 1,
                            "message": {
                                **message,
                                "text": text,
                                "entities": [
                                    {"type": "bot_command", "offset": 0, "length": len(text.split()[0])}
                                ] if text.startswith("/") else [],
                            },
                        }, application.bot)
                        await application.process_update(update)
                    finally:
                        await application.stop()
            finally:
                await application.post_shutdown(application)

        try:
            asyncio.run(run())
        finally:
            server.shutdown()
            thread.join()

    expected = [("/hook", "fake-secret", {
        "source": "telegram", "kind": "text", "chat_id": 111,
        "message_id": 42, "from": "owner", "text": text,
    })]
    assert received == (expected if forwarded else [])
    assert [(call["chat_id"], call["text"]) for call in telegram_calls] == [
        (111, reply),
    ]
