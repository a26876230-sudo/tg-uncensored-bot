import asyncio
import json

import httpx
import pytest
from telegram import Update
from telegram.request import HTTPXRequest

from bot import DEFAULT_MODEL, SYSTEM_PROMPT, Config, build_application


BOT = {"id": 123, "is_bot": True, "first_name": "Chat", "username": "chat_bot"}
USER = {"id": 456, "is_bot": False, "first_name": "Owner"}


def make_message(text, chat_id=456, chat_type="private", **fields):
    return {
        "message_id": 42, "date": 1, "chat": {"id": chat_id, "type": chat_type},
        "from": USER, "text": text, **fields,
    }


@pytest.fixture
def dispatch(monkeypatch):
    sent = []

    async def telegram_api(self, url, method, request_data=None, **kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        if endpoint == "getMe":
            result = BOT
        elif endpoint == "sendMessage":
            params = request_data.parameters
            sent.append((params["chat_id"], params["text"]))
            result = make_message(params["text"], params["chat_id"], **{"from": BOT})
        else:
            pytest.fail(f"unexpected Telegram call: {endpoint}")
        return 200, json.dumps({"ok": True, "result": result}).encode()

    monkeypatch.setattr(HTTPXRequest, "do_request", telegram_api)

    def run(model_api, messages, model=DEFAULT_MODEL):
        async def process():
            config = Config("123:fake", "ulk-fake", model)
            application = build_application(config, httpx.MockTransport(model_api))
            async with application:
                for index, message in enumerate(messages):
                    data = {"update_id": index, "message": message}
                    await application.process_update(Update.de_json(data, application.bot))
            await application.post_shutdown(application)

        asyncio.run(process())
        return sent

    return run


def completion(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_config_uses_only_chat_token_and_allows_model_override(monkeypatch):
    monkeypatch.delenv("CHAT_BOT_TOKEN", raising=False)
    monkeypatch.delenv("UNLEASHED_MODEL", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:bridge")
    monkeypatch.setenv("UNLEASHED_API_KEY", "ulk-fake")
    with pytest.raises(ValueError, match="CHAT_BOT_TOKEN"):
        Config.from_env()
    monkeypatch.setenv("CHAT_BOT_TOKEN", "123:bridge")
    with pytest.raises(ValueError, match="должен отличаться"):
        Config.from_env()
    monkeypatch.setenv("CHAT_BOT_TOKEN", "123:chat")
    assert Config.from_env() == Config("123:chat", "ulk-fake", DEFAULT_MODEL)
    monkeypatch.setenv("UNLEASHED_MODEL", "other-model")
    assert Config.from_env().model == "other-model"


def test_dispatches_only_private_or_addressed_group_messages(dispatch):
    prompts = []

    def model_api(request):
        assert str(request.url) == "https://api.unleashedapi.com/v1/chat/completions"
        assert request.method == "POST"
        assert request.headers["Authorization"] == "Bearer ulk-fake"
        payload = json.loads(request.content)
        assert payload["model"] == "selected-model"
        assert payload["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
        prompts.append(payload["messages"][1:])
        return completion("ответ")

    mention = "🙂 @CHAT_BOT привет"
    messages = [
        make_message("я пишу всем", -1, "group"),
        make_message("привет приватно"),
        make_message(mention, -1, "group", entities=[{"type": "mention", "offset": 3, "length": 9}]),
        make_message("ответ боту", -1, "group", reply_to_message=make_message("ответ", -1, "group", **{"from": BOT})),
        make_message("ещё приватно"),
    ]
    assert dispatch(model_api, messages, model="selected-model") == [
        (456, "ответ"), (-1, "ответ"), (-1, "ответ"), (456, "ответ"),
    ]
    answer = {"role": "assistant", "content": "ответ"}
    assert prompts == [
        [{"role": "user", "content": "привет приватно"}],
        [{"role": "user", "content": mention}],
        [{"role": "user", "content": mention}, answer, {"role": "user", "content": "ответ боту"}],
        [{"role": "user", "content": "привет приватно"}, answer, {"role": "user", "content": "ещё приватно"}],
    ]


def test_caps_history_and_splits_reply_at_telegram_limit(dispatch):
    prompts = []
    long_reply = "🙂" * 2049

    def model_api(request):
        prompts.append(json.loads(request.content)["messages"][1:])
        return completion(long_reply if len(prompts) == 8 else "ответ")

    replies = dispatch(model_api, [make_message(f"запрос {i}") for i in range(8)])
    assert replies[-2:] == [(456, "🙂" * 2048), (456, "🙂")]
    assert [message["content"] for message in prompts[-1] if message["role"] == "user"] == [
        f"запрос {i}" for i in range(1, 8)
    ]


@pytest.mark.parametrize(
    ("response", "reply"),
    [
        (httpx.Response(429), "Достигнут лимит запросов. Попробуйте позже."),
        (httpx.Response(503), "Не удалось получить ответ от модели. Попробуйте позже."),
        (completion(None), "Не удалось получить ответ от модели. Попробуйте позже."),
    ],
)
def test_reports_api_failures_without_saving_failed_turn(dispatch, response, reply):
    prompts = []

    def model_api(request):
        prompts.append(json.loads(request.content)["messages"][1:])
        return response if len(prompts) == 1 else completion("ответ")

    assert dispatch(model_api, [make_message("ошибка"), make_message("успех")]) == [
        (456, reply), (456, "ответ"),
    ]
    assert prompts[-1] == [{"role": "user", "content": "успех"}]
