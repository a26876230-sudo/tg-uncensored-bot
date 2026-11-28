import asyncio
import json
import sqlite3

import httpx
import pytest
from telegram import Update
from telegram.request import HTTPXRequest

from bot import (
    DEFAULT_BOT_NAME, DEFAULT_MODEL, SEARCH_NOTE, SUMMARY_PROMPT, SYSTEM_PROMPT,
    ChatBot, ChatMemory, Config, build_application,
)


BOT = {"id": 123, "is_bot": True, "first_name": "Chat", "username": "chat_bot"}
USER = {"id": 456, "is_bot": False, "first_name": "Owner"}


def make_message(text, chat_id=456, chat_type="private", **fields):
    return {
        "message_id": 42, "date": 1, "chat": {"id": chat_id, "type": chat_type},
        "from": USER, "text": text, **fields,
    }


@pytest.fixture
def dispatch(monkeypatch, tmp_path):
    sent = []
    typing_callback = None

    async def telegram_api(self, url, method, request_data=None, **kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        if endpoint == "getMe":
            result = BOT
        elif endpoint == "sendChatAction":
            assert request_data.parameters["action"] == "typing"
            if typing_callback:
                typing_callback()
            result = True
        elif endpoint == "sendMessage":
            params = request_data.parameters
            assert params["link_preview_options"] == {"is_disabled": True}
            sent.append((params["chat_id"], params["text"]))
            result = make_message(params["text"], params["chat_id"], **{"from": BOT})
        else:
            pytest.fail(f"unexpected Telegram call: {endpoint}")
        return 200, json.dumps({"ok": True, "result": result}).encode()

    monkeypatch.setattr(HTTPXRequest, "do_request", telegram_api)

    def run(
        model_api, messages, model=DEFAULT_MODEL, *, search_enabled=False, keenable_key="", name=DEFAULT_BOT_NAME,
        on_typing=None,
    ):
        nonlocal typing_callback
        typing_callback = on_typing
        sent.clear()

        async def process():
            config = Config("123:fake", "ulk-fake", model, search_enabled, keenable_key, name, tmp_path / "chat.sqlite3")
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
    monkeypatch.delenv("KEENABLE_SEARCH_ENABLED", raising=False)
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_BOT_NAME", raising=False)
    monkeypatch.delenv("CHAT_BOT_DB", raising=False)
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
    monkeypatch.setenv("CHAT_BOT_NAME", " Луна ")
    assert Config.from_env().name == "Луна"


def test_search_settings_from_env(monkeypatch):
    monkeypatch.setenv("CHAT_BOT_TOKEN", "123:chat")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:bridge")
    monkeypatch.setenv("UNLEASHED_API_KEY", "ulk-fake")
    monkeypatch.setenv("KEENABLE_SEARCH_ENABLED", "false")
    monkeypatch.setenv("KEENABLE_API_KEY", " keen-fake ")
    config = Config.from_env()
    assert not config.search_enabled
    assert config.keenable_key == "keen-fake"


def test_dispatches_only_private_or_addressed_group_messages(dispatch):
    prompts = []

    def model_api(request):
        assert str(request.url) == "https://api.unleashedapi.com/v1/chat/completions"
        assert request.method == "POST"
        assert request.headers["Authorization"] == "Bearer ulk-fake"
        payload = json.loads(request.content)
        assert payload["model"] == "selected-model"
        assert payload["messages"][0] == {"role": "system", "content": f"{SYSTEM_PROMPT}\nТебя зовут Нори."}
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
        [{"role": "user", "content": "Owner: я пишу всем"}, {"role": "user", "content": f"Owner: {mention}"}],
        [
            {"role": "user", "content": "Owner: я пишу всем"},
            {"role": "user", "content": f"Owner: {mention}"}, answer,
            {"role": "user", "content": "Owner: ответ боту\n\nЦитата из сообщения, на которое отвечают:\nответ"},
        ],
        [{"role": "user", "content": "привет приватно"}, answer, {"role": "user", "content": "ещё приватно"}],
    ]


def test_splits_reply_at_telegram_limit(dispatch):
    prompts = []
    long_reply = "🙂" * 2049

    def model_api(request):
        prompts.append(json.loads(request.content)["messages"][1:])
        return completion(long_reply if len(prompts) == 8 else "ответ")

    replies = dispatch(model_api, [make_message(f"запрос {i}") for i in range(8)])
    assert replies[-2:] == [(456, "🙂" * 2048), (456, "🙂")]
    assert [message["content"] for message in prompts[-1] if message["role"] == "user"] == [
        f"запрос {i}" for i in range(8)
    ]


def test_accepts_anonymous_admin_but_ignores_a_bot(dispatch):
    prompts = []

    def model_api(request):
        prompts.append(json.loads(request.content)["messages"][-1]["content"])
        return completion("ответ")

    group = {"id": -100123, "type": "supergroup", "title": "Группа"}
    anonymous = {"id": 1087968824, "is_bot": True, "first_name": "Group", "username": "GroupAnonymousBot"}
    mention = [{"type": "mention", "offset": 0, "length": 9}]
    messages = [
        make_message("@chat_bot робот", group["id"], group["type"], entities=mention, **{"from": BOT}),
        make_message(
            "@chat_bot админ", group["id"], group["type"],
            entities=mention, sender_chat=group, **{"from": anonymous},
        ),
    ]
    assert dispatch(model_api, messages) == [(group["id"], "ответ")]
    assert prompts == ["Группа: @chat_bot админ"]


def test_name_and_command_trigger_search_only_when_addressed(dispatch):
    searches = []

    def api(request):
        payload = json.loads(request.content)
        if request.url.host == "api.keenable.ai":
            searches.append(payload["query"])
            return httpx.Response(200, json={"results": []})
        assert payload["messages"][0]["content"].endswith("Тебя зовут Луна.")
        return completion("ответ")

    messages = [make_message(text, -1, "group") for text in [
        "Нори, привет", "лунатики и полуна", "Это _луна_ и Луна2", "/nori@other_bot не нам",
        "Ты согласна, ЛУНА?", "луна", "/nori первый вопрос", "/nori@CHAT_BOT второй вопрос",
        "/nori",
    ]]
    assert dispatch(api, messages, search_enabled=True, name="Луна") == [
        (-1, "ответ"), (-1, "ответ"), (-1, "ответ"), (-1, "ответ"), (-1, "Напишите сообщение текстом."),
    ]
    assert searches == ["Ты согласна, ЛУНА?", "первый вопрос", "второй вопрос"]


def test_search_skips_simple_greetings_but_not_a_question_with_a_greeting(dispatch):
    searches = []

    def api(request):
        if request.url.host == "api.keenable.ai":
            searches.append(json.loads(request.content)["query"])
            return httpx.Response(200, json={"results": []})
        return completion("ответ")

    messages = [make_message(text) for text in [
        "Привет, Нори!", "СПАСИБО, @chat_bot!", "Привет, какой курс биткоина сегодня?",
    ]]
    assert dispatch(api, messages, search_enabled=True) == [(456, "ответ")] * 3
    assert searches == ["Привет, какой курс биткоина сегодня?"]


def test_refreshes_typing_while_waiting_for_completion(dispatch, monkeypatch):
    monkeypatch.setattr("bot.TYPING_INTERVAL", 0.001)
    typing_refreshed = asyncio.Event()
    actions = 0

    def on_typing():
        nonlocal actions
        actions += 1
        if actions == 2:
            typing_refreshed.set()

    async def model_api(request):
        await asyncio.wait_for(typing_refreshed.wait(), timeout=1)
        return completion("ответ")

    assert dispatch(model_api, [make_message("Привет!")], on_typing=on_typing) == [(456, "ответ")]
    assert actions >= 2


def test_replies_include_quote_from_other_group_member(dispatch):
    prompts = []

    def model_api(request):
        messages = json.loads(request.content)["messages"]
        prompts.append(messages[1:])
        return completion("ответ")

    original = make_message("Только фрагмент, а не всё.", -1, "group")
    message = make_message(
        "/nori объясни", -1, "group", reply_to_message=original,
        quote={"text": "Только фрагмент", "position": 0},
    )
    assert dispatch(model_api, [message]) == [(-1, "ответ")]
    assert dispatch(model_api, [make_message("Нори, а подробнее?", -1, "group")]) == [(-1, "ответ")]
    assert prompts[0][-1] == {
        "role": "user", "content": "Owner: объясни\n\nЦитата из сообщения, на которое отвечают:\nТолько фрагмент",
    }
    assert prompts[1][:-1] == [prompts[0][-1], {"role": "assistant", "content": "ответ"}]


@pytest.mark.parametrize("second_summary_fails", [False, True])
def test_group_memory_summarizes_only_entries_older_than_the_latest_100(dispatch, second_summary_fails):
    summaries = []

    def model_api(request):
        payload = json.loads(request.content)
        messages = payload["messages"]
        if messages[0]["content"] == SUMMARY_PROMPT:
            summaries.append(json.loads(messages[1]["content"]))
            assert payload["max_tokens"] == 500
            if len(summaries) == 2 and second_summary_fails:
                return httpx.Response(503)
            return completion(f"пересказ {len(summaries)}")
        assert len(summaries) == 2
        expected_number = 1 if second_summary_fails else 2
        assert messages[1]["role"] == "system"
        assert messages[1]["content"].endswith(f'"пересказ {expected_number}"')
        assert messages[2:-1] == [
            {"role": "user", "content": f"Owner: запись {i}"} for i in range(2, 102)
        ]
        assert messages[-1]["content"] == "Owner: Нори, что я говорил?"
        return completion("ответ")

    messages = [make_message(f"запись {i}", -1, "group") for i in range(102)]
    assert dispatch(model_api, [*messages, make_message("Нори, что я говорил?", -1, "group")]) == [(-1, "ответ")]
    assert summaries == [
        {"previous_summary": previous, "messages": [
            {"role": "user", "content": f"Owner: запись {i}"} for i in indices
        ]}
        for previous, indices in (
            ("", [0]), ("пересказ 1", [1]), ("пересказ 1" if second_summary_fails else "пересказ 2", [2, 3]),
        )
    ]


def test_failed_sqlite_write_preserves_history_for_next_summary(tmp_path):
    summaries = []

    def model_api(request):
        payload = json.loads(request.content)["messages"]
        assert payload[0]["content"] == SUMMARY_PROMPT
        summaries.append(json.loads(payload[1]["content"])["messages"])
        return completion("пересказ")

    async def run():
        config = Config("123:fake", "ulk-fake", db_path=tmp_path / "chat.sqlite3")
        async with httpx.AsyncClient(transport=httpx.MockTransport(model_api)) as http:
            chatbot = ChatBot(config, http)
            try:
                memory = ChatMemory()
                entries = [{"role": "user", "content": f"запись {i}"} for i in range(101)]
                for entry in entries[:-1]:
                    await chatbot.remember(-1, memory, entry)
                db = chatbot.store.db
                db.execute("""
                    CREATE TRIGGER fail_write BEFORE INSERT ON chat_memory
                    BEGIN SELECT RAISE(ABORT, 'simulated write failure'); END
                """)
                with pytest.raises(sqlite3.IntegrityError, match="simulated write failure"):
                    await chatbot.remember(-1, memory, entries[-1])
                assert list(memory.recent) == entries[:-1]
                assert chatbot.store.load()[-1] == memory
                assert not summaries

                db.execute("DROP TRIGGER fail_write")
                await chatbot.remember(-1, memory, entries[-1])
                assert summaries == [entries[:1]]
                assert list(memory.recent) == entries[1:]
                assert memory.summary == "пересказ"
                assert chatbot.store.load()[-1] == memory
            finally:
                chatbot.store.close()

    asyncio.run(run())


def test_recreated_bot_loads_recent_messages_and_summary_for_the_same_chat(dispatch):
    def summary_api(request):
        payload = json.loads(request.content)["messages"]
        assert payload[0]["content"] == SUMMARY_PROMPT
        assert json.loads(payload[1]["content"])["messages"] == [{"role": "user", "content": "Owner: запись 0"}]
        return completion("Старые записи")

    messages = [make_message(f"запись {i}", -1, "group") for i in range(101)]
    messages.append(make_message("другой разговор", -2, "group"))
    assert dispatch(summary_api, messages) == []

    prompts = []

    def reply_api(request):
        messages = json.loads(request.content)["messages"]
        if messages[0]["content"] == SUMMARY_PROMPT:
            return completion("Обновлённый пересказ")
        prompts.append(messages)
        return completion("помню")

    assert dispatch(reply_api, [
        make_message("Нори, вспомни", -1, "group"),
        make_message("Нори, вспомни", -2, "group"),
    ]) == [(-1, "помню"), (-2, "помню")]
    assert prompts[0][1]["role"] == "system"
    assert prompts[0][1]["content"].endswith('"Старые записи"')
    assert prompts[0][2:-1] == [{"role": "user", "content": f"Owner: запись {i}"} for i in range(1, 101)]
    assert prompts[1][1:-1] == [{"role": "user", "content": "Owner: другой разговор"}]


@pytest.mark.parametrize("key", ["", "keen-fake"])
def test_search_sources_reach_model_but_do_not_enter_history(dispatch, key):
    calls = []
    source = {"title": "Источник", "url": "https://example.com/price", "snippet": "Цена биткоина"}
    second_source = {"title": "Без выдержки", "url": "https://example.com/no-snippet"}

    def api(request):
        payload = json.loads(request.content)
        assert request.method == "POST"
        if request.url.host == "api.keenable.ai":
            calls.append(("search", payload["query"]))
            assert request.extensions["timeout"]["read"] == 4
            assert payload["max_results"] == 5
            assert "Authorization" not in request.headers
            if key:
                assert request.url.path == "/v1/search"
                assert request.headers["X-API-Key"] == key
            else:
                assert request.url.path == "/v1/search/public"
                assert request.headers["X-Keenable-Title"] == "tg-uncensored-bot"
                assert "X-API-Key" not in request.headers
            return httpx.Response(200, json={"results": [source, None, second_source]})
        assert request.url.host == "api.unleashedapi.com"
        assert "X-API-Key" not in request.headers
        assert payload["max_tokens"] == 1200
        assert payload["reasoning_effort"] == "none"
        messages = payload["messages"]
        assert messages[0]["role"] == "system"
        assert SEARCH_NOTE in messages[0]["content"]
        query, sources = messages[-1]["content"].split("\n\nРезультаты веб-поиска (JSON):\n")
        calls.append(("complete", query))
        assert json.loads(sources) == [source, {**second_source, "snippet": ""}]
        if query == "ещё вопрос":
            assert messages[1:-1] == [
                {"role": "user", "content": "курс биткоина"},
                {"role": "assistant", "content": "ответ"},
            ]
        return completion("ответ")

    messages = [make_message("чат", -1, "group"), make_message("курс биткоина"), make_message("ещё вопрос")]
    assert dispatch(api, messages, search_enabled=True, keenable_key=key) == [(456, "ответ")] * 2
    assert calls == [
        ("search", "курс биткоина"), ("complete", "курс биткоина"),
        ("search", "ещё вопрос"), ("complete", "ещё вопрос"),
    ]


def test_model_reply_does_not_get_a_sources_block_or_link_preview(dispatch):
    source = {"title": "Источник", "url": "https://example.com/source", "snippet": "выдержка"}
    text = "Ответ модели с просочившейся ссылкой https://example.net/leak"

    def api(request):
        if request.url.host == "api.keenable.ai":
            return httpx.Response(200, json={"results": [source]})
        messages = json.loads(request.content)["messages"]
        assert "ссылки в чат не отправляй" in messages[0]["content"]
        assert "не отправляй ссылки и списки источников" in messages[0]["content"]
        assert source["url"] in messages[-1]["content"]
        return completion(text)

    assert dispatch(api, [make_message("вопрос")], search_enabled=True) == [(456, text)]


@pytest.mark.parametrize("failure", [429, "timeout", "invalid JSON"])
def test_search_failures_still_answer_without_sources(dispatch, failure):
    def api(request):
        if request.url.host == "api.keenable.ai":
            if failure == "timeout":
                raise httpx.ReadTimeout("timeout", request=request)
            if failure == "invalid JSON":
                return httpx.Response(200, text="not JSON")
            return httpx.Response(failure)
        assert json.loads(request.content)["messages"] == [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\nТебя зовут Нори."},
            {"role": "user", "content": "вопрос"},
        ]
        return completion("ответ без поиска")

    assert dispatch(api, [make_message("вопрос")], search_enabled=True) == [(456, "ответ без поиска")]


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
