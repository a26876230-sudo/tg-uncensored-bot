import asyncio
import json
import re
import sqlite3
from types import SimpleNamespace

import httpx
import pytest
from telegram import Update
from telegram.request import HTTPXRequest

from bot import (
    ANSWER_BOUNDARY_PROMPT, BOUNDARY_OVERLOAD, BOUNDARY_PROMPT, BOUNDARY_REFUSAL,
    CHIME_CHAT_COOLDOWN, CHIME_GLOBAL_COOLDOWN, CHIME_MIN_MESSAGES, CHIME_PROMPT,
    DEFAULT_BOT_NAME, DEFAULT_MODEL, SEARCH_NOTE, SUMMARY_PROMPT, SYSTEM_PROMPT,
    ChatBot, ChatMemory, Config, build_application,
)
from nori_media import VISION_MODEL


BOT = {"id": 123, "is_bot": True, "first_name": "Chat", "username": "chat_bot"}
USER = {"id": 456, "is_bot": False, "first_name": "Owner"}


def make_message(text, chat_id=456, chat_type="private", **fields):
    return {
        "message_id": 42, "date": 1, "chat": {"id": chat_id, "type": chat_type},
        "from": USER, "text": text, **fields,
    }


@pytest.fixture
def dispatch(monkeypatch, tmp_path):
    monkeypatch.setattr("bot.random.random", lambda: 1.0)
    sent = []
    typing_callback = None
    download_callback = None
    telegram_downloads = {
        "voice-file": b"OggS synthetic voice",
        "large-photo": b"synthetic image",
    }

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
        elif endpoint == "getFile":
            file_id = request_data.parameters["file_id"]
            if download_callback:
                download_callback(file_id)
            result = {"file_id": file_id, "file_unique_id": file_id, "file_path": file_id}
        elif endpoint in telegram_downloads:
            return 200, telegram_downloads[endpoint]
        elif endpoint == "sendVoice":
            params = request_data.parameters
            upload = request_data.multipart_data["voice"]
            assert upload[:2] == ("nori.mp3", b"synthetic mp3")
            sent.append((params["chat_id"], "<voice>"))
            result = make_message("", params["chat_id"], **{"from": BOT})
        else:
            pytest.fail(f"unexpected Telegram call: {endpoint}")
        return 200, json.dumps({"ok": True, "result": result}).encode()

    monkeypatch.setattr(HTTPXRequest, "do_request", telegram_api)

    def run(
        model_api, messages, model=DEFAULT_MODEL, *, search_enabled=False, keenable_key="", name=DEFAULT_BOT_NAME,
        on_typing=None, boundary_api=None, groq_key="", on_download=None,
    ):
        nonlocal typing_callback, download_callback
        typing_callback = on_typing
        download_callback = on_download
        sent.clear()

        async def process():
            config = Config(
                "123:fake", "ulk-fake", model, search_enabled, keenable_key, name,
                tmp_path / "chat.sqlite3", groq_key=groq_key,
            )
            model_transport = httpx.MockTransport(model_api)
            boundary_transport = httpx.MockTransport(boundary_api or (lambda request: completion("НЕТ ПРОСТО")))

            async def route(request):
                if request.url.host == "api.unleashedapi.com":
                    first = json.loads(request.content)["messages"][0]["content"]
                    if first in {BOUNDARY_PROMPT, ANSWER_BOUNDARY_PROMPT}:
                        return await boundary_transport.handle_async_request(request)
                return await model_transport.handle_async_request(request)

            application = build_application(config, httpx.MockTransport(route))
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
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("CHAT_BOT_VISION_MODEL", raising=False)
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
    monkeypatch.setenv("GROQ_API_KEY", " groq-fake ")
    monkeypatch.setenv("CHAT_BOT_VISION_MODEL", " vision-model ")
    assert Config.from_env().groq_key == "groq-fake"
    assert Config.from_env().vision_model == "vision-model"


def telegram_voice(duration=3):
    return {"file_id": "voice-file", "file_unique_id": "voice", "duration": duration}


PHOTOS = [
    {"file_id": "small-photo", "file_unique_id": "small", "width": 75, "height": 75},
    {"file_id": "large-photo", "file_unique_id": "large", "width": 640, "height": 640},
]


def test_addressed_voice_is_transcribed_then_answered_with_telegram_voice(dispatch, monkeypatch):
    requests = []
    synthesized = []

    async def synthesize(self, text):
        synthesized.append(text)
        return b"synthetic mp3"

    def api(request):
        requests.append(request.url.host)
        if request.url.host == "api.groq.com":
            assert request.url.path == "/openai/v1/audio/transcriptions"
            assert request.headers["Authorization"] == "Bearer groq-fake"
            assert b"OggS synthetic voice" in request.content
            assert b"whisper-large-v3" in request.content
            return httpx.Response(200, json={"text": "Сколько будет два плюс два?"})
        assert "Сколько будет два плюс два?" in json.loads(request.content)["messages"][-1]["content"]
        return completion("Четыре.")

    monkeypatch.setattr("nori_media.VoiceSynthesizer.synthesize", synthesize)
    replies = dispatch(
        api, [make_message(None, voice=telegram_voice())], groq_key="groq-fake", search_enabled=True,
    )
    assert replies == [(456, "<voice>")]
    assert requests == ["api.groq.com", "api.unleashedapi.com"]
    assert synthesized == ["Четыре."]


def test_reply_to_noris_voice_follows_the_normal_text_search_path(dispatch):
    searches = []

    def api(request):
        if request.url.host == "api.keenable.ai":
            searches.append(json.loads(request.content)["query"])
            return httpx.Response(200, json={"results": []})
        return completion("Ответ с поиском.")

    assert dispatch(api, [
        make_message(
            "Какая погода сейчас?", -1, "group",
            reply_to_message=make_message(None, -1, "group", voice=telegram_voice(), **{"from": BOT}),
        ),
    ], search_enabled=True, on_download=lambda _: pytest.fail("не скачивай голосовое Нори")) == [
        (-1, "Ответ с поиском."),
    ]
    assert searches == ["Какая погода сейчас?"]


def test_group_photo_is_understood_only_when_addressed_or_quoted(dispatch):
    downloads = []
    prompts = []
    caption = "Нори, что за мем?"
    photo = make_message(None, -1, "group", photo=PHOTOS)

    def api(request):
        payload = json.loads(request.content)
        if request.url.host == "api.groq.com":
            assert request.url.path == "/openai/v1/chat/completions"
            assert payload["model"] == VISION_MODEL
            assert payload["messages"][0]["content"][1] == {
                "type": "image_url", "image_url": {"url": "data:image/jpeg;base64,c3ludGhldGljIGltYWdl"},
            }
            return completion('Кот с подписью «понедельник».')
        prompts.append(payload["messages"])
        return completion("Это мем про понедельник.")

    replies = dispatch(api, [
        photo,
        make_message(None, -1, "group", voice=telegram_voice()),
        make_message(None, -1, "group", photo=PHOTOS, caption=caption),
        make_message("Нори, объясни", -1, "group", reply_to_message=photo),
        make_message("Нори, привет", -1, "group"),
    ], groq_key="groq-fake", on_download=downloads.append)
    assert replies == [(-1, "Это мем про понедельник.")] * 3
    assert downloads == ["large-photo", "large-photo"]
    assert caption in prompts[0][-1]["content"]
    assert all("Кот с подписью" in prompt[-1]["content"] for prompt in prompts[:2])
    assert all("Кот с подписью" not in message["content"] for message in prompts[-1])


def test_large_voice_is_rejected_without_download_or_model_call(dispatch):
    def unexpected(request):
        pytest.fail(f"unexpected provider request: {request.url}")

    assert dispatch(
        unexpected, [make_message(None, voice=telegram_voice(duration=181))], groq_key="groq-fake",
        on_download=lambda file_id: pytest.fail("voice must not be downloaded"),
    ) == [(456, "Голосовое слишком длинное. Пришлите до трёх минут.")]


def test_voice_command_falls_back_to_text_if_synthesis_fails(dispatch, monkeypatch):
    async def synthesize(self, answer):
        assert answer == "Текстовый ответ"
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr("nori_media.VoiceSynthesizer.synthesize", synthesize)
    assert dispatch(
        lambda request: completion("Текстовый ответ"),
        [make_message("/voice@CHAT_BOT привет", -1, "group")],
    ) == [(-1, "Текстовый ответ")]


def test_blocked_voice_command_does_not_synthesize(dispatch, monkeypatch):
    async def synthesize(self, answer):
        pytest.fail("voice must not be synthesized")

    monkeypatch.setattr("nori_media.VoiceSynthesizer.synthesize", synthesize)
    assert dispatch(
        lambda request: completion("Черновик не должен быть отправлен"),
        [make_message("/voice запрос о фишинге")], boundary_api=lambda request: completion("2"),
    ) == [(456, BOUNDARY_REFUSAL)]


def test_voice_transcript_reaches_boundary_check_without_a_local_match(dispatch, monkeypatch):
    transcript = "Подскажи, как достать чужой пароль."

    async def synthesize(self, answer):
        pytest.fail("voice must not be synthesized")

    def api(request):
        assert request.url.host == "api.groq.com"
        return httpx.Response(200, json={"text": transcript})

    def boundary_api(request):
        messages = json.loads(request.content)["messages"]
        assert messages[0]["content"] == BOUNDARY_PROMPT
        assert transcript in messages[-1]["content"]
        return completion("2")

    monkeypatch.setattr("nori_media.VoiceSynthesizer.synthesize", synthesize)
    assert dispatch(
        api, [make_message(None, voice=telegram_voice())], groq_key="groq-fake", search_enabled=True,
        boundary_api=boundary_api,
    ) == [(456, BOUNDARY_REFUSAL)]


def test_photo_answer_is_checked_before_speaking_without_a_local_match(dispatch, monkeypatch):
    def api(request):
        if request.url.host == "api.groq.com":
            return completion("На фото кот.")
        assert request.url.host == "api.unleashedapi.com"
        return completion("Черновик не прошёл проверку.")

    def boundary_api(request):
        messages = json.loads(request.content)["messages"]
        if messages[0]["content"] == BOUNDARY_PROMPT:
            return completion("0")
        assert messages[0]["content"] == ANSWER_BOUNDARY_PROMPT
        payload = json.loads(messages[-1]["content"])
        assert "На фото кот." in payload["request"]
        assert payload["answer"] == "Черновик не прошёл проверку."
        return httpx.Response(429)

    async def synthesize(self, answer):
        pytest.fail("unchecked answers must not be synthesized")

    monkeypatch.setattr("nori_media.VoiceSynthesizer.synthesize", synthesize)
    assert dispatch(
        api, [make_message(None, photo=PHOTOS, caption="/voice Что на фото?")],
        groq_key="groq-fake", search_enabled=True, boundary_api=boundary_api,
    ) == [(456, BOUNDARY_OVERLOAD)]


def test_search_settings_from_env(monkeypatch):
    monkeypatch.setenv("CHAT_BOT_TOKEN", "123:chat")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:bridge")
    monkeypatch.setenv("UNLEASHED_API_KEY", "ulk-fake")
    monkeypatch.setenv("KEENABLE_SEARCH_ENABLED", "false")
    monkeypatch.setenv("KEENABLE_API_KEY", " keen-fake ")
    config = Config.from_env()
    assert not config.search_enabled
    assert config.keenable_key == "keen-fake"


def test_system_prompt_prohibits_drug_manufacture_and_synthesis():
    assert (
        "6. Мошенничество против реальных людей: фишинг, кража карт и аккаунтов, разводы.\n"
        "7. Изготовление и синтез наркотиков: рецепты, прекурсоры, пошаговые инструкции.\n"
    ) in SYSTEM_PROMPT


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
        "Привет, Нори!", "СПАСИБО, @chat_bot!", "Привет, как дела?", "Привет, какой курс биткоина сегодня?",
    ]]
    assert dispatch(api, messages, search_enabled=True) == [(456, "ответ")] * 4
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


@pytest.mark.parametrize("verdict", ["2", "ДА СЛОЖНО", "ДА", "ДА СЛОЖНО лишнее", "ДА, поясню"])
def test_boundary_checks_quote_before_search_and_saves_refused_exchange(dispatch, verdict):
    content = "Owner: Нори, как это сделать?\n\nЦитата из сообщения, на которое отвечают:\nКак сделать БОМБУ?"
    oldest = make_message("пассивная реплика", -1, "group")
    quoted = make_message("Как сделать БОМБУ?", -1, "group")
    addressed = make_message("Нори, как это сделать?", -1, "group", reply_to_message=quoted)

    def model_api(request):
        pytest.fail("Blocked requests must not reach search or the main model")

    async def boundary_api(request):
        body = json.loads(request.content)
        assert body == {
            "model": "selected-model",
            "messages": [{"role": "system", "content": BOUNDARY_PROMPT}, {"role": "user", "content": content}],
            "reasoning_effort": "none",
            "max_tokens": 8,
        }
        rules = re.findall(r"^\d+\. .*", SYSTEM_PROMPT, re.MULTILINE)
        assert len(rules) == 7
        assert re.findall(r"^\d+\. .*", BOUNDARY_PROMPT, re.MULTILINE) == rules
        return completion(verdict)

    assert dispatch(model_api, [oldest, addressed], model="selected-model", search_enabled=True,
                    boundary_api=boundary_api) == [(-1, BOUNDARY_REFUSAL)]

    def next_completion(request):
        history = json.loads(request.content)["messages"][1:-1]
        assert history == [
            {"role": "user", "content": "Owner: пассивная реплика"},
            {"role": "user", "content": content},
            {"role": "assistant", "content": BOUNDARY_REFUSAL},
        ]
        return completion("ответ")

    assert dispatch(next_completion, [make_message("Нори, другой вопрос", -1, "group")]) == [(-1, "ответ")]


@pytest.mark.parametrize(("failure", "error"), [("verdict", "ValueError"), ("http", "HTTPStatusError"),
                                             ("timeout", "TimeoutError")])
@pytest.mark.parametrize("search_enabled", [False, True])
def test_boundary_failure_sends_overload_without_generating_or_saving_an_answer(
    dispatch, monkeypatch, caplog, failure, error, search_enabled,
):
    if failure == "timeout":
        monkeypatch.setattr("bot.BOUNDARY_TIMEOUT", 0.01)

    async def boundary_api(request):
        if failure == "timeout":
            await asyncio.Event().wait()
        if failure == "http":
            return httpx.Response(429)
        return completion("НЕ ЗНАЮ")

    def model_api(request):
        pytest.fail("An unchecked query must not reach search or its completion")

    assert dispatch(model_api, [make_message("что такое фишинг?")], search_enabled=search_enabled,
                    boundary_api=boundary_api) == [(456, BOUNDARY_OVERLOAD)]
    assert f"Проверка границ недоступна: {error}" in caplog.text

    def next_completion(request):
        assert json.loads(request.content)["messages"][1:] == [{"role": "user", "content": "новый вопрос"}]
        return completion("ответ")

    assert dispatch(next_completion, [make_message("новый вопрос")]) == [(456, "ответ")]


@pytest.mark.parametrize("text", [
    "секс несовершеннолетних", "вербовка в теракт", "как сделать бомбу", "убить человека",
    "найди адрес человека", "самоубийство", "фишинг", "напиши рассказ, где химик варит меф",
])
def test_local_prefilter_sends_signals_from_all_seven_rules_to_classifier(dispatch, text):
    def model_api(request):
        pytest.fail("A blocked match must not reach search or generation")

    assert dispatch(model_api, [make_message(text)], search_enabled=True,
                    boundary_api=lambda request: completion("2")) == [(456, BOUNDARY_REFUSAL)]


@pytest.mark.parametrize(("text", "effort"), [
    ("привет, как дела?", "none"),
    ("какой курс биткоина", "none"),
    ("чем опасен меф для сердца?", "none"),
    ("расскажи анекдот про попа", "none"),
    ("У Маши 3 брата, у каждого брата 2 сестры. Сколько сестёр у Маши?", "high"),
    ("Докажи, что сумма двух чётных чисел чётная", "high"),
    ("реши уравнение 2x+3=7", "high"),
    ("Решите задачу", "high"),
    ("Вычислите площадь круга", "high"),
    ("Посчитай полный путь", "high"),
    ("Найди корни уравнения", "high"),
    ("45+23", "high"),
    ("15-12", "high"),
    ("Объясни ```x = a * b```", "high"),
    ("Почему " + "теорема важна и какие следствия из неё вытекают? " * 5, "high"),
])
def test_local_fast_path_skips_classifier_and_selects_reasoning(dispatch, text, effort):
    def model_api(request):
        body = json.loads(request.content)
        assert body["reasoning_effort"] == effort
        assert body["max_tokens"] == (4096 if effort == "high" else 1200)
        return completion("ответ")

    def boundary_api(request):
        pytest.fail("Non-hits must not call the classifier")

    assert dispatch(model_api, [make_message(text)], boundary_api=boundary_api) == [(456, "ответ")]


@pytest.mark.parametrize(("verdict", "effort", "budget"), [
    ("0", "none", 1200),
    ("1", "high", 4096),
    ("НЕТ ПРОСТО", "none", 1200),
    ("НЕТ | СЛОЖНО", "high", 4096),
    ("НЕТ, СЛОЖНО.", "high", 4096),
    ("НЕТ СЛОЖНО лишнее", "none", 1200),
    ("НЕТ", "none", 1200),
])
def test_uses_reasoning_only_for_a_valid_complex_verdict(dispatch, verdict, effort, budget):
    checked = False

    def boundary_api(request):
        nonlocal checked
        checked = True
        return completion(verdict)

    def model_api(request):
        assert checked
        if request.url.host == "api.keenable.ai":
            return httpx.Response(200, json={"results": []})
        body = json.loads(request.content)
        assert body["reasoning_effort"] == effort
        assert body["max_tokens"] == budget
        return httpx.Response(200, json={"choices": [{
            "message": {"content": "ответ", "reasoning_content": "внутренний текст"},
            "finish_reason": "stop",
        }]})

    assert dispatch(model_api, [make_message("вопрос о фишинге")], search_enabled=True,
                    boundary_api=boundary_api) == [(456, "ответ")]


def test_passive_group_message_never_calls_boundary_classifier(dispatch):
    def unexpected(request):
        pytest.fail("Passive group messages must not call external services")

    assert dispatch(unexpected, [make_message("почему опасен фишинг?", -1, "group")],
                    search_enabled=True, boundary_api=unexpected) == []


@pytest.mark.parametrize("answer", [
    "В обсуждении упоминается псевдоэфедрин.",
    "В списке упоминается нитрат аммония.",
    "Смешай 25 мл жидкости.",
    "Список: ацетон и соляная кислота.",
    "Улица Тестовая, 42.",
    "Пример номера: +7 (000) 000-00-00.",
])
def test_output_signals_trigger_review_with_question_before_sending(dispatch, answer):
    def boundary_api(request):
        messages = json.loads(request.content)["messages"]
        assert messages[0]["content"] == ANSWER_BOUNDARY_PROMPT
        assert json.loads(messages[1]["content"]) == {"request": "привет", "answer": answer}
        return completion("0")

    assert dispatch(lambda request: completion(answer), [make_message("привет")],
                    boundary_api=boundary_api) == [(456, answer)]


@pytest.mark.parametrize(("verdict", "expected"), [
    ("2", BOUNDARY_REFUSAL), ("1", BOUNDARY_OVERLOAD), (None, BOUNDARY_OVERLOAD),
])
def test_output_review_withholds_the_draft_on_block_or_failure(dispatch, verdict, expected):
    question = "расскажи подробнее"
    answer = "В черновике упоминается гексоген."

    def boundary_api(request):
        assert json.loads(request.content)["messages"][0]["content"] == ANSWER_BOUNDARY_PROMPT
        return completion(verdict) if verdict is not None else httpx.Response(429)

    assert dispatch(lambda request: completion(answer), [make_message(question)],
                    boundary_api=boundary_api) == [(456, expected)]

    def next_completion(request):
        messages = json.loads(request.content)["messages"][1:]
        if expected == BOUNDARY_REFUSAL:
            assert messages[:-1] == [
                {"role": "user", "content": question}, {"role": "assistant", "content": BOUNDARY_REFUSAL},
            ]
        else:
            assert len(messages) == 1
        return completion("ответ")

    assert dispatch(next_completion, [make_message("другой вопрос")]) == [(456, "ответ")]


def test_unleashed_request_slots_are_shared_and_waiting_does_not_consume_check_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr("bot.BOUNDARY_TIMEOUT", 0.01)

    async def run():
        started = asyncio.Event()
        release = asyncio.Event()
        prompts = []
        latest = {"role": "user", "content": "вопрос"}

        async def api(request):
            prompt = json.loads(request.content)["messages"][0]["content"]
            prompts.append(prompt)
            if len(prompts) == 2:
                started.set()
            if prompt == BOUNDARY_PROMPT:
                return completion("0")
            await release.wait()
            return completion("ответ")

        async with httpx.AsyncClient(transport=httpx.MockTransport(api)) as http:
            first = ChatBot(Config("123:fake", "ulk-fake", db_path=tmp_path / "first.db"), http)
            second = ChatBot(Config("123:fake", "ulk-fake", db_path=tmp_path / "second.db"), http)
            try:
                async with asyncio.TaskGroup() as tasks:
                    tasks.create_task(first.complete(ChatMemory(), latest, []))
                    tasks.create_task(second.remember(1, ChatMemory(), *([latest] * 101)))
                    await asyncio.wait_for(started.wait(), timeout=1)
                    check = tasks.create_task(first.classify_message(latest))
                    try:
                        await asyncio.sleep(0.03)
                        assert len(prompts) == 2
                        assert not check.done()
                    finally:
                        release.set()
                assert prompts[-1] == BOUNDARY_PROMPT
                assert check.result() == (False, False)
            finally:
                first.store.close()
                second.store.close()

    asyncio.run(run())


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


@pytest.mark.parametrize(("finish_reason", "expected"), [("length", "Ответ …"), ("stop", "Ответ")])
def test_marks_replies_truncated_by_token_limit(dispatch, finish_reason, expected):
    def model_api(request):
        return httpx.Response(200, json={"choices": [{
            "message": {"content": "Ответ "}, "finish_reason": finish_reason,
        }]})

    assert dispatch(model_api, [make_message("вопрос")]) == [(456, expected)]


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


def test_people_memory_learns_locally_and_recall_is_scoped_and_persistent(dispatch):
    same_name = {**USER, "id": 789}

    def no_completion(request):
        pytest.fail("learning a fact shouldn't call the model")

    assert dispatch(no_completion, [
        make_message("Меня зовут Лена. Работаю дизайнером. Люблю джаз.", -1, "group"),
        make_message("Люблю рок.", -1, "group", **{"from": same_name}),
        make_message("Работаю учителем.", -2, "group"),
        make_message(
            "Это цитата, а не мои интересы.", -1, "group",
            reply_to_message=make_message("Люблю готовить.", -1, "group", **{"from": same_name}),
        ),
    ]) == []
    prompts = []

    def reply_api(request):
        messages = json.loads(request.content)["messages"]
        notes = next(msg["content"] for msg in messages if msg["content"].startswith("Заметки об участниках"))
        people = json.loads(notes.split("\n", 1)[1])
        prompts.append(people)
        return completion("помню")

    assert dispatch(reply_api, [
        make_message("Нори, вспомни", -1, "group"),
        make_message("Нори, вспомни", -1, "group", **{"from": same_name}),
        make_message("Нори, вспомни", -2, "group"),
    ]) == [(-1, "помню"), (-1, "помню"), (-2, "помню")]
    assert [(profile["имя"], profile["интересы"]) for profile in prompts[0]] == [
        ("Лена", "джаз"), ("Owner", "рок"),
    ]
    assert prompts[0][0]["занятие"] == "дизайнером"
    assert prompts[0][0]["автор нового сообщения"] == "да"
    assert prompts[1][0]["интересы"] == "рок"
    assert prompts[1][0]["автор нового сообщения"] == "да"
    assert prompts[2] == [{
        "имя": "Owner", "занятие": "учителем", "имя в Telegram": "Owner", "автор нового сообщения": "да",
    }]


@pytest.mark.parametrize("source", ["forward", "blockquote", "expandable_blockquote_caption"])
def test_people_memory_does_not_attribute_forwarded_or_quoted_facts_to_sender(dispatch, source):
    quote = "Люблю рок."
    if source == "forward":
        message = make_message(quote, -1, "group", forward_origin={
            "type": "user", "date": 1, "sender_user": {**USER, "id": 789},
        })
    else:
        prefix = "😀. "
        body = prefix + quote + "\nРаботаю врачом."
        entity = {"type": source.split("_caption")[0], "offset": 4, "length": len(quote)}
        if source.endswith("_caption"):
            message = make_message(None, -1, "group", caption=body, caption_entities=[entity])
        else:
            message = make_message(body, -1, "group", entities=[entity])

    def model_api(request):
        messages = json.loads(request.content)["messages"]
        notes = next(msg["content"] for msg in messages if msg["content"].startswith("Заметки об участниках"))
        person, = json.loads(notes.split("\n", 1)[1])
        assert person["интересы"] == "джаз"
        if source != "forward":
            assert person["занятие"] == "врачом"
        return completion("помню")

    assert dispatch(model_api, [
        make_message("Люблю джаз.", -1, "group"), message, make_message("Нори, вспомни", -1, "group"),
    ]) == [(-1, "помню")]


@pytest.mark.parametrize("question", [
    "Нори, люблю ли я рок?",
    "Люблю рок?",
    "Люблю ли я рок",
    "Меня зовут Маша, да?",
    "Работаю поваром?",
])
def test_people_memory_questions_do_not_overwrite_facts(dispatch, question):
    def model_api(request):
        messages = json.loads(request.content)["messages"]
        notes = next(msg["content"] for msg in messages if msg["content"].startswith("Заметки об участниках"))
        person, = json.loads(notes.split("\n", 1)[1])
        assert (person["имя"], person["занятие"], person["интересы"]) == ("Лена", "врачом", "джаз")
        return completion("помню")

    assert dispatch(model_api, [
        make_message("Меня зовут Лена. Работаю врачом. Люблю джаз.", -1, "group"),
        make_message(question, -1, "group"),
        make_message("Нори, вспомни", -1, "group"),
    ])[-1:] == [(-1, "помню")]


@pytest.mark.parametrize("address", ["@CHAT_BOT, ", "Нори — "])
def test_people_memory_learns_after_a_leading_bot_address(dispatch, address):
    def model_api(request):
        messages = json.loads(request.content)["messages"]
        notes = next(msg["content"] for msg in messages if msg["content"].startswith("Заметки об участниках"))
        person, = json.loads(notes.split("\n", 1)[1])
        assert person["имя"] == "Лена"
        return completion("помню")

    assert dispatch(model_api, [
        make_message(f"{address}меня зовут Лена.", -1, "group"),
        make_message("Нори, вспомни", -1, "group"),
    ])[-1:] == [(-1, "помню")]


CHIME_TEXT = "Этот новый фильм про далёкий космос мне определённо понравился!"


def test_chime_in_requires_topic_roll_message_count_and_cooldowns(dispatch, monkeypatch):
    clock = 0
    roll = 0
    monkeypatch.setattr("bot.time", SimpleNamespace(monotonic=lambda: clock))
    monkeypatch.setattr("bot.random.random", lambda: roll)
    checked = []

    def check(request):
        checked.append(json.loads(request.content)["messages"][-1]["content"])
        return completion("НЕТ ПРОСТО")

    def reply(request):
        assert request.url.host == "api.unleashedapi.com"
        payload = json.loads(request.content)
        assert CHIME_PROMPT in payload["messages"][0]["content"]
        assert payload["max_tokens"] == 120
        return completion("Хорошее кино!")

    def messages():
        nonlocal clock, roll
        yield make_message(CHIME_TEXT, -1, "group")
        for _ in range(CHIME_MIN_MESSAGES - 1):
            yield make_message("Договорились, принесу этот большой пакет с продуктами завтра.", -1, "group")
        yield make_message("Хороший фильм", -1, "group")
        roll = 1
        yield make_message(CHIME_TEXT, -1, "group")
        roll = 0
        yield make_message(CHIME_TEXT, -1, "group")
        yield make_message(CHIME_TEXT, -1, "group")
        for _ in range(CHIME_MIN_MESSAGES):
            yield make_message("обычное сообщение", -1, "group")
            yield make_message("обычное сообщение", -2, "group")
        yield make_message(CHIME_TEXT, -2, "group")
        clock += CHIME_GLOBAL_COOLDOWN
        yield make_message(CHIME_TEXT, -1, "group")
        yield make_message(CHIME_TEXT, -2, "group")
        clock = CHIME_CHAT_COOLDOWN
        yield make_message(CHIME_TEXT, -1, "group")

    assert dispatch(reply, messages(), search_enabled=True, boundary_api=check) == [
        (-1, "Хорошее кино!"), (-2, "Хорошее кино!"), (-1, "Хорошее кино!"),
    ]
    assert checked == [f"Owner: {CHIME_TEXT}"] * 3


@pytest.mark.parametrize("outcome", ["blocked", "invalid", "classifier 429", "reply 429"])
def test_chime_in_stays_silent_on_boundary_or_model_failure(dispatch, monkeypatch, outcome):
    clock = 0
    monkeypatch.setattr("bot.time", SimpleNamespace(monotonic=lambda: clock))
    monkeypatch.setattr("bot.random.random", lambda: 0)
    checks, replies = [], []

    def check(request):
        content = json.loads(request.content)["messages"][-1]["content"]
        assert CHIME_TEXT in content
        assert "текст цитаты" in content
        checks.append(content)
        if outcome == "classifier 429":
            return httpx.Response(429)
        return completion({"blocked": "ДА ПРОСТО", "invalid": "ошибка", "reply 429": "НЕТ ПРОСТО"}[outcome])

    def reply(request):
        replies.append(request)
        return httpx.Response(429)

    def messages():
        nonlocal clock
        candidate = make_message(
            CHIME_TEXT, -1, "group", reply_to_message=make_message("текст цитаты", -1, "group"),
        )
        for _ in range(CHIME_MIN_MESSAGES - 1):
            yield make_message("обычное сообщение", -1, "group")
        yield candidate
        yield candidate
        if outcome.endswith("429"):
            clock += CHIME_CHAT_COOLDOWN
            yield candidate

    assert dispatch(reply, messages(), boundary_api=check) == []
    assert len(checks) == 1
    assert len(replies) == (outcome == "reply 429")
