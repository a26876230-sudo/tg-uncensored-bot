import sys
from unittest.mock import Mock

from telegram_common import split_text
import tg_send


def test_split_text_preserves_emoji_and_content():
    text = "я" * 4095 + "🎙" + "б" * 4095
    chunks = list(split_text(text))
    assert chunks == ["я" * 4095, "🎙" + "б" * 4094, "б"]
    assert "".join(chunks) == text


def test_send_text_uses_first_owner_and_splits(monkeypatch):
    post = Mock(return_value=Mock(ok=True, **{"json.return_value": {"ok": True}}))
    monkeypatch.setattr(tg_send.requests, "post", post)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
    monkeypatch.setenv("TELEGRAM_OWNER_ID", " 111, 222 ")
    monkeypatch.setattr(sys, "argv", ["tg_send.py", "--text", "я" * 4097])

    assert tg_send.main() == 0
    assert [call.kwargs["data"] for call in post.call_args_list] == [
        {"chat_id": 111, "text": "я" * 4096},
        {"chat_id": 111, "text": "я"},
    ]
    assert all(call.args[0].endswith("/sendMessage") for call in post.call_args_list)


def test_send_apk_as_document(monkeypatch, tmp_path):
    apk = tmp_path / "app.apk"
    apk.write_bytes(b"apk")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
    monkeypatch.setattr(sys, "argv", [
        "tg_send.py", "--chat", "222", "--file", str(apk), "--caption", "Готово",
    ])

    def post(url, *, data, files, **kwargs):
        assert url.endswith("/sendDocument")
        assert data == {"chat_id": 222, "caption": "Готово"}
        name, file = files["document"]
        assert name == "app.apk"
        assert file.read() == b"apk"
        return Mock(ok=True, **{"json.return_value": {"ok": True}})

    monkeypatch.setattr(tg_send.requests, "post", post)
    assert tg_send.main() == 0
