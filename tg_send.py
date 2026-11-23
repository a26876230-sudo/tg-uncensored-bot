import argparse
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

from telegram_common import parse_owner_ids, split_text


MAX_FILE_SIZE = 50 * 1024 * 1024


def send(token: str, method: str, data: dict, files: dict | None = None):
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/{method}",
            data=data,
            files=files,
            timeout=(10, 120),
            allow_redirects=False,
        )
    except requests.RequestException:
        raise RuntimeError("Не удалось связаться с Telegram Bot API.") from None
    try:
        accepted = response.ok and response.json().get("ok") is True
    except (ValueError, AttributeError):
        accepted = False
    if not accepted:
        hint = " Лимит отправки файла через Telegram Bot API — 50 МБ." if files else ""
        raise RuntimeError(f"Telegram отклонил {method} (HTTP {response.status_code}).{hint}")


def main():
    parser = argparse.ArgumentParser(description="Отправить сообщение или файл в Telegram.")
    parser.add_argument("--chat", type=int, help="ID чата (по умолчанию первый TELEGRAM_OWNER_ID).")
    content = parser.add_mutually_exclusive_group(required=True)
    content.add_argument("--text", help="Текст сообщения.")
    content.add_argument("--file", type=Path, help="Путь к файлу до 50 МБ.")
    parser.add_argument("--caption", help="Подпись к файлу.")
    args = parser.parse_args()
    if args.caption is not None and args.file is None:
        parser.error("--caption требует --file")
    if args.text is not None and not args.text.strip():
        parser.error("--text не должен быть пустым")

    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        parser.error("Укажите TELEGRAM_BOT_TOKEN")

    try:
        chat_id = args.chat if args.chat is not None else parse_owner_ids(
            os.environ.get("TELEGRAM_OWNER_ID", "")
        )[0]
        if args.file is not None:
            data = {"chat_id": chat_id}
            if args.caption is not None:
                data["caption"] = args.caption
            if args.file.stat().st_size > MAX_FILE_SIZE:
                raise ValueError("Лимит отправки файла через Telegram Bot API — 50 МБ.")
            with args.file.open("rb") as file:
                send(token, "sendDocument", data, files={"document": (args.file.name, file)})
        else:
            for chunk in split_text(args.text):
                send(token, "sendMessage", {"chat_id": chat_id, "text": chunk})
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
