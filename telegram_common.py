def parse_owner_ids(value: str) -> tuple[int, ...]:
    try:
        ids = tuple(int(item.strip()) for item in value.split(","))
    except ValueError:
        ids = ()
    if not ids or any(user_id <= 0 for user_id in ids):
        raise ValueError("TELEGRAM_OWNER_ID должен содержать числовые ID через запятую.")
    return ids


def split_text(text: str, limit: int = 4096):
    start = length = 0
    for index, char in enumerate(text):
        size = 2 if ord(char) > 0xFFFF else 1
        if length + size > limit:
            yield text[start:index]
            start, length = index, 0
        length += size
    if start < len(text):
        yield text[start:]
