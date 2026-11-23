FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY bridge.py run.py telegram_common.py tg_send.py ./

CMD ["python", "-u", "run.py"]
