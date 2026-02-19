FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY libs/common /app/libs/common
COPY apps/bot /app/apps/bot

RUN pip install --no-cache-dir /app/libs/common /app/apps/bot

CMD ["interview-bot"]
