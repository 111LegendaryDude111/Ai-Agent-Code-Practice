FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY libs/common /app/libs/common
COPY services/orchestrator /app/services/orchestrator

RUN pip install --no-cache-dir /app/libs/common /app/services/orchestrator

CMD ["interview-orchestrator"]
