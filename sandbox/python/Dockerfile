FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends time \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /usr/sbin/nologin sandbox

WORKDIR /workspace

COPY sandbox/python/run.sh /usr/local/bin/run.sh
RUN chmod 0555 /usr/local/bin/run.sh

USER sandbox

ENTRYPOINT ["/usr/local/bin/run.sh"]
