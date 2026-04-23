FROM ubuntu:22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y python3 python3-pip python3-venv curl ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

COPY llm_service/requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY llm_service/entrypoint.sh /app/entrypoint.sh
COPY llm_service/app /app/llm_service/app

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000 11434

ENTRYPOINT ["/app/entrypoint.sh"]
