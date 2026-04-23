"""FastAPI service that proxies generation requests to Ollama."""

from __future__ import annotations

import os
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DEFAULT_OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
DEFAULT_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "180"))

app = FastAPI(title="NLP Lab 2 LLM Service", version="1.0.0")


class GenerateRequest(BaseModel):
    """Input payload for one text generation request."""

    prompt: str = Field(..., description="Input prompt for the LLM.")
    model: str = Field(
        default=DEFAULT_MODEL,
        description="Model name known by Ollama.",
    )
    stream: bool = Field(
        default=False,
        description="Streaming mode for Ollama generate endpoint.",
    )
    system: str | None = Field(
        default=None,
        description="Optional system prompt passed directly to Ollama.",
    )


def call_ollama_generate(payload: GenerateRequest) -> dict[str, Any]:
    """Call Ollama `/api/generate` and return the raw JSON response."""
    endpoint = f"{DEFAULT_OLLAMA_API_URL.rstrip('/')}/api/generate"
    response = requests.post(
        endpoint,
        json=payload.model_dump(exclude_none=True),
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


@app.post("/generate")
def generate_text(request: GenerateRequest) -> dict[str, Any]:
    """Proxy text generation request from FastAPI to Ollama."""
    try:
        return call_ollama_generate(request)
    except requests.RequestException as error:
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {error}") from error
