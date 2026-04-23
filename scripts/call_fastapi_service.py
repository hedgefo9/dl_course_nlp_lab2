"""Call local FastAPI LLM service running in Docker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests

DEFAULT_FASTAPI_URL = "http://127.0.0.1:8001"
DEFAULT_MODEL = "qwen2.5:0.5b"


def build_parser() -> argparse.ArgumentParser:
    """Create command-line parser."""
    parser = argparse.ArgumentParser(description="Send one prompt to FastAPI LLM service.")
    parser.add_argument(
        "--url",
        default=DEFAULT_FASTAPI_URL,
        help="Base URL of FastAPI service.",
    )
    parser.add_argument(
        "--prompt",
        default="Определи: это спам? \"You won a free iPhone, click now!\"",
        help="Prompt text sent to the model.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model name.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--system-file",
        type=Path,
        default=None,
        help="Optional path to a file with system prompt.",
    )
    return parser


def call_fastapi(
    url: str,
    prompt: str,
    model: str,
    timeout: int,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Send prompt to FastAPI `/generate` endpoint and return JSON response."""
    endpoint = f"{url.rstrip('/')}/generate"
    payload = {
        "prompt": prompt,
        "model": model,
        "stream": False,
    }
    if system_prompt is not None:
        payload["system"] = system_prompt
    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def main() -> None:
    """Parse CLI arguments, call service and print formatted JSON."""
    args = build_parser().parse_args()
    system_prompt = None
    if args.system_file is not None:
        system_prompt = args.system_file.read_text(encoding="utf-8")
    data = call_fastapi(
        url=args.url,
        prompt=args.prompt,
        model=args.model,
        timeout=args.timeout,
        system_prompt=system_prompt,
    )
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
