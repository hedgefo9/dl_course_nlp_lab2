"""Call Ollama HTTP API exposed from Docker container."""

from __future__ import annotations

import argparse
import json
from typing import Any

import requests

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11435"
DEFAULT_MODEL = "qwen2.5:0.5b"


def build_parser() -> argparse.ArgumentParser:
    """Create command-line parser."""
    parser = argparse.ArgumentParser(description="Send one prompt to containerized Ollama API.")
    parser.add_argument(
        "--url",
        default=DEFAULT_OLLAMA_URL,
        help="Base URL of exposed Ollama API.",
    )
    parser.add_argument(
        "--prompt",
        default="Classify SMS as spam or ham: WINNER! Claim your free ticket now!",
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
    return parser


def call_ollama(url: str, prompt: str, model: str, timeout: int) -> dict[str, Any]:
    """Send prompt to Ollama `/api/generate` endpoint and return JSON response."""
    endpoint = f"{url.rstrip('/')}/api/generate"
    payload = {
        "prompt": prompt,
        "model": model,
        "stream": False,
    }
    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def main() -> None:
    """Parse CLI arguments, call service and print formatted JSON."""
    args = build_parser().parse_args()
    data = call_ollama(
        url=args.url,
        prompt=args.prompt,
        model=args.model,
        timeout=args.timeout,
    )
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
