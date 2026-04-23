"""Run batch inference against a local Ollama server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

DEFAULT_MODEL = "qwen2.5:0.5b"
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_JSON_OUTPUT = Path("artifacts/inference_results.json")
DEFAULT_MARKDOWN_OUTPUT = Path("artifacts/inference_results.md")
DEFAULT_PROMPTS = [
    "Привет! Представься в одном предложении.",
    "Объясни разницу между стеком и кучей в программировании в двух предложениях.",
    "Составь короткий список из трёх полезных привычек для продуктивной учебы.",
    "Напиши вежливый отказ от встречи на сегодня.",
    "Переведи на английский: Сегодня я изучаю NLP.",
    "Сколько будет 17 * 23?",
    "Дай идею простого weekend-проекта на Python.",
    "Что такое overfitting в машинном обучении?",
    "Сформулируй определение API одним предложением.",
    "Предложи одну тему для небольшого доклада по ИИ.",
]


def build_parser() -> argparse.ArgumentParser:
    """Create and return command-line parser for the script."""
    parser = argparse.ArgumentParser(description="Batch inference via Ollama HTTP API.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama server host.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name in Ollama.")
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Path to a text file with one prompt per line.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout for each request in seconds.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help="Path to JSON file with inference results.",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
        help="Path to Markdown file with two-column inference table.",
    )
    return parser


def load_prompts(prompts_file: Path | None) -> list[str]:
    """Load prompts from a file or return built-in defaults."""
    if prompts_file is None:
        return DEFAULT_PROMPTS.copy()

    text = prompts_file.read_text(encoding="utf-8")
    prompts = [line.strip() for line in text.splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in file: {prompts_file}")
    return prompts


def query_ollama(host: str, model: str, prompt: str, timeout: int) -> str:
    """Send one prompt to Ollama and return model response."""
    endpoint = f"{host.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return str(data.get("response", "")).strip()


def run_batch_inference(
    host: str,
    model: str,
    prompts: list[str],
    timeout: int,
) -> list[dict[str, str]]:
    """Run inference for each prompt and return a list of records."""
    results: list[dict[str, str]] = []
    total = len(prompts)
    for index, prompt in enumerate(prompts, start=1):
        print(f"[{index}/{total}] Sending prompt...")
        answer = query_ollama(host=host, model=model, prompt=prompt, timeout=timeout)
        results.append({"prompt": prompt, "response": answer})
    return results


def save_json(results: list[dict[str, str]], output_path: Path) -> None:
    """Save inference results into a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def escape_markdown_cell(text: str) -> str:
    """Escape markdown table cell content."""
    return text.replace("|", "\\|").replace("\n", "<br>")


def save_markdown_table(results: list[dict[str, str]], output_path: Path) -> None:
    """Save two-column inference report as a markdown table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Запрос к LLM | Вывод LLM |",
        "|---|---|",
    ]
    for item in results:
        prompt = escape_markdown_cell(item["prompt"])
        response = escape_markdown_cell(item["response"])
        lines.append(f"| {prompt} | {response} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Parse arguments, run inference and save artifacts."""
    args = build_parser().parse_args()
    prompts = load_prompts(args.prompts_file)
    results = run_batch_inference(
        host=args.host,
        model=args.model,
        prompts=prompts,
        timeout=args.timeout,
    )
    save_json(results=results, output_path=args.json_output)
    save_markdown_table(results=results, output_path=args.md_output)
    print(f"Saved JSON report: {args.json_output}")
    print(f"Saved Markdown report: {args.md_output}")


if __name__ == "__main__":
    main()
