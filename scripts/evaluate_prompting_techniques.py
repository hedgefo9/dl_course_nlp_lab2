"""Evaluate prompting techniques for SMS spam classification with LLM."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

DEFAULT_DATASET = Path("spam.csv")
DEFAULT_FASTAPI_URL = "http://127.0.0.1:8001"
DEFAULT_MODEL = "qwen2.5:0.5b"
DEFAULT_SAMPLES_PER_CLASS = 400
DEFAULT_SEED = 42
DEFAULT_TIMEOUT = 180
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_ATTEMPT_SLEEP_SECONDS = 1.5
DEFAULT_METRICS_OUTPUT = Path("artifacts/prompting_metrics.json")
DEFAULT_PREDICTIONS_OUTPUT = Path("artifacts/prompting_predictions.csv")
DEFAULT_MARKDOWN_OUTPUT = Path("artifacts/prompting_metrics.md")

TECHNIQUE_ORDER = [
    "zero_shot",
    "cot",
    "few_shot",
    "cot_few_shot",
]

TECHNIQUE_TITLES = {
    "zero_shot": "zero-shot",
    "cot": "CoT",
    "few_shot": "few-shot",
    "cot_few_shot": "CoT + few-shot",
}

SYSTEM_PROMPTS = {
    "cot": """
Ты классификатор SMS-спама.
Применяй технику Chain-of-Thought (CoT): перед финальным ответом последовательно оцени признаки:
1) срочный призыв к действию;
2) обещание выигрыша или выгоды;
3) запрос денег, персональных данных, перехода по ссылке или ответа на короткий номер;
4) признаки обычной личной переписки.

Верни только JSON и ничего больше:
{"reasoning":"<краткая строка>","verdict":<0 или 1>}

Требования:
- reasoning: одна строка на русском языке, 10-25 слов;
- verdict: цифра 1 для spam и 0 для ham.
""".strip(),
    "few_shot": """
Ты классификатор SMS-спама.
Используй few-shot подход и опирайся на примеры:

Пример 1:
SMS: "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121."
Ответ: {"reasoning":"Есть обещание выигрыша и призыв отправить сообщение на короткий номер.","verdict":1}

Пример 2:
SMS: "Hey, are we still meeting near the station at 7?"
Ответ: {"reasoning":"Обычное личное сообщение без рекламы, давления и попытки выманить деньги.","verdict":0}

Пример 3:
SMS: "URGENT! Your account has been selected for a cash reward, call now."
Ответ: {"reasoning":"Срочность и обещание денежного вознаграждения указывают на типичный спам-шаблон.","verdict":1}

Формат ответа для нового сообщения:
{"reasoning":"<краткая строка>","verdict":<0 или 1>}

Требования:
- reasoning: одна строка на русском языке;
- verdict: цифра 1 для spam и 0 для ham.
""".strip(),
    "cot_few_shot": """
Ты классификатор SMS-спама.
Используй CoT + few-shot:
1) сначала внутренне оцени признаки спама (срочность, обещание выгоды, финансовый или персональный риск, нейтральность переписки);
2) сверяйся с примерами;
3) после анализа выдай итог только в JSON.

Пример 1:
SMS: "WINNER!! As a valued network customer you have been selected to receive a prize."
Ответ: {"reasoning":"Маркер WINNER и обещание приза указывают на рекламно-мошеннический паттерн.","verdict":1}

Пример 2:
SMS: "Can you send me the report before lunch?"
Ответ: {"reasoning":"Рабочая просьба без рекламы, без призов и без давления по оплате.","verdict":0}

Выходной формат строго:
{"reasoning":"<краткая строка>","verdict":<0 или 1>}

Требования:
- reasoning: одна строка на русском языке;
- verdict: цифра 1 для spam и 0 для ham.
""".strip(),
}


@dataclass(frozen=True)
class SmsSample:
    """One labeled SMS sample."""

    text: str
    label: int


def build_parser() -> argparse.ArgumentParser:
    """Create command-line parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot, CoT, few-shot and CoT+few-shot on spam.csv.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to spam.csv.")
    parser.add_argument("--url", default=DEFAULT_FASTAPI_URL, help="FastAPI base URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name in Ollama.")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        help="Number of samples per class for balanced evaluation.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Maximum attempts per request.",
    )
    parser.add_argument(
        "--attempt-sleep",
        type=float,
        default=DEFAULT_ATTEMPT_SLEEP_SECONDS,
        help="Sleep between retries in seconds.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_OUTPUT,
        help="Path to JSON file with metrics.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=DEFAULT_PREDICTIONS_OUTPUT,
        help="Path to CSV file with sample-level predictions.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
        help="Path to markdown report with metric table.",
    )
    return parser


def load_dataset(path: Path) -> list[SmsSample]:
    """Load spam.csv and return list of labeled SMS samples."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            samples: list[SmsSample] = []
            with path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    label_raw = (row.get("v1") or "").strip().lower()
                    text = (row.get("v2") or "").strip()
                    if label_raw not in {"ham", "spam"} or not text:
                        continue
                    label = 1 if label_raw == "spam" else 0
                    samples.append(SmsSample(text=text, label=label))
            if samples:
                return samples
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "spam.csv",
        b"",
        0,
        1,
        "Cannot decode dataset with tried encodings: utf-8, utf-8-sig, latin-1",
    )


def balanced_sample(samples: list[SmsSample], samples_per_class: int, seed: int) -> list[SmsSample]:
    """Return balanced sample with an equal number of ham and spam rows."""
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive.")

    ham_samples = [sample for sample in samples if sample.label == 0]
    spam_samples = [sample for sample in samples if sample.label == 1]
    randomizer = random.Random(seed)
    randomizer.shuffle(ham_samples)
    randomizer.shuffle(spam_samples)

    if len(ham_samples) < samples_per_class or len(spam_samples) < samples_per_class:
        raise ValueError(
            "Not enough samples for balanced subset: "
            f"ham={len(ham_samples)}, spam={len(spam_samples)}, requested={samples_per_class} per class."
        )

    selected = ham_samples[:samples_per_class] + spam_samples[:samples_per_class]
    randomizer.shuffle(selected)
    return selected


def build_user_prompt(technique: str, sms_text: str) -> str:
    """Build user prompt for a specific prompting technique."""
    if technique == "zero_shot":
        return (
            "Классифицируй SMS как spam или ham.\n"
            "Верни только одну цифру: 1 для spam, 0 для ham.\n"
            f'SMS: "{sms_text}"'
        )

    return (
        "Классифицируй SMS сообщение.\n"
        f'SMS: "{sms_text}"\n'
        "Верни ответ строго в требуемом JSON-формате."
    )


def build_request_payload(technique: str, sms_text: str, model: str) -> dict[str, Any]:
    """Create request payload for FastAPI `/generate` endpoint."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": build_user_prompt(technique=technique, sms_text=sms_text),
        "stream": False,
    }
    if technique in SYSTEM_PROMPTS:
        payload["system"] = SYSTEM_PROMPTS[technique]
    return payload


def send_generate_request(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Send one request to FastAPI `/generate` and return JSON response."""
    endpoint = f"{url.rstrip('/')}/generate"
    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def send_with_retry(
    url: str,
    payload: dict[str, Any],
    timeout: int,
    max_attempts: int,
    attempt_sleep: float,
) -> dict[str, Any]:
    """Send request with retry strategy."""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return send_generate_request(url=url, payload=payload, timeout=timeout)
        except requests.RequestException as error:
            last_error = error
            if attempt == max_attempts:
                break
            time.sleep(attempt_sleep)
    if last_error is None:
        raise RuntimeError("Unknown error in retry loop.")
    raise last_error


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract first balanced JSON object from model text output."""
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start == -1:
        return None
    depth = 0
    for index in range(start, len(stripped)):
        char = stripped[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def normalize_verdict(value: Any) -> int | None:
    """Normalize verdict value to integer class label (0 or 1)."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value if value in {0, 1} else None
    if isinstance(value, float):
        int_value = int(value)
        return int_value if int_value in {0, 1} else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"0", "1"}:
            return int(stripped)
    return None


def extract_first_binary_digit(text: str) -> int | None:
    """Extract first standalone binary digit (0 or 1) from text."""
    match = re.search(r"\b([01])\b", text)
    if match is None:
        return None
    return int(match.group(1))


def keyword_fallback_prediction(text: str) -> int:
    """Return heuristic prediction based on spam/ham keywords."""
    lower_text = text.lower()
    has_spam = "spam" in lower_text
    has_ham = "ham" in lower_text
    if has_spam and not has_ham:
        return 1
    if has_ham and not has_spam:
        return 0
    return 0


def parse_zero_shot_response(raw_response: str) -> tuple[int, bool, str]:
    """Parse model response for zero-shot strategy."""
    digit = extract_first_binary_digit(raw_response)
    if digit is not None:
        return digit, True, raw_response.strip()
    return keyword_fallback_prediction(raw_response), False, raw_response.strip()


def parse_structured_response(raw_response: str) -> tuple[int, bool, str]:
    """Parse model response expected as JSON with `reasoning` and `verdict`."""
    parsed = extract_json_object(raw_response)
    if parsed is not None:
        verdict = normalize_verdict(parsed.get("verdict"))
        reasoning = str(parsed.get("reasoning", "")).strip()
        if verdict is not None:
            return verdict, True, reasoning

    digit = extract_first_binary_digit(raw_response)
    if digit is not None:
        return digit, False, raw_response.strip()
    return keyword_fallback_prediction(raw_response), False, raw_response.strip()


def parse_prediction(technique: str, raw_response: str) -> tuple[int, bool, str]:
    """Parse prediction and reasoning from model response."""
    if technique == "zero_shot":
        return parse_zero_shot_response(raw_response)
    return parse_structured_response(raw_response)


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Compute accuracy, precision, recall and F1 for positive class = spam."""
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)
    total = len(y_true)

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Round float metrics to four digits for compact reporting."""
    rounded: dict[str, float] = {}
    for key, value in metrics.items():
        if key in {"tp", "tn", "fp", "fn"}:
            rounded[key] = value
        else:
            rounded[key] = round(value, 4)
    return rounded


def evaluate_technique(
    samples: list[SmsSample],
    technique: str,
    url: str,
    model: str,
    timeout: int,
    max_attempts: int,
    attempt_sleep: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one prompting technique and return metrics + detailed rows."""
    y_true: list[int] = []
    y_pred: list[int] = []
    rows: list[dict[str, Any]] = []
    parse_errors = 0
    request_errors = 0
    total = len(samples)

    for index, sample in enumerate(samples, start=1):
        if index == 1 or index % 20 == 0 or index == total:
            print(f"[{TECHNIQUE_TITLES[technique]}] {index}/{total}")

        payload = build_request_payload(technique=technique, sms_text=sample.text, model=model)
        raw_response = ""
        try:
            response_json = send_with_retry(
                url=url,
                payload=payload,
                timeout=timeout,
                max_attempts=max_attempts,
                attempt_sleep=attempt_sleep,
            )
            raw_response = str(response_json.get("response", "")).strip()
        except requests.RequestException as error:
            request_errors += 1
            raw_response = f"REQUEST_ERROR: {error}"

        prediction, parsed_ok, reasoning = parse_prediction(
            technique=technique,
            raw_response=raw_response,
        )
        if not parsed_ok:
            parse_errors += 1

        y_true.append(sample.label)
        y_pred.append(prediction)
        rows.append(
            {
                "technique": technique,
                "text": sample.text,
                "label": sample.label,
                "prediction": prediction,
                "parsed_ok": parsed_ok,
                "reasoning": reasoning,
                "raw_response": raw_response,
            }
        )

    metric_values = round_metrics(compute_metrics(y_true=y_true, y_pred=y_pred))
    metric_values["sample_count"] = float(total)
    metric_values["parse_error_rate"] = round(parse_errors / total if total else 0.0, 4)
    metric_values["request_error_rate"] = round(request_errors / total if total else 0.0, 4)
    summary = {
        "technique": technique,
        "technique_title": TECHNIQUE_TITLES[technique],
        "metrics": metric_values,
    }
    return summary, rows


def write_metrics_json(result: dict[str, Any], path: Path) -> None:
    """Write aggregated metrics to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def write_predictions_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write row-level predictions to CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = [
        "technique",
        "label",
        "prediction",
        "parsed_ok",
        "text",
        "reasoning",
        "raw_response",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_report(result: dict[str, Any], path: Path) -> None:
    """Write compact markdown table with metrics by prompting technique."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Метрики по техникам промптинга",
        "",
        f"- Модель: `{result['model']}`",
        f"- Размер выборки: `{result['sample_count']}`",
        f"- Датасет: `{result['dataset_path']}`",
        "",
        "| Техника | Accuracy | Precision | Recall | F1 | Parse error rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for item in result["techniques"]:
        metrics = item["metrics"]
        lines.append(
            f"| {item['technique_title']} | "
            f"{metrics['accuracy']:.4f} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | "
            f"{metrics['parse_error_rate']:.4f} |"
        )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run full evaluation pipeline and save artifacts."""
    args = build_parser().parse_args()

    all_samples = load_dataset(args.dataset)
    selected_samples = balanced_sample(
        samples=all_samples,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )

    class_balance = {
        "ham": sum(1 for sample in selected_samples if sample.label == 0),
        "spam": sum(1 for sample in selected_samples if sample.label == 1),
    }
    print(
        "Selected samples:",
        len(selected_samples),
        "| class balance:",
        class_balance,
    )

    all_rows: list[dict[str, Any]] = []
    technique_results: list[dict[str, Any]] = []

    for technique in TECHNIQUE_ORDER:
        summary, rows = evaluate_technique(
            samples=selected_samples,
            technique=technique,
            url=args.url,
            model=args.model,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            attempt_sleep=args.attempt_sleep,
        )
        technique_results.append(summary)
        all_rows.extend(rows)

    result = {
        "dataset_path": str(args.dataset),
        "model": args.model,
        "url": args.url,
        "sample_count": len(selected_samples),
        "class_balance": class_balance,
        "samples_per_class": args.samples_per_class,
        "max_attempts": args.max_attempts,
        "techniques": technique_results,
        "system_prompts": SYSTEM_PROMPTS,
    }

    write_metrics_json(result=result, path=args.metrics_output)
    write_predictions_csv(rows=all_rows, path=args.predictions_output)
    write_markdown_report(result=result, path=args.markdown_output)

    print(f"Saved metrics JSON: {args.metrics_output}")
    print(f"Saved predictions CSV: {args.predictions_output}")
    print(f"Saved markdown summary: {args.markdown_output}")


if __name__ == "__main__":
    main()
