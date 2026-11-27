"""Score public multiple-choice concepts with a pinned Nebius Muse Spark Open route."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import requests

from . import CLASSES
from .audit_muse_spark_open import LETTERS, MODEL_IDS, audit_cell
from .teacher import scoring_prompt, validate_questions, write_json


URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "meta/muse-spark-open-20261015"
PROVIDER = "nebius/fp4"
PARAMETERS = {
    "model": MODEL, "provider": {"only": [PROVIDER], "order": [PROVIDER], "require_parameters": True,
                                "allow_fallbacks": False},
    "temperature": 1, "top_p": 1, "reasoning": {"effort": "low", "exclude": True},
    "max_tokens": 512, "logprobs": True, "top_logprobs": 20,
}
LOCAL = threading.local()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def concepts(source):
    if not source:
        return dict(CLASSES)
    spec = load(source)
    codes, names = spec["class_codes"], spec["class_names"]
    if not codes or len(codes) != len(names) or len(set(codes)) != len(codes):
        raise ValueError("Invalid aligned concept file")
    return dict(zip(codes, names))


def fetch(api_key, class_code, number, prompt):
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
    row = {"class_code": class_code, "question_number": number, "prompt_sha256": digest(prompt.encode("utf-8")),
           "started_at_utc": timestamp()}
    for repeat in range(4):
        try:
            with LOCAL.session.post(URL, headers={"Authorization": f"Bearer {api_key}"},
                                    json={**PARAMETERS, "messages": [{"role": "user", "content": prompt}]},
                                    timeout=(12, 90)) as req:
                row["http_status"] = req.status_code
                if req.status_code == 200:
                    data = req.json()
                    row.pop("error", None)
                    if "choices" not in data or not data["choices"]:
                        row["error"] = "missing response choices"
                    else:
                        rendered = data["choices"][0]
                        row["response"] = {
                            "id": data.get("id"), "model": data.get("model"), "provider": data.get("provider"),
                            "created": data.get("created"), "usage": data.get("usage"), "system_fingerprint": data.get("system_fingerprint"),
                            "choice": {"content": rendered.get("message", {}).get("content"), "logprobs": rendered.get("logprobs"),
                                       "finish_reason": rendered.get("finish_reason")},
                        }
                        if data.get("provider") != "Nebius" or data.get("model") not in MODEL_IDS:
                            row["error"] = "returned model or provider differs from the frozen route"
                        return row
                if req.status_code not in (408, 429, 500, 502, 503, 504):
                    row["error"] = "routing or HTTP request rejected"
                    return row
                retry_after = req.headers.get("Retry-After", "")
                delay = min(30, float(retry_after)) if retry_after.isdigit() else min(15, 2 ** repeat)
        except requests.RequestException as error:
            row["error"] = type(error).__name__
            row["http_status"] = None
            delay = min(15, 2 ** repeat)
        except (TypeError, ValueError, KeyError):
            row["error"] = "invalid JSON response"
            delay = min(15, 2 ** repeat)
        time.sleep(delay)
    return row


def audit(raw, prompt, reference=None):
    if raw.get("error") or raw.get("http_status") != 200:
        return {k: raw.get(k) for k in ("class_code", "question_number", "prompt_sha256", "http_status", "error")}
    return audit_cell(raw, prompt, reference, PARAMETERS["top_logprobs"])


def estimate(audited, fraction=0.5):
    scores = audited.get("first_visible_scores") or {}
    exact = scores.get("exact_a_to_e_conditional_probabilities")
    if exact:
        return [exact[letter] for letter in LETTERS]
    observed = scores.get("native_option_probabilities") or {}
    cutoff = scores.get("top_k_cutoff_absolute_probability")
    if not cutoff or not any(x is not None and x > 0 for x in observed.values()):
        return None
    values = [observed.get(letter) if observed.get(letter) is not None else fraction * cutoff for letter in LETTERS]
    total = math.fsum(values)
    if total <= 0:
        return None
    return [value / total for value in values]


def run(args):
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key or any(ch.isspace() for ch in api_key):
        raise SystemExit("Set OPENROUTER_API_KEY to an OpenRouter API key.")
    questions = validate_questions(load(args.questions))
    roster = concepts(args.concepts)
    expected = {}
    if args.rubric:
        rubric = load(args.rubric)
        options = rubric["expected_options_by_question"]
        if rubric["class_codes"] != list(roster) or len(options) != len(questions):
            raise ValueError("The rubric must follow the question and class order.")
        for number, row in enumerate(options, 1):
            if len(row) != len(roster) or any(letter not in tuple(LETTERS) for letter in row):
                raise ValueError("The rubric needs one option letter per class and question.")
            expected.update({(code, number): letter for code, letter in zip(roster, row, strict=True)})
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    pairs = [(code, j, scoring_prompt(question, name)) for code, name in roster.items()
             for j, question in enumerate(questions, 1)]
    protocol = {
        "endpoint": URL, "request_parameters": PARAMETERS, "question_sha256": digest(args.questions.read_bytes()),
        "concept_sha256": digest(args.concepts.read_bytes()) if args.concepts else None,
        "rubric_sha256": digest(args.rubric.read_bytes()) if args.rubric else None,
        "fixed_missing_mass_estimate": "First visible token: each censored bare option has half its verified top-20 cutoff, then normalize",
        "selection": "First structurally scoreable response for a given prompt; retries only for missing/unusable response",
    }
    protocol_path = args.out / "request_protocol.json"
    if protocol_path.exists() and load(protocol_path) != protocol:
        raise ValueError("Cannot reuse a response directory with different frozen inputs")
    write_json(protocol_path, protocol)

    def one(pair):
        code, j, prompt = pair
        reference = expected.get((code, j))
        fingerprint = digest(prompt.encode("utf-8"))
        path = args.cache / (fingerprint + ".json")
        attempts = load(path) if path.is_file() else []
        for raw in attempts:
            assert raw["prompt_sha256"] == fingerprint and raw["class_code"] == code and raw["question_number"] == j
            audited = audit(raw, prompt, reference)
            if estimate(audited) is not None:
                return attempts, audited
        while len(attempts) < args.semantic_attempts:
            raw = fetch(api_key, code, j, prompt)
            attempts.append(raw)
            write_json(path, attempts)
            audited = audit(raw, prompt, reference)
            if estimate(audited) is not None or raw.get("http_status") in (400, 401, 402, 403, 404):
                return attempts, audited
        return attempts, audit(attempts[-1], prompt, reference)

    first = one(pairs[0])
    if first[1].get("http_status") in (400, 401, 402, 403, 404):
        write_json(args.out / "probe_failure.json", {"protocol": protocol, "attempts": first[0]})
        raise SystemExit(f"Muse request rejected with HTTP {first[1]['http_status']}. See {args.out / 'probe_failure.json'}.")
    raw_rows, cells = [first[0]], [first[1]]
    print("Pinned route preflight", first[1].get("http_status"), "scoreable", estimate(first[1]) is not None, flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for attempts, entry in pool.map(one, pairs[1:]):
            raw_rows.append(attempts)
            cells.append(entry)
            if len(cells) % 24 == 0:
                print(f"{len(cells)}/{len(pairs)} queried; {sum(estimate(x) is not None for x in cells)} scoreable", flush=True)
    write_json(args.out / "native-responses.json", {**protocol, "rows_of_attempts": raw_rows, "completed_at_utc": timestamp()})
    summary = {
        "requested_cells": len(pairs), "scoreable_cells": sum(estimate(x) is not None for x in cells),
        "http_responses": dict(Counter(str(x.get("http_status")) for x in cells)),
        "first_visible_bare_letters": sum(bool(x.get("first_visible_token_is_bare_option")) for x in cells),
        "exact_rows": sum(bool((x.get("first_visible_scores") or {}).get("exact_a_to_e_conditional_probabilities")) for x in cells),
        "returned_models": sorted({raw["response"]["model"] for group in raw_rows for raw in group
                                   if raw.get("response") and not raw.get("error")}),
    }
    write_json(args.out / "scoring-audit.json", {**protocol, "summary": summary, "rows": cells})
    if summary["scoreable_cells"] == len(pairs):
        probabilities = [estimate(cell) for cell in cells]
        n = len(questions)
        signature = {**protocol, **summary, "class_codes": list(roster), "class_names": list(roster.values()),
                     "answer_labels": list(LETTERS), "questions": questions,
                     "probabilities": [probabilities[i:i + n] for i in range(0, len(probabilities), n)]}
        write_json(args.out / "signature.json", signature)
    print(summary, flush=True)
    if summary["scoreable_cells"] != len(pairs):
        raise SystemExit("Muse returned incomplete scores. See scoring-audit.json in the output directory.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("questions", "out", "cache"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--concepts", type=Path)
    p.add_argument("--rubric", type=Path, help="Optional reference answers for the score audit")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--semantic-attempts", type=int, default=3)
    args = p.parse_args()
    if min(args.workers, args.semantic_attempts) < 1:
        p.error("Worker and retry limits must be positive")
    run(args)


if __name__ == "__main__":
    main()
