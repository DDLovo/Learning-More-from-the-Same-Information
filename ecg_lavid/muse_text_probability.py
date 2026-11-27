"""Collect Muse's visible probability reports for the frozen public rhythm question bank."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np
import requests
from scipy.special import rel_entr

from . import CLASSES
from .audit_muse_spark_open import MODEL_IDS
from .score_muse_concepts import PARAMETERS, URL, digest, load, timestamp
from .teacher import LETTERS, scoring_prompt, validate_questions, write_json
from .train import sha256


TEXT_PROMPT_APPEND = (
    '\n\nFor this request, replace the single-letter response format above with one JSON object. Choose '
    'your best answer letter. Also give your own subjective probability for each of the five options '
    'being the best answer, using numbers from 0 through 1 whose sum is 1. Report your beliefs '
    'directly. The outer object must have exactly two keys: "answer" (one uppercase letter A through E) '
    'and "probabilities" (an object with exactly the uppercase keys "A", "B", "C", "D", "E" and numeric '
    'values). Return the JSON object alone, with no code fences or commentary.'
)
SUM_TOLERANCE = 0.001
TRANSIENT = {408, 429, 500, 502, 503, 504}
RATE_LOCK = threading.Lock()
RATE_SPACING_SECONDS = 2.15
LAST_REQUEST = 0.0
THREAD = threading.local()
BUNDLE_TOKENS = 8192


def reject_duplicate_keys(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def object_only(content):
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty content")
    return json.loads(content, object_pairs_hook=reject_duplicate_keys)


def read_probabilities(entry, tolerance):
    if not isinstance(entry, dict) or set(entry) != {"answer", "probabilities"}:
        raise ValueError("expected only answer and probabilities")
    answer, numbers = entry["answer"], entry["probabilities"]
    if not isinstance(answer, str) or len(answer) != 1 or answer not in LETTERS:
        raise ValueError("invalid declared answer")
    if not isinstance(numbers, dict) or set(numbers) != set(LETTERS):
        raise ValueError("expected the five probability keys")
    observed = [numbers[letter] for letter in LETTERS]
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in observed):
        raise ValueError("invalid probability value")
    total = math.fsum(observed)
    if abs(total - 1.0) > tolerance:
        raise ValueError("probabilities outside the frozen sum tolerance")
    probability = [x / total for x in observed]
    tops = [letter for letter, p in zip(LETTERS, probability) if p == max(probability)]
    return {"answer": answer, "raw_numbers": numbers, "raw_sum": total, "probabilities": probability,
            "first_argmax": tops[0], "argmax_ties": tops, "declared_answer_differs_from_argmax": answer != tops[0]}


def request(api_key, parameters, prompt):
    global LAST_REQUEST
    if not hasattr(THREAD, "session"):
        THREAD.session = requests.Session()
    with RATE_LOCK:
        time.sleep(max(0., LAST_REQUEST + RATE_SPACING_SECONDS - time.monotonic()))
        LAST_REQUEST = time.monotonic()
    row = {"request_started_utc": timestamp(), "prompt_sha256": digest(prompt.encode("utf-8"))}
    try:
        with THREAD.session.post(URL, headers={"Authorization": f"Bearer {api_key}"},
                                 json={**parameters, "messages": [{"role": "user", "content": prompt}]},
                                 timeout=(15, 150), allow_redirects=False) as response:
            row.update(status=response.status_code, retry_after=response.headers.get("Retry-After"))
            if response.status_code != 200:
                return row
            data = response.json()
            options = data.get("choices") or []
            if not options:
                row["response_error"] = "no choices"
                return row
            choice = options[0]
            content = choice.get("message", {}).get("content")
            if isinstance(content, str):
                content = content.replace(api_key, "[redacted]")
            row.update(response_id=data.get("id"), model=data.get("model"), provider=data.get("provider"),
                       usage=data.get("usage"), finish_reason=choice.get("finish_reason"), content=content)
            return row
    except requests.RequestException as error:
        row.setdefault("status", None)
        row["transport_or_decode_error"] = type(error).__name__
        return row


def parse_reply(row, tolerance):
    if row["status"] != 200 or row.get("provider") != "Nebius" or row.get("model") not in MODEL_IDS:
        raise ValueError("missing completion on the frozen model route")
    if row.get("finish_reason") == "length":
        raise ValueError("completion length reached")
    return read_probabilities(object_only(row.get("content")), tolerance)


def collect_one(pair, destination, api, key, tolerance):
    identifier, prompt = pair
    path = destination / f"{identifier}.json"
    rows = load(path) if path.is_file() else []
    for existing in rows:
        if existing["prompt_sha256"] != digest(prompt.encode("utf-8")):
            raise ValueError(f"Cached prompt differs for {identifier}")
        if "valid" in existing:
            return rows
    malformed = sum(row["status"] == 200 for row in rows)
    transport = sum(row["status"] in TRANSIENT or row["status"] is None for row in rows)
    while malformed < 3 and transport < 5:
        row = request(key, api, prompt)
        try:
            row["valid"] = parse_reply(row, tolerance)
        except (ValueError, TypeError) as error:
            diagnostic = str(error)
            row["validation_error"] = diagnostic if len(diagnostic) < 250 else type(error).__name__
        rows.append(row)
        write_json(path, rows)
        if "valid" in row or row["status"] not in TRANSIENT | {200, None}:
            break
        if row["status"] == 200:
            malformed += 1
        else:
            transport += 1
            try:
                retry = float(row.get("retry_after") or 2 ** transport)
            except ValueError:
                retry = 2 ** transport
            time.sleep(min(60., max(0., retry)))
    return rows


def native_inputs(args, protocol, questions):
    native_path = args.native_dir / "signature.json"
    native_sha = sha256(native_path)
    native = load(native_path)
    if native["class_codes"] != list(CLASSES) or native["questions"] != questions or native["answer_labels"] != list(LETTERS):
        raise ValueError("The frozen class order or question bank differs from the native artifact")
    audited = load(args.native_dir / "scoring-audit.json")["rows"]
    if len(audited) != len(CLASSES) * len(questions):
        raise ValueError("Expected a complete native audit for this class and question bank")
    pairs = []
    top = []
    for class_index, (code, name) in enumerate(CLASSES.items()):
        for offset, question in enumerate(questions):
            original = scoring_prompt(question, name)
            row = audited[class_index * len(questions) + offset]
            if (row["class_code"] != code or row["question_number"] != offset + 1 or
                row["prompt_sha256"] != digest(original.encode("utf-8"))):
                raise ValueError("A reconstructed prompt differs from the archived native prompt")
            winner = row["first_visible_scores"]["restricted_top_option"]
            if winner not in LETTERS:
                raise ValueError("Missing certified native option argmax")
            top.append(LETTERS.index(winner))
            pairs.append((f"{code}-{offset + 1:02d}", original + protocol["append_exact"]))
    hard = np.eye(len(LETTERS), dtype=np.float64)[top].reshape(len(CLASSES), len(questions), len(LETTERS))
    soft = np.asarray(native["probabilities"], dtype=np.float64)
    if (soft.shape != hard.shape or not np.isfinite(soft).all() or (soft < 0).any()
            or not np.allclose(soft.sum(axis=-1), 1., atol=1e-8, rtol=0.)):
        raise ValueError("Expected a normalized native 4 × 12 × 5 signature")
    np.testing.assert_array_equal(hard.argmax(axis=-1), soft.argmax(axis=-1))
    native_copy = args.out / "native_soft.json"
    if not native_copy.exists():
        native_copy.write_bytes(native_path.read_bytes())
    if sha256(native_copy) != native_sha:
        raise ValueError("Archived native intervention differs from the frozen source")
    base = {"class_codes": native["class_codes"], "class_names": native["class_names"],
            "answer_labels": list(LETTERS), "questions": questions, "native_signature_sha256": native_sha}
    write_json(args.out / "native_top1_onehot.json", base | {
        "question_source": "Frozen discriminative bank, one-hot of the certified native A–E winner",
        "probabilities": hard.tolist(),
    })
    return pairs, hard, soft, base


def probability_summary(text, native, answers, native_hard):
    clinical_top = native_hard.argmax(axis=-1)
    textual_top = text.argmax(axis=-1)
    positive_text = np.where(text > 0, text, 1.0)
    positive_native = np.where(native > 0, native, 1.0)
    mixture = (text + native) / 2
    return {
        "cells": int(answers.size),
        "native_top1_matches_text_probability_top1": int((clinical_top == textual_top).sum()),
        "native_top1_matches_declared_text_answer": int((clinical_top == answers).sum()),
        "text_answer_matches_text_probability_top1": int((answers == textual_top).sum()),
        "mean_five_option_total_variation": float((abs(text - native).sum(axis=-1) / 2).mean()),
        "mean_jensen_shannon_nats": float((rel_entr(text, mixture) + rel_entr(native, mixture)).sum(axis=-1).mean() / 2),
        "mean_native_entropy_nats": float(-(native * np.log(positive_native)).sum(axis=-1).mean()),
        "mean_text_entropy_nats": float(-(text * np.log(positive_text)).sum(axis=-1).mean()),
        "zero_text_coordinates": int((text == 0).sum()),
    }


def rubric_summary(path, native, text, declared):
    reference = load(path)
    if reference.get("class_codes", reference.get("class_order")) != list(CLASSES):
        raise ValueError("The reference class order differs from the signatures")
    rows = reference.get("expected_options_by_question", reference.get("letters_by_question"))
    gold = np.asarray([[LETTERS.index(letter) for letter in row] for row in rows]).T
    if gold.shape != declared.shape:
        raise ValueError("The reference and the extracted signatures cover different question banks")
    one_hot = np.eye(len(LETTERS))[gold]
    result = {"rubric_sha256": sha256(path), "declared_correct": int((declared == gold).sum())}
    for name, probabilities in (("native_soft", native), ("text_soft", text)):
        expected = np.take_along_axis(probabilities, gold[..., None], axis=-1)[..., 0]
        result[name] = {"argmax_correct": int((probabilities.argmax(axis=-1) == gold).sum()),
                        "nll_nats": float(-np.log(expected).mean()) if (expected > 0).all() else None,
                        "zero_reference_probabilities": int((expected == 0).sum()),
                        "brier": float(np.square(probabilities - one_hot).sum(axis=-1).mean())}
    return result


def collect_bundle(args, pairs, api, key, tolerance, native, native_hard):
    archive = args.out / "bundle_replies.json"
    if archive.exists():
        return
    header = (
        "Answer each of the following public ECG rhythm questions independently, using its own five choices. "
        "Return one JSON array of 48 objects. Each object must contain exactly id, answer, and probabilities. "
        "Copy its supplied id. Give the single best uppercase A–E letter as answer, and your subjective "
        "probabilities for all five uppercase keys A, B, C, D, E, from 0 through 1 summing to 1. "
        "Return only the JSON array. Do not omit or duplicate any id."
    )
    prompt = header + "\n\n" + "\n\n".join(f"ID {key}\n{body}" for key, body in pairs)
    parameters = {**api, "max_tokens": BUNDLE_TOKENS}
    row = request(key, parameters, prompt)
    report = {"prompt": prompt, "request_parameters": parameters, "response": row, "expected_cells": len(pairs)}
    scored = {}
    errors = {}
    if row["status"] == 200 and row.get("provider") == "Nebius" and row.get("model") in MODEL_IDS:
        try:
            entries = object_only(row.get("content"))
            if not isinstance(entries, list):
                raise ValueError("expected a single JSON array")
            expected = set(identifier for identifier, _ in pairs)
            seen = set()
            for number, entry in enumerate(entries):
                if not isinstance(entry, dict) or set(entry) != {"id", "answer", "probabilities"}:
                    errors[str(number)] = "unexpected row fields"
                    continue
                identifier = entry["id"]
                if not isinstance(identifier, str) or identifier not in expected or identifier in seen:
                    errors[str(number)] = "missing, unknown or repeated id"
                    continue
                seen.add(identifier)
                try:
                    scored[identifier] = read_probabilities({k: entry[k] for k in ("answer", "probabilities")}, tolerance)
                except ValueError as error:
                    errors[identifier] = str(error)
        except (ValueError, TypeError) as error:
            errors["document"] = str(error)
    report.update(valid_rows=len(scored), invalid_details=errors, parsed=scored)
    if len(scored) == len(pairs) and not errors and row.get("finish_reason") != "length":
        text = np.asarray([scored[identifier]["probabilities"] for identifier, _ in pairs]).reshape(native.shape)
        words = np.asarray([LETTERS.index(scored[identifier]["answer"]) for identifier, _ in pairs]).reshape(native.shape[:2])
        report["comparison_to_native"] = probability_summary(text, native, words, native_hard)
    write_json(archive, report)


def run(args):
    questions = validate_questions(load(args.questions), expected_count=12)
    api = {key: value for key, value in PARAMETERS.items() if key not in ("logprobs", "top_logprobs")}
    protocol = {"questions_sha256": sha256(args.questions), "append_exact": TEXT_PROMPT_APPEND, "api": api,
                "absolute_sum_tolerance": SUM_TOLERANCE}
    protocol_sha = digest(json.dumps(protocol, sort_keys=True).encode("utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    pairs, native_hard, native, base = native_inputs(args, protocol, questions)
    if args.native_only:
        print("Frozen native soft and certified hard signatures ready", flush=True)
        return
    call = {"endpoint": URL, "protocol_sha256": protocol_sha, "native_signature_sha256": base["native_signature_sha256"],
            "api": protocol["api"], "bundle_max_tokens": BUNDLE_TOKENS, "prompts": dict(pairs)}
    manifest = args.out / "call_manifest.json"
    if manifest.exists() and load(manifest) != call:
        raise ValueError("Existing replies use a different protocol")
    write_json(manifest, call)
    cache = args.out / "individual_attempts"
    cache.mkdir(exist_ok=True)
    tolerance = SUM_TOLERANCE
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY to an OpenRouter API key.")
    rows = [collect_one(pairs[0], cache, api, key, tolerance)]
    if rows[0] and rows[0][-1]["status"] in (400, 401, 402, 403, 404):
        raise SystemExit(f"The frozen API route rejected the real first item (HTTP {rows[0][-1]['status']})")
    with ThreadPoolExecutor(max_workers=2) as pool:
        for number, attempts in enumerate(pool.map(
            lambda pair: collect_one(pair, cache, api, key, tolerance), pairs[1:],
        ), 2):
            rows.append(attempts)
            if number % 8 == 0:
                print(f"{number}/{len(pairs)} items finished", flush=True)
    replies = {identifier: attempts for (identifier, _), attempts in zip(pairs, rows)}
    write_json(args.out / "individual_replies.json", {"replies": replies, "completed_at_utc": timestamp()})
    complete = [next((row["valid"] for row in group if "valid" in row), None) for group in rows]
    counts = Counter(str(row["status"]) for group in rows for row in group)
    audit = {"valid_cells": sum(cell is not None for cell in complete), "total_cells": len(pairs),
             "api_attempt_status_counts": dict(counts),
             "bad_cells": [identifier for (identifier, _), item in zip(pairs, complete) if item is None]}
    if not all(item is not None for item in complete):
        write_json(args.out / "parsing_audit.json", audit)
        raise SystemExit(f"Only {audit['valid_cells']}/{len(pairs)} items valid; student stage stopped")
    soft = np.asarray([row["probabilities"] for row in complete]).reshape(native.shape)
    answers = np.asarray([LETTERS.index(row["answer"]) for row in complete]).reshape(native.shape[:2])
    hard = np.eye(len(LETTERS), dtype=np.float64)[answers]
    for name, distribution in (("text_soft", soft), ("text_answer_onehot", hard)):
        write_json(args.out / f"{name}.json", base | {
            "question_source": f"Frozen discriminative bank: {name}",
            "protocol_sha256": protocol_sha, "probabilities": distribution.tolist(),
        })
    audit.update(probability_summary(soft, native, answers, native_hard))
    if args.rubric:
        audit["reference_scores"] = rubric_summary(args.rubric, native, soft, answers)
    audit["text_probability_ties"] = sum(len(cell["argmax_ties"]) > 1 for cell in complete)
    audit["maximum_abs_raw_sum_error"] = max(abs(cell["raw_sum"] - 1.) for cell in complete)
    audit["hard_signatures_exactly_identical"] = bool(np.array_equal(hard, native_hard))
    write_json(args.out / "parsing_audit.json", audit)
    print(f"Saved complete four signatures; exact hard-set duplication: {audit['hard_signatures_exactly_identical']}", flush=True)
    if args.bundle:
        collect_bundle(args, pairs, api, key, tolerance, native, native_hard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("questions", "native-dir", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--rubric", type=Path)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--bundle", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
