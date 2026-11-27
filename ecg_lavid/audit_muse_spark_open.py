"""Audit archived Muse Spark Open scores at the start of its visible response."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from statistics import median


LETTERS = "ABCDE"
MODEL_IDS = {"meta/muse-spark-open", "meta/muse-spark-open-20261015"}
END_MARKERS = {"<|eot|>", "<|end|>", "<|endoftext|>", "<|return|>"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def packed_token(entry):
    given = entry.get("bytes")
    return bytes(given) if given else (entry.get("token") or "").encode("utf-8")


def aligned_visible_positions(entries, content):
    """Some responses include reasoning-tail tokens before the visible suffix."""
    stop = len(entries)
    while stop and entries[stop - 1].get("token") in END_MARKERS:
        stop -= 1
    suffix = b""
    expected = content.encode("utf-8")
    for start in range(stop - 1, -1, -1):
        suffix = packed_token(entries[start]) + suffix
        if suffix == expected and expected:
            return start, stop
        if len(suffix) >= len(expected):
            break
    return None, None


def printed_decision(content, prompt):
    match = re.fullmatch(r"\s*([A-E])\s*", content)
    if match:
        return match[1], match.start(1), "bare"
    match = re.fullmatch(r"\s*\*\*([A-E])\.?\*\*\s*", content)
    if match:
        return match[1], match.start(1), "markdown_letter"
    matches = [(m[1], m.start(1), "explicit_answer_line") for m in re.finditer(
        r"(?m)^[ \t]*(?:\*\*)?(?i:(?:final[ \t]+)?answer[ \t]*:)[ \t]*([A-E])(?=[.\s*]|$)", content,
    )]
    options = {line for line in prompt.splitlines() if re.match(r"^[A-E]\. ", line)}
    for match in re.finditer(r"(?m)^[ \t]*(?:\*\*)?([A-E]\. [^\r\n]+)", content):
        option = match[1].removesuffix("**").rstrip()
        if option in options:
            matches.append((option[0], match.start(1), "explicit_option_text"))
    if matches and len({choice for choice, _, _ in matches}) == 1:
        return max(matches, key=lambda item: item[1])
    return None, None, None


def usable_logprob(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and -9999 < value <= 0


def score_position(entry, requested_k, options=LETTERS):
    observed = {option: None for option in options}
    top = entry.get("top_logprobs") or []
    raw = [(packed_token(item), item.get("logprob")) for item in top]
    for option in options:
        token = option.encode("ascii")
        seen = {value for word, value in raw if word == token and usable_logprob(value)}
        if not seen and packed_token(entry) == token and usable_logprob(entry.get("logprob")):
            seen.add(entry["logprob"])
        if len(seen) == 1:
            observed[option], = seen
    def descending(items):
        return all(left[1] >= right[1] for left, right in zip(items, items[1:]))

    order = None
    if (
        len(raw) == requested_k and len({word for word, _ in raw}) == len(raw)
        and all(usable_logprob(value) for _, value in raw)
        and math.fsum(math.exp(value) for _, value in raw) <= 1.0001
    ):
        if descending(raw):
            order = "descending"
        elif (raw[0] == (packed_token(entry), entry.get("logprob")) and descending(raw[1:])
              and raw[0][1] >= raw[-1][1]):
            order = "sampled_token_then_descending_alternatives"
    cutoff_logprob = min(value for _, value in raw) if order is not None else None
    probabilities = {letter: math.exp(value) if value is not None else None for letter, value in observed.items()}
    known_mass = math.fsum(value for value in probabilities.values() if value is not None)
    missing = [letter for letter, value in observed.items() if value is None]
    unknown_bound = len(missing) * math.exp(cutoff_logprob) if cutoff_logprob is not None else None
    best = max((option for option in options if observed[option] is not None), key=observed.get, default=None)
    rank = best if best is not None and (not missing or (cutoff_logprob is not None and observed[best] > cutoff_logprob)) else None
    return {
        "native_option_logprobs": observed,
        "native_option_probabilities": probabilities,
        ("exact_a_to_e_conditional_probabilities" if options == LETTERS else "exact_conditional_probabilities"): None if missing else {
            letter: value / known_mass for letter, value in probabilities.items()
        },
        "observed_bare_option_mass": known_mass,
        "missing_options": missing,
        "verified_full_top_k": order is not None,
        "top_k_order": order,
        "top_k_cutoff_absolute_probability": math.exp(cutoff_logprob) if order is not None else None,
        "missing_absolute_mass_upper_bound": unknown_bound,
        "missing_conditional_mass_upper_bound": (
            unknown_bound / (known_mass + unknown_bound) if unknown_bound is not None and known_mass + unknown_bound else None
        ),
        "restricted_top_option": rank,
    }


def audit_cell(raw, prompt, reference, requested_k):
    response = raw.get("response") or {}
    row = {k: raw.get(k) for k in ("class_code", "question_number", "prompt_sha256", "http_status")}
    row["reference"] = reference
    row["response_id"] = response.get("id")
    if raw.get("http_status") != 200:
        return row
    if response.get("model") not in MODEL_IDS or response.get("provider") != "Nebius":
        raise ValueError("Unexpected model or provider in scoring archive")
    choice = response["choice"]
    content = choice.get("content") or ""
    entries = (choice.get("logprobs") or {}).get("content") or []
    decision, decision_index, form = printed_decision(content, prompt)
    start, stop = aligned_visible_positions(entries, content)
    row.update(printed_option=decision, printed_option_format=form, first_visible_logprob_index=start,
               reasoning_tokens=(response.get("usage") or {}).get("completion_tokens_details", {}).get("reasoning_tokens"),
               first_visible_token=None, first_visible_scores=None, printed_option_logprob_index=None)
    if start is None:
        return row
    row.update(first_visible_token=entries[start]["token"], first_visible_scores=score_position(entries[start], requested_k),
               first_visible_token_is_bare_option=packed_token(entries[start]) in (b"A", b"B", b"C", b"D", b"E"))
    if decision_index is not None:
        target_byte = len(content[:decision_index].encode("utf-8"))
        offset = 0
        for index in range(start, stop):
            offset += len(packed_token(entries[index]))
            if target_byte < offset:
                row["printed_option_logprob_index"] = index
                break
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("folder", "frozen-scorer", "rubric"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    primary = read_json(args.folder / "native-responses.json")
    recovery = read_json(args.folder / "native-responses-recovery.json")
    frozen = read_json(args.frozen_scorer)
    rubric = read_json(args.rubric)
    if sha256(args.frozen_scorer.read_bytes()).hexdigest() != primary["frozen_prompt_source_sha256"]:
        raise ValueError("Scoring prompt source has changed")
    if sha256(args.rubric.read_bytes()).hexdigest() != primary["reference_rubric_sha256"]:
        raise ValueError("Reference rubric has changed")
    if primary["request_parameters"] != recovery["request_parameters"] or frozen["class_codes"] != rubric["class_codes"]:
        raise ValueError("Scoring parameters or auxiliary class order disagree")
    raw = {(r["class_code"], r["question_number"]): r for r in primary["rows"]}
    for r in recovery["rows"]:
        slot = (r["class_code"], r["question_number"])
        if raw[slot]["http_status"] != 429 or r["source_http_status"] != 429:
            raise ValueError("Recovery must use only previously rate-limited cells")
        raw[slot] = r

    rows = []
    for class_index, code in enumerate(frozen["class_codes"]):
        for question_index, prompt in enumerate(frozen["scoring_prompts"][class_index]):
            record = raw[(code, question_index + 1)]
            if sha256(prompt.encode("utf-8")).hexdigest() != record["prompt_sha256"]:
                raise ValueError("A request does not match its frozen prompt")
            rows.append(audit_cell(record, prompt, rubric["expected_options_by_question"][question_index][class_index],
                                   primary["request_parameters"]["top_logprobs"]))

    scored = [row for row in rows if row.get("first_visible_scores")]
    ranked = [row for row in scored if row["first_visible_scores"]["restricted_top_option"] is not None]
    printed = [row for row in rows if row.get("printed_option") is not None]
    bare = [row for row in printed if row["printed_option_format"] == "bare"]
    truncated = [row for row in scored if row["first_visible_scores"]["missing_options"]]
    bounded = [row for row in truncated if row["first_visible_scores"]["missing_conditional_mass_upper_bound"] is not None]
    key = lambda row: row["first_visible_scores"]["missing_conditional_mass_upper_bound"]
    highest = max(bounded, key=key, default=None)
    score_correct = lambda row: row["first_visible_scores"]["restricted_top_option"] == row["reference"]
    answer_correct = lambda row: row["printed_option"] == row["reference"]
    summary = {
        "cells": len(rows),
        "successful_responses": sum(row["http_status"] == 200 for row in rows),
        "aligned_visible_start": len(scored),
        "first_visible_token_bare_letter": sum(row["first_visible_token_is_bare_option"] for row in scored),
        "aligned_start_after_internal_logprob_prefix": sum(row["first_visible_logprob_index"] > 0 for row in scored),
        "observed_option_count_distribution": dict(Counter(5 - len(row["first_visible_scores"]["missing_options"]) for row in scored)),
        "complete_rows": len(scored) - len(truncated),
        "observed_count_of_240_bare_option_probabilities": sum(5 - len(row["first_visible_scores"]["missing_options"]) for row in scored),
        "ranked_rows": len(ranked),
        "restricted_correct": sum(map(score_correct, ranked)),
        "restricted_correct_by_class": {code: sum(score_correct(r) for r in ranked if r["class_code"] == code) for code in frozen["class_codes"]},
        "strict_bare_visible_answers": len(bare),
        "strict_bare_correct": sum(map(answer_correct, bare)),
        "explicit_visible_answers": len(printed),
        "explicit_visible_correct": sum(map(answer_correct, printed)),
        "printed_answer_position_differs_from_visible_start": sum(
            row.get("printed_option_logprob_index") is not None and row["printed_option_logprob_index"] != row["first_visible_logprob_index"]
            for row in printed
        ),
        "truncated_rows": len(truncated),
        "truncated_rows_with_bound": len(bounded),
        "missing_conditional_mass_upper_bound_median": median(map(key, bounded)) if bounded else None,
        "missing_conditional_mass_upper_bound_maximum": {"class_code": highest["class_code"],
                                                           "question_number": highest["question_number"],
                                                           "bound": key(highest)} if highest else None,
        "sum_reported_cost_usd": math.fsum(((r.get("response") or {}).get("usage") or {}).get("cost") or 0 for r in raw.values()),
    }
    result = {"request_parameters": primary["request_parameters"], "summary": summary, "rows": rows}
    (args.folder / "scoring-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
