"""Build teacher-answer correctness controls from a Qwen signature and a Muse score audit."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path


LETTERS = "ABCDE"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(path):
    return {"file": path.as_posix(), "sha256": sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def winner(row):
    return LETTERS[row.index(max(row))]


def validate_probability(row):
    if len(row) != len(LETTERS) or not all(math.isfinite(p) and 0 <= p <= 1 for p in row):
        raise ValueError("Every row must contain five finite probabilities")
    if not math.isclose(math.fsum(row), 1.0, abs_tol=1e-6):
        raise ValueError("Option probabilities must sum to one")


def swaps(source, desired, codes, *, require_strict_top1):
    output = []
    changes = []
    for code, rows, targets in zip(codes, source, desired, strict=True):
        block = []
        for number, (row, target) in enumerate(zip(rows, targets, strict=True), start=1):
            validate_probability(row)
            adjusted = list(row)
            original = winner(row)
            if original != target:
                source_index, target_index = LETTERS.index(original), LETTERS.index(target)
                adjusted[source_index], adjusted[target_index] = adjusted[target_index], adjusted[source_index]
                changes.append({"class_code": code, "question_number": number, "before": original, "requested": target,
                                "after": winner(adjusted), "values_changed": row != adjusted,
                                "probabilities_exchanged": [row[source_index], row[target_index]]})
            if (sorted(adjusted) != sorted(row) or adjusted[LETTERS.index(target)] != max(adjusted)
                    or (require_strict_top1 and winner(adjusted) != target)):
                raise ValueError("A swap failed to preserve the row's probability multiset or set its answer")
            block.append(adjusted)
        output.append(block)
    return output, changes


def agreement(matrix, references):
    pairs = [(row, target) for block, expected in zip(matrix, references, strict=True)
             for row, target in zip(block, expected, strict=True)]
    return {"strict_top1": sum(winner(row) == target for row, target in pairs),
            "reference_is_a_maximum": sum(row[LETTERS.index(target)] == max(row) for row, target in pairs)}


def muse_estimate(record, prompt, expected):
    if record["prompt_sha256"] != sha256(prompt.encode("utf-8")).hexdigest() or record["reference"] != expected:
        raise ValueError("Muse scores don't match the frozen prompt and reference")
    if record["http_status"] != 200:
        raise ValueError("Muse response did not succeed")
    source = record["first_visible_scores"]
    native = source["native_option_probabilities"]
    missing = [letter for letter in LETTERS if native[letter] is None]
    if missing != source["missing_options"]:
        raise ValueError("Archived missing-option metadata disagrees with the probability values")
    observed_mass = math.fsum(value for value in native.values() if value is not None)
    if observed_mass <= 0 or not math.isclose(observed_mass, source["observed_bare_option_mass"], abs_tol=1e-12):
        raise ValueError("Archived observed A–E mass is inconsistent")
    cutoff = source["top_k_cutoff_absolute_probability"]
    if missing and (not source["verified_full_top_k"] or cutoff is None or not 0 < cutoff < 1):
        raise ValueError("Missing Muse options need a verified positive top-20 cutoff")
    maximum_missing_mass = len(missing) * cutoff if missing else 0.0
    maximum_missing_share = maximum_missing_mass / (observed_mass + maximum_missing_mass)
    if missing and not math.isclose(maximum_missing_share, source["missing_conditional_mass_upper_bound"], abs_tol=1e-12):
        raise ValueError("Derived Muse probability bound disagrees with the archived audit")

    absolute = {letter: native[letter] if native[letter] is not None else 0.5 * cutoff for letter in LETTERS}
    normalizer = math.fsum(absolute.values())
    estimate = [absolute[letter] / normalizer for letter in LETTERS]
    if not missing and estimate != [source["exact_a_to_e_conditional_probabilities"][letter] for letter in LETTERS]:
        raise ValueError("A complete native conditional probability changed")
    bounds = {
        letter: [0.0, cutoff / (observed_mass + cutoff)] if letter in missing else [
            native[letter] / (observed_mass + maximum_missing_mass), native[letter] / observed_mass,
        ] for letter in LETTERS
    }
    for value, (low, high) in zip(estimate, bounds.values(), strict=True):
        if not low - 1e-12 <= value <= high + 1e-12:
            raise ValueError("Estimated probability is outside its native-data bounds")
    validate_probability(estimate)
    if winner(estimate) != source["restricted_top_option"]:
        raise ValueError("Imputation changed the top option established by the native rankings")
    return estimate, {"class_code": record["class_code"], "question_number": record["question_number"],
                      "native_absolute_probabilities": native, "top_20_absolute_cutoff": cutoff,
                      "estimated_absolute_probabilities": absolute, "conditional_probability_bounds": bounds,
                      "missing_conditional_mass_upper_bound": maximum_missing_share}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("questions", "rubric", "qwen", "muse-audit", "out"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    qwen, bank, rubric = read(args.qwen), read(args.questions), read(args.rubric)
    codes = qwen["class_codes"]
    if qwen["questions"] != bank or codes != rubric["class_codes"] or qwen["answer_labels"] != list(LETTERS):
        raise ValueError("Frozen bank, rubric, and teacher rows use different orders")
    qwen_ref = fingerprint(args.qwen)
    references = [list(block) for block in zip(*rubric["expected_options_by_question"], strict=True)]
    template = {"class_codes": codes, "class_names": qwen["class_names"], "questions": bank, "answer_labels": list(LETTERS),
                "bank": fingerprint(args.questions), "rubric": fingerprint(args.rubric)}
    raw = qwen["probabilities"]
    before = [[winner(row) for row in block] for block in raw]
    corrected, qwen_changes = swaps(raw, references, codes, require_strict_top1=False)
    muse = read(args.muse_audit)
    records = {(r["class_code"], r["question_number"]): r for r in muse["rows"]}
    expected_rows = {(code, number) for code in codes for number in range(1, len(bank) + 1)}
    if len(records) != len(muse["rows"]) or set(records) != expected_rows:
        raise ValueError("Expected one Muse score for every class and question")
    estimated, audit, exact = [], [], 0
    for class_index, code in enumerate(codes):
        block = []
        for index, prompt in enumerate(qwen["scoring_prompts"][class_index]):
            row, detail = muse_estimate(records[(code, index + 1)], prompt, references[class_index][index])
            exact += sum(value is None for value in detail["native_absolute_probabilities"].values()) == 0
            block.append(row)
            audit.append(detail)
        estimated.append(block)
    observed = sum(sum(value is not None for value in item["native_absolute_probabilities"].values()) for item in audit)
    muse_forced, muse_changes = swaps(estimated, before, codes, require_strict_top1=True)
    conditions = {"qwen3_native": raw, "qwen3_corrected": corrected, "muse_native_estimate": estimated, "muse_qwen3_argmax": muse_forced}
    agreements = {name: agreement(matrix, references) for name, matrix in conditions.items()}
    qwen_file = out / "qwen3_corrected.json"
    write(qwen_file, {**template, "probabilities": corrected, "source": qwen_ref,
                      "derivation": "Exchange the highest and reference probabilities on each disagreeing row."})
    muse_file = out / "muse_native_estimate.json"
    forced_file = out / "muse_qwen3_argmax.json"
    write(muse_file, {**template, "probabilities": estimated, "source": fingerprint(args.muse_audit),
                      "derivation": "Use half the verified top-20 cutoff per missing option, then normalize over A–E."})
    write(forced_file, {**template, "probabilities": muse_forced, "source": fingerprint(muse_file),
                        "argmax_source": qwen_ref,
                        "derivation": "Exchange option probabilities so that each highest-ranked option matches the Qwen signature."})
    write(out / "generation_audit.json", {
        "sources": [fingerprint(path) for path in (args.questions, args.rubric, args.qwen, args.muse_audit)],
        "outputs": [fingerprint(path) for path in (qwen_file, muse_file, forced_file)],
        "class_codes": codes, "answer_labels": list(LETTERS), "shape": [len(codes), len(bank), len(LETTERS)],
        "agreement_with_rubric": agreements,
        "top1_tie_break": "First option in A-to-E order, matching the archived scorer.",
        "swaps_preserve_each_row_probability_multiset_exactly": True,
        "qwen3_corrected_swaps": qwen_changes, "muse_qwen3_argmax_swaps": muse_changes,
        "muse_exact_rows": exact, "muse_truncated_rows": len(expected_rows) - exact,
        "muse_native_option_probabilities_observed": observed,
        "muse_row_bounds_and_imputation": audit,
    })
    for path in (qwen_file, muse_file, forced_file):
        print(f"Ready: {path} ({fingerprint(path)['sha256']})", flush=True)


if __name__ == "__main__":
    main()
