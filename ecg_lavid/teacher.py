"""Generate ECG rhythm MCQs and extract class-level causal-LM option distributions."""

import argparse
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import re

import torch

from . import CLASSES


LETTERS = "ABCDE"
PLACEHOLDER = "<object>"
QUESTION_TOPICS = (
    "adult ventricular rate category",
    "beat-to-beat regularity of the R-R intervals",
    "presence and morphology of distinct P waves in lead II",
    "atrial activity visible between ventricular complexes",
    "timing of discrete P waves relative to QRS complexes",
    "whether a standard PR interval can be measured",
    "repetition of organized atrial activations across consecutive cycles",
    "duration of typical ventricular cycles",
    "agreement of discrete atrial event times across simultaneous leads",
    "number of ventricular complexes in a fixed-duration strip",
    "whether a pre-QRS atrial deflection is preserved in averaged R-aligned beats",
    "relationship between the rhythm label alone and QRS duration",
)


def generation_prompt(question_count):
    class_names = ", ".join(CLASSES.values())
    return f"""Create exactly {question_count} multiple-choice questions that elicit useful distinctions among these ECG rhythm classes: {class_names}.

The observations are 10-second, resting, 12-lead adult ECG waveforms. Ask about one attribute observable on the waveform per question. Cover different concepts, including atrial activity, ventricular timing and rate, and the relationship between atrial activity and QRS complexes. Use typical presentations and allow a variable or uncharacteristic feature where the rhythm alone does not specify the value. Keep classes that share a trait grouped in the same answer option. Avoid assuming that a rhythm implies normal health. Exclude treatment, demographics, other diagnoses, and prognosis. Do not refer to any example ECG or report.

Write each question as a sentence with exactly one literal {PLACEHOLDER} standing for the rhythm name. Give exactly five distinct, mutually exclusive options as plain text strings. Neither the question nor the options may name a rhythm class. Do not include a correct answer, an explanation, or option letters.

Return only a valid JSON array with {question_count} objects. Every object must have exactly two fields: a string \"question\" and an array \"options\" containing five strings."""


def validate_questions(questions, expected_count=None):
    if not isinstance(questions, list) or not questions:
        raise ValueError("Expected a nonempty JSON array")
    if expected_count is not None and len(questions) != expected_count:
        raise ValueError(f"Expected {expected_count} questions, got {len(questions)}")
    rhythm_names = "|".join(re.escape(name) for name in (*CLASSES, *CLASSES.values()))
    class_name_pattern = re.compile(rf"\b(?:{rhythm_names})\b", re.IGNORECASE)
    seen = set()
    validated = []
    for number, item in enumerate(questions, start=1):
        if not isinstance(item, dict) or set(item) != {"question", "options"}:
            raise ValueError(f"Question {number} needs question and options fields")
        question, options = item["question"], item["options"]
        if not isinstance(question, str) or question.count(PLACEHOLDER) != 1:
            raise ValueError(f"Question {number} needs one {PLACEHOLDER}")
        if not isinstance(options, list) or len(options) != len(LETTERS) or not all(isinstance(o, str) for o in options):
            raise ValueError(f"Question {number} needs {len(LETTERS)} string options")
        question = " ".join(question.split())
        options = [" ".join(option.split()) for option in options]
        normalized = [option.casefold() for option in options]
        if "" in normalized or len(set(normalized)) != len(LETTERS):
            raise ValueError(f"Question {number} has empty or duplicate options")
        if any(class_name_pattern.search(text) for text in (question, *options)):
            raise ValueError(f"Question {number} names a class explicitly")
        if PLACEHOLDER in " ".join(options) or question.casefold() in seen:
            raise ValueError(f"Question {number} has a placeholder in the options or duplicates a question")
        seen.add(question.casefold())
        validated.append({"question": question, "options": options})
    return validated


def parse_response(raw):
    # Models sometimes put valid JSON in a Markdown code fence.
    start = raw.find("[")
    if start == -1:
        raise ValueError("Generated response contains no JSON array")
    questions, _ = json.JSONDecoder().raw_decode(raw[start:])
    return questions


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def render_prompt(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
    )


def generate_texts(args, model, tokenizer, prompts):
    rendered = [render_prompt(tokenizer, prompt) for prompt in prompts]
    encoded = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **encoded, do_sample=False, max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id, temperature=None, top_p=None, top_k=None,
        )
    return tokenizer.batch_decode(outputs[:, encoded.input_ids.shape[1]:], skip_special_tokens=True)


def generate_questions(args, model, tokenizer):
    attempts = []
    prompt = generation_prompt(args.question_count)
    for attempt in range(1, 4):
        raw, = generate_texts(args, model, tokenizer, [prompt])
        attempts.append({"attempt": attempt, "prompt": prompt, "raw_response": raw})
        try:
            questions = validate_questions(parse_response(raw), args.question_count)
        except ValueError as error:
            attempts[-1]["validation_error"] = str(error)
            write_json(args.out.with_suffix(".generation.json"), attempts)
            print(f"Generation attempt {attempt} failed validation: {error}", flush=True)
            prompt = generation_prompt(args.question_count) + f"\nAn earlier attempt failed validation: {error}. Check this before returning JSON."
        else:
            return questions, attempts
    raise RuntimeError("No valid MCQ bank. The generation log contains all attempts.")


def single_question_prompt(topic, generation_mode):
    classes = ", ".join(CLASSES.values())
    if generation_mode == "per-question-json":
        output_format = "Return one JSON object with these keys and value types: {\"question\": \"[full sentence with <object>]\", \"options\": [\"[description 1]\", \"[description 2]\", \"[description 3]\", \"[description 4]\", \"[description 5]\"]}."
    else:
        output_format = """Write six lines, using these exact labels followed by your own text:
Q: [full sentence about the expected ECG feature in <object>]
A: [observation 1]
B: [observation 2]
C: [observation 3]
D: [observation 4]
E: [observation 5]"""
    return f"""Write ONE multiple-choice question about {topic} on a typical 10-second resting adult 12-lead ECG. Possible rhythm names are {classes}. For this task, the unqualified sinus label refers to the ordinary rate range.

The question should ask a reader to predict a typical, directly observable feature of a rhythm they are given. In the question, keep the exact literal text {PLACEHOLDER} where the rhythm name would go. A reader will replace it with a rhythm name later. Write a complete question. The answer choices are five different, mutually exclusive descriptions of possible ECG observations. Allow a variable feature or a rhythm whose label leaves the feature unspecified when relevant. Use the rhythm names as context for designing the question; leave all names and diagnosis labels out of the question and answer choices. Output just the question and five choices, withholding answers and explanations.

{output_format}"""


def parse_single_question(raw, generation_mode):
    if generation_mode == "per-question-json":
        start = raw.find("{")
        if start == -1:
            raise ValueError("Generated response contains no JSON object")
        question, _ = json.JSONDecoder().raw_decode(raw[start:])
        return question
    fields = {}
    pattern = re.compile(r"^\s*(Q(?:uestion)?|[A-E])\s*[:.)]\s*(.+)$", re.IGNORECASE)
    for line in raw.splitlines():
        match = pattern.match(line.strip().replace("**", ""))
        if match:
            key, value = match.groups()
            key = key.upper()[0]
            if key in fields:
                raise ValueError(f"Generated response repeats field {key}")
            fields[key] = value
    if set(fields) != set("Q" + LETTERS):
        raise ValueError("Generated response needs labeled Q and A–E lines")
    return {"question": fields["Q"], "options": [fields[letter] for letter in LETTERS]}


def generate_individual_questions(args, model, tokenizer):
    topics = QUESTION_TOPICS[:args.question_count]
    prompts = [single_question_prompt(topic, args.generation_mode) for topic in topics]
    questions = []
    records = []
    generation_path = args.out.with_suffix(".generation.json")
    for start in range(0, len(prompts), args.batch_size):
        end = start + args.batch_size
        raw_texts = generate_texts(args, model, tokenizer, prompts[start:end])
        for slot, raw in enumerate(raw_texts, start=start):
            record = {"slot": slot + 1, "topic": topics[slot], "prompt": prompts[slot], "raw_response": raw}
            records.append(record)
            try:
                question = parse_single_question(raw, args.generation_mode)
                questions = validate_questions([*questions, question])
            except ValueError as error:
                record["validation_error"] = str(error)
        write_json(generation_path, records)
        print(f"Generated {len(records)}/{len(topics)} questions, {len(questions)} passed format validation", flush=True)
    if len(questions) != len(topics):
        raise RuntimeError(f"Only {len(questions)}/{len(topics)} questions passed format validation. See {generation_path}")
    return questions, records


def scoring_prompt(question, rhythm):
    instantiated = question["question"].replace(PLACEHOLDER, rhythm)
    options = "\n".join(f"{letter}. {option}" for letter, option in zip(LETTERS, question["options"]))
    return f"For a typical adult 12-lead resting ECG, select the best answer. Reply with the single option letter A, B, C, D, or E.\n\n{instantiated}\n{options}"


def extract_scores(model, tokenizer, questions, batch_size, concept_names=None):
    names = tuple(CLASSES.values()) if concept_names is None else tuple(concept_names)
    option_ids = []
    for letter in LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids) != letter:
            raise ValueError(f"Option {letter} is not a single literal token")
        option_ids.append(ids[0])
    if len(set(option_ids)) != len(LETTERS):
        raise ValueError("Answer letters must use distinct token IDs")

    prompts = [[scoring_prompt(question, name) for question in questions] for name in names]
    rendered = [render_prompt(tokenizer, prompt) for block in prompts for prompt in block]
    prompt_prefix, _, assistant_suffix = rendered[0].partition(prompts[0][0])
    if assistant_suffix.count("<think>") > assistant_suffix.count("</think>"):
        raise ValueError("The chat template starts a thinking block at the scoring position. Use a direct-answer checkpoint.")
    for text in rendered:
        prefix = tokenizer.encode(text, add_special_tokens=False)
        for letter, token_id in zip(LETTERS, option_ids):
            if tokenizer.encode(text + letter, add_special_tokens=False) != [*prefix, token_id]:
                raise ValueError(f"Option {letter} is not a one-token continuation of the chat prompt")

    model_parameters = inspect.signature(model.forward).parameters
    logits_limit = {}
    for argument in ("logits_to_keep", "num_logits_to_keep"):
        if argument in model_parameters:
            logits_limit[argument] = 1

    rows = []
    letter_mass = []
    letter_probabilities = []
    top_tokens = []
    with torch.inference_mode():
        for start in range(0, len(rendered), batch_size):
            encoded = tokenizer(rendered[start:start + batch_size], return_tensors="pt", padding=True,
                                add_special_tokens=False).to(model.device)
            logits = model(**encoded, use_cache=False, **logits_limit).logits[:, -1].float()
            selected = logits[:, option_ids]
            log_normalizer = logits.logsumexp(-1)
            top_logits, top_ids = logits.max(-1)
            rows.append(selected.cpu())
            letter_mass.append((selected.logsumexp(-1) - log_normalizer).exp().cpu())
            letter_probabilities.append((selected - log_normalizer[:, None]).exp().cpu())
            top_tokens.extend({"token_id": token_id, "text": tokenizer.decode([token_id]), "probability": probability}
                              for token_id, probability in zip(top_ids.tolist(), (top_logits - log_normalizer).exp().tolist()))

    scores = torch.cat(rows).view(len(names), len(questions), len(LETTERS))
    masses = torch.cat(letter_mass).view(len(names), len(questions))
    audit = {
        "prefix_before_user_message": prompt_prefix,
        "assistant_suffix_after_user_message": assistant_suffix,
        "chat_template_sha256": hashlib.sha256(json.dumps(tokenizer.chat_template, sort_keys=True).encode("utf-8")).hexdigest(),
        "absolute_option_probabilities": torch.cat(letter_probabilities).view_as(scores).tolist(),
        "top_full_vocabulary_token": [top_tokens[start:start + len(questions)]
                                      for start in range(0, len(top_tokens), len(questions))],
    }
    print(f"Scored {len(rendered)} class-question pairs", flush=True)
    return scores, masses, option_ids, prompts, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--question-count", type=int, default=12)
    parser.add_argument("--questions", type=Path, help="Score an existing JSON MCQ bank instead of generating one")
    parser.add_argument("--concepts", type=Path, help="JSON with aligned class_codes and class_names for a provided bank")
    parser.add_argument("--generation-mode", choices=("bank", "per-question", "per-question-json"), default="bank")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()
    if args.question_count < 1 or args.batch_size < 1 or (args.max_new_tokens is not None and args.max_new_tokens < 1):
        parser.error("Question, batch and token limits must be positive")
    if args.generation_mode != "bank" and args.question_count > len(QUESTION_TOPICS):
        parser.error(f"Per-question generation supports at most {len(QUESTION_TOPICS)} questions")
    class_codes, class_names = list(CLASSES), list(CLASSES.values())
    if args.concepts:
        if not args.questions:
            parser.error("--concepts requires --questions")
        concepts = json.loads(args.concepts.read_text(encoding="utf-8-sig"))
        class_codes, class_names = concepts["class_codes"], concepts["class_names"]
        if (not class_codes or len(class_codes) != len(class_names)
                or any(not isinstance(value, str) or not value.strip() for value in (*class_codes, *class_names))
                or len(set(class_codes)) != len(class_codes)):
            parser.error("Concepts must contain equally long lists of nonempty codes and names, with distinct codes")
    args.max_new_tokens = args.max_new_tokens or (240 * args.question_count + 32 if args.generation_mode == "bank" else 256)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    else:
        dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=dtype).to(device).eval()
    if args.questions:
        questions = validate_questions(json.loads(args.questions.read_text(encoding="utf-8-sig")))
        attempts = []
    elif args.generation_mode == "bank":
        questions, attempts = generate_questions(args, model, tokenizer)
    else:
        questions, attempts = generate_individual_questions(args, model, tokenizer)

    logits, letter_mass, option_ids, prompts, token_audit = extract_scores(model, tokenizer, questions, args.batch_size, class_names)
    probabilities = logits.softmax(dim=-1)
    entropy = -(probabilities * logits.log_softmax(dim=-1)).sum(dim=-1)
    difference = probabilities[:, None] - probabilities[None, :]
    pairwise_distance = difference.square().mean(dim=(-1, -2)).sqrt()
    if not probabilities.isfinite().all():
        raise ValueError("Teacher option scores are not finite")
    artifact = {
        "class_codes": class_codes,
        "class_names": class_names,
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "device": str(device),
        "dtype": str(dtype),
        "software": {"torch": str(torch.__version__), "transformers": version("transformers")},
        "question_source": "provided_file" if args.questions else "generated_by_teacher",
        "question_file": args.questions.name if args.questions else None,
        "generation_mode": args.generation_mode if args.questions is None else None,
        "generation": attempts,
        "questions": questions,
        "scoring_prompts": prompts,
        "answer_labels": list(LETTERS),
        "answer_token_ids": option_ids,
        "temperature": 1.0,
        "option_logits": logits.tolist(),
        "probabilities": probabilities.tolist(),
        "diagnostics": {
            "mean_entropy_nats": entropy.mean().item(),
            "per_class_entropy_nats": entropy.mean(-1).tolist(),
            "option_mass_over_full_vocabulary": letter_mass.tolist(),
            "pairwise_rms_distance": pairwise_distance.tolist(),
            **token_audit,
        },
    }
    if args.concepts:
        artifact["concept_file"] = args.concepts.name
        artifact["concept_file_sha256"] = hashlib.sha256(args.concepts.read_bytes()).hexdigest()
    write_json(args.out, artifact)
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
