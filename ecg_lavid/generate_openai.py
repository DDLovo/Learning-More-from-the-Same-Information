"""Generate a structured ECG question bank using the official OpenAI Responses API."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path

from . import CLASSES
from .teacher import LETTERS, validate_questions, write_json


QUESTION_COUNT = 12
BASE_URL = "https://api.openai.com/v1"


def make_prompt():
    classes = ", ".join(CLASSES.values())
    return f"""Design a bank of {QUESTION_COUNT} multiple-choice questions for a study of transferring concept knowledge to a waveform-only ECG classifier. The publicly named benchmark classes are {classes}. Here the unqualified sinus-rhythm label means ordinary adult ventricular rate without a separate fast or slow rhythm designation. A rhythm label describes a rhythm and may coexist with other ECG findings.

Each question will be scored separately after the same literal placeholder <object> is replaced by each of the four rhythm names in turn. Ask about one feature that can be observed or estimated from a typical adult 10-second resting 12-lead surface ECG. Keep exactly one literal <object> in each question, in the position for the rhythm name. Form each question as a complete sentence. The five choices should be concise descriptions of distinct possible ECG observations, without diagnosis or class names in the question or the choices. Every benchmark class should have a defensible best option for a typical uncomplicated presentation. Related classes may share an option. When a rhythm does not specify the feature, allow an appropriate variable or unspecified choice. Avoid overlapping numeric ranges and answers that are simultaneously true for the same presentation.

Cover concepts including typical ventricular rate, R-R interval regularity, organized atrial activity and P-wave morphology, and the temporal relationship between atrial and ventricular deflections. Include at least one rate item that gives the three sinus classes distinct choices and at least one item that distinguishes atrial fibrillation from the other three by a feature observable in the waveform. Seek different observations and reasoning operations where possible. Describe typical patterns and account for important limits such as the variability of ventricular rate in fibrillation. Keep questions independent of patient age within adulthood, therapy, prognosis, disease background, and individual records. Vary answer order across the bank. Return questions and answer choices, with no answers or explanations."""


def response_schema():
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array", "minItems": QUESTION_COUNT, "maxItems": QUESTION_COUNT,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "options": {"type": "array", "minItems": len(LETTERS), "maxItems": len(LETTERS),
                                    "items": {"type": "string"}},
                    },
                    "required": ["question", "options"], "additionalProperties": False,
                },
            },
        },
        "required": ["questions"], "additionalProperties": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="A reasoning model available through the official API")
    parser.add_argument("--reasoning-effort", default="high", choices=("low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import httpx
    from openai import APIConnectionError, APIStatusError, OpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or any(ch.isspace() for ch in api_key):
        raise SystemExit("Set OPENAI_API_KEY to an OpenAI API key.")

    prompt = make_prompt()
    output_format = {"type": "json_schema", "name": "ecg_rhythm_mcqs", "strict": True, "schema": response_schema()}
    provenance = {
        "provider": "OpenAI",
        "endpoint": BASE_URL + "/responses",
        "requested_model": args.model,
        "request_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "reasoning_effort": args.reasoning_effort,
        "temperature_requested": None,
        "openai_sdk_version": version("openai"),
        "max_output_tokens": 6_000,
        "store": False,
        "prompt": prompt,
        "response_format": output_format,
    }
    provenance_path = args.out.with_suffix(".provenance.json")
    if args.out.exists() or provenance_path.exists():
        raise SystemExit("Use a new output path so the previous generation attempt is preserved")
    with OpenAI(api_key=api_key, base_url=BASE_URL, timeout=180.0, max_retries=0,
                http_client=httpx.Client(trust_env=True)) as client:
        try:
            response = client.responses.create(
                model=args.model, input=prompt, reasoning={"effort": args.reasoning_effort},
                text={"format": output_format}, max_output_tokens=provenance["max_output_tokens"], store=False,
            )
        except APIStatusError as error:
            provenance.update(status="http_error", http_status=error.status_code)
            write_json(provenance_path, provenance)
            raise SystemExit(f"Official OpenAI API returned HTTP {error.status_code}") from None
        except APIConnectionError:
            provenance["status"] = "connection_error"
            write_json(provenance_path, provenance)
            raise SystemExit("Could not connect to the official OpenAI API") from None

    provenance.update(
        status=response.status, resolved_model=response.model,
        created_at_utc=datetime.fromtimestamp(response.created_at, timezone.utc).isoformat(),
        response_id=response.id, usage=response.usage.model_dump() if response.usage else None,
        temperature_returned=getattr(response, "temperature", None), raw_response=response.output_text,
    )
    write_json(provenance_path, provenance)
    if response.status != "completed" or not response.output_text:
        raise SystemExit(f"OpenAI generation did not complete: {response.status}")
    questions = validate_questions(json.loads(response.output_text)["questions"], QUESTION_COUNT)
    write_json(args.out, questions)
    print(f"Saved {len(questions)} questions from {response.model} to {args.out}", flush=True)


if __name__ == "__main__":
    main()
