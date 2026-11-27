"""Materialize the two frozen per-record PTB-XL+ Uni-G interval prompts."""

import argparse
from pathlib import Path

from .teacher import write_json


FEATURES = ("RR_Mean_Global", "RR_StdDev_Global")
QUESTIONS = {
    "RR_Mean_Global": {
        "question": "Which supplied range contains the mean duration, in milliseconds, of the R-to-R intervals that Uni-G identifies as sinus intervals on this ECG?",
        "waveform_operation": "Where sinus intervals can be identified on this ECG, measure the time between the R peaks bounding each such interval and average those durations."
    },
    "RR_StdDev_Global": {
        "question": "Which supplied range contains the standard deviation, in milliseconds, of the durations of the R-to-R intervals that Uni-G identifies as normal intervals on this ECG?",
        "waveform_operation": "Where normal intervals can be identified on this ECG, measure the time between the R peaks bounding each such interval and estimate the standard deviation of those durations around their mean."
    }
}


def validate_questions(questions, fields=FEATURES):
    if not isinstance(questions, dict) or set(questions) != set(fields):
        raise ValueError(f"Expected question stems for {fields}")
    for item in questions.values():
        if not isinstance(item, dict) or set(item) != {"question", "waveform_operation"}:
            raise ValueError("Each RR question requires a stem and a waveform operation")
        if any(not isinstance(value, str) or not value.strip() for value in item.values()):
            raise ValueError("Empty RR question text")
        if not item["question"].rstrip().endswith("?"):
            raise ValueError("Expected a question stem")
    return questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, validate_questions(QUESTIONS))


if __name__ == "__main__":
    main()
