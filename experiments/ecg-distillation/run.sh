#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash experiments/ecg-distillation/run.sh prepare OUTPUT PTBXL_ROOT [DOWNLOAD_WORKERS]
  bash experiments/ecg-distillation/run.sh STAGE OUTPUT

Stages: questions, score-qwen, score-muse, train-qwen, train-muse, score-muse-text, train-muse-text, controls, codes
Environment: PYTHON (default python), DEVICE (default cuda).
The Muse scoring stages read OPENROUTER_API_KEY.
EOF
}

bad_arguments() {
  usage >&2
  exit 2
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi
(( $# >= 2 )) || bad_arguments
stage=$1
out=$2
shift 2
case "$stage" in
  prepare) (( $# == 1 || $# == 2 )) || bad_arguments ;;
  questions|score-qwen|score-muse|train-qwen|train-muse|score-muse-text|train-muse-text|controls|codes)
    (( $# == 0 )) || bad_arguments ;;
  *) bad_arguments ;;
esac

python=${PYTHON:-python}
device=${DEVICE:-cuda}
questions="$out/ecg-questions.json"
rubric="$out/ecg-rubric.json"
qwen="$out/ecg-qwen.json"
muse="$out/ecg-muse"
muse_text="$out/ecg-muse-text"
prepared="$out/ptbxl-rhythm.npz"

write_questions() {
  "$python" -m ecg_lavid.rhythm_question_bank --bank astra --out "$questions" --rubric "$rubric"
}

train_student() {
  "$python" -m ecg_lavid.train --data "$prepared" --teacher "$1" --out "$2" \
    --device "$device" --methods ce lavid permuted --seeds 17 29 43 \
    --epochs 25 --fractions 0.1 --lavid-weight 0.4 --validation-only
}

train_muse_format() {
  "$python" -m ecg_lavid.train --data "$prepared" --teacher "$muse_text/$1.json" \
    --out "$out/ecg-muse-format-students/$1" --device "$device" --methods lavid \
    --seeds 17 29 43 71 101 137 --epochs 25 --batch-size 256 --fractions 0.1 \
    --lavid-weight 0.4 --validation-only
}

case "$stage" in
  prepare)
    "$python" -m ecg_lavid.prepare --ptbxl "$1" --download-workers "${2:-0}" --out "$prepared"
    write_questions
    ;;
  questions)
    write_questions
    ;;
  score-qwen)
    "$python" -m ecg_lavid.teacher --model Qwen/Qwen3-4B-Instruct-2507 \
      --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
      --questions "$questions" --out "$qwen" --device "$device"
    ;;
  score-muse)
    "$python" -m ecg_lavid.score_muse_concepts --questions "$questions" \
      --rubric "$rubric" --out "$muse" --cache "$out/ecg-muse-cache"
    ;;
  train-qwen)
    train_student "$qwen" "$out/ecg-qwen-students"
    ;;
  train-muse)
    train_student "$muse/signature.json" "$out/ecg-muse-students"
    ;;
  score-muse-text)
    text_questions="$muse_text/questions.json"
    text_rubric="$muse_text/rubric.json"
    "$python" -m ecg_lavid.rhythm_question_bank --bank discriminative --out "$text_questions" --rubric "$text_rubric"
    "$python" -m ecg_lavid.score_muse_concepts --questions "$text_questions" --rubric "$text_rubric" \
      --out "$muse_text/native" --cache "$muse_text/native-cache"
    "$python" -m ecg_lavid.muse_text_probability --questions "$text_questions" \
      --rubric "$text_rubric" --native-dir "$muse_text/native" --out "$muse_text"
    ;;
  train-muse-text)
    duplicate=$("$python" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["hard_signatures_exactly_identical"])' \
      "$muse_text/parsing_audit.json")
    for signature in native_soft text_soft native_top1_onehot; do
      train_muse_format "$signature"
    done
    if [[ $duplicate == True ]]; then
      printf '%s\n' 'Identical hard targets share the native_top1_onehot runs.'
    else
      train_muse_format text_answer_onehot
    fi
    ;;
  controls)
    "$python" -m ecg_lavid.teacher_correctness --questions "$questions" --rubric "$rubric" \
      --qwen "$qwen" --muse-audit "$muse/scoring-audit.json" --out "$out/ecg-teacher-controls"
    ;;
  codes)
    "$python" -m ecg_lavid.ecoc --out "$out/ecg-codebooks"
    "$python" -m ecg_lavid.train --data "$prepared" --codebook "$out/ecg-codebooks/five_way.json" \
      --out "$out/ecg-code-students" --device "$device" --methods ce lavid \
      --seeds 17 29 43 71 101 137 --epochs 25 --fractions 0.1 --lavid-weight 0.4 --validation-only
    ;;
esac
