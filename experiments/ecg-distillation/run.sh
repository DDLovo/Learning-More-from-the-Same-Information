#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash experiments/ecg-distillation/run.sh prepare OUTPUT PTBXL_ROOT [DOWNLOAD_WORKERS]
  bash experiments/ecg-distillation/run.sh STAGE OUTPUT

Stages: questions, score-qwen, score-muse, train-qwen, train-muse, controls, codes
Environment: PYTHON (default python), DEVICE (default cuda).
The score-muse stage reads OPENROUTER_API_KEY.
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
  questions|score-qwen|score-muse|train-qwen|train-muse|controls|codes) (( $# == 0 )) || bad_arguments ;;
  *) bad_arguments ;;
esac

python=${PYTHON:-python}
device=${DEVICE:-cuda}
questions="$out/ecg-questions.json"
rubric="$out/ecg-rubric.json"
qwen="$out/ecg-qwen.json"
muse="$out/ecg-muse"
prepared="$out/ptbxl-rhythm.npz"

write_questions() {
  "$python" -m ecg_lavid.rhythm_question_bank --bank astra --out "$questions" --rubric "$rubric"
}

train_student() {
  "$python" -m ecg_lavid.train --data "$prepared" --teacher "$1" --out "$2" \
    --device "$device" --methods ce lavid permuted --seeds 17 29 43 \
    --epochs 25 --fractions 0.1 --lavid-weight 0.4 --validation-only
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
