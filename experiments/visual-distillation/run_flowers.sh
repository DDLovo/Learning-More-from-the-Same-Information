#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash experiments/visual-distillation/run_flowers.sh FLOWERS_ROOT AUTHOR_SOURCE OUTPUT

Environment: PYTHON (default python), DEVICE (default cuda), WORKERS (optional).
Optional FLOWERS_RGB_CACHE and TEACHER_LOGIT_CACHE specify cache directories.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi
if (( $# != 3 )); then
  usage >&2
  exit 2
fi
images=$1
author=$2
out=$3
python=${PYTHON:-python}
device=${DEVICE:-cuda}
qwen="$author/questions/flowers/logits/qwen2.5-7b_chat_50-5.json"
code="$out/flowers-code.json"
teacher="$out/visual-teachers/flowers/seed-17/author-best.pt"
common=(--images "$images" --author-source "$author" --device "$device")
rgb=()
logits=()
if [[ -n ${FLOWERS_RGB_CACHE:-} ]]; then
  rgb=(--training-rgb-cache "$FLOWERS_RGB_CACHE")
fi
if [[ -n ${TEACHER_LOGIT_CACHE:-} ]]; then
  logits=(--teacher-logit-cache "$TEACHER_LOGIT_CACHE")
fi

"$python" -m ecg_lavid.visual_benchmark_data --dataset flowers --author-source "$author" --signature "$qwen" --out "$code"
"$python" -m ecg_lavid.train_visual_lavid_teacher --dataset flowers "${common[@]}" "${rgb[@]}" \
  --out "$out/visual-teachers" --seed 17 --workers "${WORKERS:-8}"

for seed in 17 29 43; do
  "$python" -m ecg_lavid.train_visual_lavid_ls "${common[@]}" "${rgb[@]}" "${logits[@]}" \
    --out "$out/flowers-hybrids" --teacher-checkpoint "$teacher" --target-kind qwen --target-file "$qwen" \
    --seed "$seed" --workers "${WORKERS:-4}"
done
for seed in 17 29 43; do
  "$python" -m ecg_lavid.train_visual_lavid_ls "${common[@]}" "${rgb[@]}" "${logits[@]}" \
    --out "$out/flowers-hybrids" --teacher-checkpoint "$teacher" --target-kind ecoc --target-file "$code" \
    --seed "$seed" --workers "${WORKERS:-4}"
  "$python" -m ecg_lavid.train_visual_benchmark --dataset flowers --architecture resnet18 --method ecoc-only \
    "${common[@]}" "${rgb[@]}" --target "$code" --out "$out/visual-grid" --seed "$seed" --workers "${WORKERS:-4}"
done
