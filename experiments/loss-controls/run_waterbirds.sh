#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash experiments/loss-controls/run_waterbirds.sh BACKBONE IMAGES AUTHOR_SOURCE OUTPUT

Backbones: resnet18, mobilenetv2.
IMAGES must contain official Waterbirds-95 metadata.csv and numbered species directories.
Trains the completed subset documented in WATERBIRDS.md.
Environment: PYTHON (default python), DEVICE (default cuda), WORKERS (default 4).
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi
if (( $# != 4 )); then
  usage >&2
  exit 2
fi
backbone=$1
images=$2
author=$3
out=$4
case "$backbone" in
  resnet18) arms=(ce raw-onehot prob-onehot squentropy ecoc-soft-ce) ;;
  mobilenetv2) arms=(ce raw-onehot ecoc-soft-ce) ;;
  *) usage >&2; exit 2 ;;
esac

python=${PYTHON:-python}
code="$out/cub200-five-way-code.json"
"$python" -m ecg_lavid.visual_benchmark_data --dataset cub200 --author-source "$author" \
  --signature "$author/questions/cub200/logits/qwen2.5-7b_chat_50-5.json" --out "$code"
"$python" -m ecg_lavid.train_visual_loss_matrix --dataset waterbirds --architecture "$backbone" \
  --images "$images" --author-source "$author" --code "$code" --out "$out" \
  --seed 17 --device "${DEVICE:-cuda}" --workers "${WORKERS:-4}" --waterbirds-beta 40 --arms "${arms[@]}"
