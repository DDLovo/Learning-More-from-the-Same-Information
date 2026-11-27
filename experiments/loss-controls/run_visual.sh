#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash experiments/loss-controls/run_visual.sh DATASET BACKBONE IMAGES AUTHOR_SOURCE OUTPUT [VIT_WEIGHTS]

Datasets: flowers, cub200.
Backbones: resnet18, mobilenetv2, vit_small_patch16. ViT requires VIT_WEIGHTS.
Environment: PYTHON (default python), DEVICE (default cuda), WORKERS (optional).
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
(( $# == 5 || $# == 6 )) || bad_arguments
dataset=$1
backbone=$2
images=$3
author=$4
out=$5
case "$dataset" in
  flowers|cub200) ;;
  *) bad_arguments ;;
esac
weights=()
case "$backbone" in
  resnet18|mobilenetv2) (( $# == 5 )) || bad_arguments ;;
  vit_small_patch16)
    (( $# == 6 )) || bad_arguments
    weights=(--weights "$6")
    ;;
  *) bad_arguments ;;
esac

python=${PYTHON:-python}
device=${DEVICE:-cuda}
arms=(ce raw-onehot prob-onehot ce-same-onehot squentropy ce-independent-onehot ecoc-soft-ce ecoc ecoc-constant)
common=(--dataset "$dataset" --images "$images" --author-source "$author" --out "$out" --seed 17 --device "$device")

if [[ $backbone == resnet18 ]]; then
  precision=bfloat16
  if [[ $device == cpu ]]; then
    precision=float32
  fi
  "$python" -m ecg_lavid.train_visual_lavid "${common[@]}" --precision "$precision" \
    --workers "${WORKERS:-8}" --arms "${arms[@]}" ecoc-soft-ce-gradmatch
  exit 0
fi

signature="$author/questions/$dataset/logits/qwen2.5-7b_chat_50-5.json"
code="$out/$dataset-five-way-code.json"
calibration="$out/$dataset-$backbone-gstar.json"
"$python" -m ecg_lavid.visual_benchmark_data --dataset "$dataset" --author-source "$author" \
  --signature "$signature" --out "$code"

matrix=(--architecture "$backbone" --code "$code" --workers "${WORKERS:-4}" "${weights[@]}" "${common[@]}")
"$python" -m ecg_lavid.train_visual_loss_matrix "${matrix[@]}" \
  --calibrate-gradmatch --gradient-calibration "$calibration"
"$python" -m ecg_lavid.train_visual_loss_matrix "${matrix[@]}" --arms "${arms[@]}"
"$python" -m ecg_lavid.train_visual_loss_matrix "${matrix[@]}" \
  --arms ecoc-soft-ce-gradmatch --gradient-calibration "$calibration"
