#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash experiments/loss-controls/run_visual.sh [--match-resnet18-h] DATASET BACKBONE IMAGES AUTHOR_SOURCE OUTPUT [VIT_WEIGHTS]

Datasets: flowers, cub200.
Backbones: resnet18, mobilenetv2, vit_small_patch16. ViT requires VIT_WEIGHTS.
Completed runs per dataset: CNNs A–I and G★; ViT-S/16 A, B, C, E, G.
Use --match-resnet18-h for the completed ViT-S/16 G and H calibration runs.
Use run_waterbirds.sh for the completed Waterbirds-95 subset.
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
matched=false
if [[ ${1:-} == --match-resnet18-h ]]; then
  matched=true
  shift
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
if [[ $matched == true ]]; then
  [[ $backbone == vit_small_patch16 ]] || bad_arguments
  out="$out/resnet18-h-matched"
fi

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
if [[ $matched == true ]]; then
  match_args=(--match-resnet18-h --gradient-calibration "$out/$dataset-reference.json")
  "$python" -m ecg_lavid.train_visual_loss_matrix "${matrix[@]}" "${match_args[@]}" --calibrate-gradmatch
  "$python" -m ecg_lavid.train_visual_loss_matrix "${matrix[@]}" "${match_args[@]}" --arms ecoc-soft-ce ecoc
elif [[ $backbone == vit_small_patch16 ]]; then
  "$python" -m ecg_lavid.train_visual_loss_matrix "${matrix[@]}" --arms ce raw-onehot prob-onehot squentropy ecoc-soft-ce
else
  "$python" -m ecg_lavid.train_visual_loss_matrix "${matrix[@]}" \
    --calibrate-gradmatch --gradient-calibration "$calibration"
  "$python" -m ecg_lavid.train_visual_loss_matrix "${matrix[@]}" --arms "${arms[@]}"
  "$python" -m ecg_lavid.train_visual_loss_matrix "${matrix[@]}" \
    --arms ecoc-soft-ce-gradmatch --gradient-calibration "$calibration"
fi
