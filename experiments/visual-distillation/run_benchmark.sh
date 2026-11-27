#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash experiments/visual-distillation/run_benchmark.sh DATASET STUDENT IMAGES AUTHOR_SOURCE OUTPUT SEED [SIGNATURE]

Datasets: flowers, cub200, fgvc_aircraft, oxfordpets, stanfordcars, caltech101.
Students: resnet18, mobilenetv2, shufflenetv2. Seeds: 17, 29, 43.
Provide SIGNATURE for caltech101 after reconstructing its Qwen target.
Environment: PYTHON (default python), DEVICE (default cuda), WORKERS (optional).
METHOD selects all (default), lavid-ls-qwen, lavid-ls-ecoc, or ecoc-only.
TEACHER_LOGIT_CACHE optionally stores and replays per-dataset native BF16 teacher logits.
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
(( $# == 6 || $# == 7 )) || bad_arguments
dataset=$1
student=$2
images=$3
author=$4
out=$5
seed=$6
method=${METHOD:-all}
case "$dataset" in
  flowers|cub200|fgvc_aircraft|oxfordpets|stanfordcars) ;;
  caltech101) (( $# == 7 )) || bad_arguments ;;
  *) bad_arguments ;;
esac
case "$student" in
  resnet18|mobilenetv2|shufflenetv2) ;;
  *) bad_arguments ;;
esac
case "$seed" in
  17|29|43) ;;
  *) bad_arguments ;;
esac
case "$method" in
  all|lavid-ls-qwen|lavid-ls-ecoc|ecoc-only) ;;
  *) bad_arguments ;;
esac

python=${PYTHON:-python}
device=${DEVICE:-cuda}
signature=${7:-"$author/questions/$dataset/logits/qwen2.5-7b_chat_50-5.json"}
code="$out/$dataset-code.json"
teacher="$out/visual-teachers/$dataset/seed-17/author-best.pt"
common=(--dataset "$dataset" --images "$images" --author-source "$author" --device "$device")
cohort=()
if [[ $dataset != flowers && $dataset != cub200 ]]; then
  "$python" experiments/visual-distillation/prepare_benchmark_cohort.py --dataset "$dataset" --images "$images" \
    --author "$author" --out "$out/cohorts/$dataset"
  cohort=(--cohort "$out/cohorts/$dataset/cohort.csv")
fi

if [[ $method != lavid-ls-qwen ]]; then
  "$python" -m ecg_lavid.visual_benchmark_data --dataset "$dataset" --author-source "$author" "${cohort[@]}" \
    --signature "$signature" --out "$code"
fi
if [[ $method != ecoc-only ]]; then
  "$python" -m ecg_lavid.train_visual_lavid_teacher "${common[@]}" "${cohort[@]}" \
    --out "$out/visual-teachers" --seed 17 --workers "${WORKERS:-8}"
fi

student_args=("${common[@]}" "${cohort[@]}" --architecture "$student" --out "$out/visual-grid" \
  --seed "$seed" --workers "${WORKERS:-4}")
cache=()
record=()
if [[ -n ${TEACHER_LOGIT_CACHE:-} ]]; then
  cache=(--teacher-logit-cache "$TEACHER_LOGIT_CACHE/$dataset")
  record=(--record-teacher-logits)
fi
if [[ $method == all || $method == lavid-ls-qwen ]]; then
  "$python" -m ecg_lavid.train_visual_benchmark "${student_args[@]}" --method lavid-ls-qwen \
    --target "$signature" --teacher-checkpoint "$teacher" "${cache[@]}" "${record[@]}"
fi
if [[ $method == all || $method == lavid-ls-ecoc ]]; then
  "$python" -m ecg_lavid.train_visual_benchmark "${student_args[@]}" --method lavid-ls-ecoc \
    --target "$code" --teacher-checkpoint "$teacher" "${cache[@]}"
fi
if [[ $method == all || $method == ecoc-only ]]; then
  "$python" -m ecg_lavid.train_visual_benchmark "${student_args[@]}" --method ecoc-only --target "$code"
fi
