#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash experiments/ecg-distillation/run_rr.sh fit PTBXL_NPZ UNIG_FEATURES_CSV OUTPUT
  bash experiments/ecg-distillation/run_rr.sh replay PTBXL_NPZ OUTPUT
  bash experiments/ecg-distillation/run_rr.sh ecgdeli PTBXL_PLUS_CACHE UNIG_FEATURES_CSV OUTPUT
  bash experiments/ecg-distillation/run_rr.sh ludb LUDB_CACHE OUTPUT
Environment: PYTHON (default python), DEVICE (default cuda), PRECISION (default auto).
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
(( $# >= 1 )) || bad_arguments
stage=$1
shift
python=${PYTHON:-python}
device=${DEVICE:-cuda}

case "$stage" in
  fit)
    (( $# == 3 )) || bad_arguments
    data=$1
    features=$2
    out=$3
    questions="$out/rr-questions.json"
    "$python" -m ecg_lavid.rr_questions --out "$questions"
    common=(--data "$data" --features "$features" --questions "$questions" --device "$device"
            --precision "${PRECISION:-auto}" --seeds 17 29 43 --epochs 25 --batch-size 256 --lavid-weight 0.4)
    "$python" -m ecg_lavid.train_rr "${common[@]}" --fraction 0.1 --out "$out/budget-10pct"
    "$python" -m ecg_lavid.train_rr "${common[@]}" --fraction 1 --out "$out/budget-full"
    ;;
  replay)
    (( $# == 2 )) || bad_arguments
    out=$2
    "$python" -m ecg_lavid.evaluate_rr_aux replay --data "$1" \
      --results "$out/budget-10pct" "$out/budget-full" --out "$out/raw-aux" --device "$device"
    ;;
  ecgdeli)
    (( $# == 3 )) || bad_arguments
    out=$3
    "$python" -m ecg_lavid.prepare_rr_fiducials --test-index "$out/budget-full/test_indices.csv" \
      --cache "$1" --features "$2" --download --out "$out/ecgdeli-reference.csv"
    "$python" -m ecg_lavid.evaluate_rr_aux score --raw "$out/raw-aux" \
      --ecgdeli "$out/ecgdeli-reference.csv" --out "$out/ordinal-rank.json"
    ;;
  ludb)
    (( $# == 2 )) || bad_arguments
    out=$2
    "$python" -m ecg_lavid.prepare_ludb --cache "$1" --out "$out/ludb-inputs"
    "$python" -m ecg_lavid.evaluate_ludb_aux --waveforms "$out/ludb-inputs/waveforms.npz" \
      --manual "$out/ludb-inputs/manual_reference.csv" --results "$out/budget-10pct" "$out/budget-full" \
      --out "$out/ludb-evaluation" --device "$device"
    ;;
  *) bad_arguments ;;
esac
