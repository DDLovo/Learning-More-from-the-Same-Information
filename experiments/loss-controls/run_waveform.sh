#!/usr/bin/env bash
set -eo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash experiments/loss-controls/run_waveform.sh PREPARED_PTBXL_NPZ OUTPUT

Environment: PYTHON (default python), DEVICE (default cuda), MISSING (mask or interpolation, default mask).
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi
if (( $# != 2 )); then
  usage >&2
  exit 2
fi
prepared=$1
out=$2
python=${PYTHON:-python}
device=${DEVICE:-cuda}
missing=${MISSING:-mask}
if [[ $missing != mask && $missing != interpolation ]]; then
  usage >&2
  exit 2
fi

"$python" -m ecg_lavid.ecoc --out "$out/ecg-codebooks"
"$python" -m ecg_lavid.train_ptb_crossmodal --data "$prepared" \
  --codebook "$out/ecg-codebooks/five_way.json" --out "$out/ptb-losses" \
  --architectures cnn bilstm --fractions 1.0 --seeds 17 --device "$device" \
  --arms A raw-onehot prob-onehot ce-same-onehot squentropy ce-independent-onehot ecoc-soft-ce H ecoc-soft-ce-gradmatch
"$python" -m ecg_lavid.evaluate_ptb_loss_controls --data "$prepared" --baselines "$out/ptb-losses" \
  --runs "$out/ptb-losses" --out "$out/ptb-missing-$missing" --fold 10 --missing "$missing" --device "$device"
