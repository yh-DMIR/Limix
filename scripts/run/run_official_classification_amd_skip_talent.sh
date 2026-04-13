#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-benchmark_limix_classification_amd_skip_csv.py}
ROOT=${ROOT:-.}
BENCHMARKS=${BENCHMARKS:-talent_cls=dataset/talent_cls}
MODEL_PATH=${MODEL_PATH:-ckpt/LimiX-16M.ckpt}
CONFIG_PATH=${CONFIG_PATH:-}
OUT_DIR=${OUT_DIR:-result/LimiX_official_classification_amd_skip_talent}
WORKERS=${WORKERS:-1}
GPUS=${GPUS:-3}
SKIP_DATASET_NAMES=${SKIP_DATASET_NAMES:-}
SKIP_DATASET_NAMES_FILE=${SKIP_DATASET_NAMES_FILE:-}

CMD=(
  "${PYTHON}" "${SCRIPT}"
  --root "${ROOT}"
  --benchmarks "${BENCHMARKS}"
  --model-path "${MODEL_PATH}"
  --out-dir "${OUT_DIR}"
  --workers "${WORKERS}"
  --gpus "${GPUS}"
  --test-size 0.2
  --verbose
)

if [[ -n "${CONFIG_PATH}" ]]; then
  CMD+=(--config-path "${CONFIG_PATH}")
fi

if [[ -n "${SKIP_DATASET_NAMES}" ]]; then
  CMD+=(--skip-dataset-names "${SKIP_DATASET_NAMES}")
fi

if [[ -n "${SKIP_DATASET_NAMES_FILE}" ]]; then
  CMD+=(--skip-dataset-names-file "${SKIP_DATASET_NAMES_FILE}")
fi

"${CMD[@]}"
