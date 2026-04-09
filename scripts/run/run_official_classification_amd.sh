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
SCRIPT=${SCRIPT:-benchmark_limix_classification_amd.py}
ROOT=${ROOT:-.}
BENCHMARKS=${BENCHMARKS:-openml_cc18_csv=../limix/openml_cc18_csv,tabarena_cls=dataset/tabarena/cls,tabzilla_csv=../limix/tabzilla_csv,talent_csv=../limix/talent_csv}
MODEL_PATH=${MODEL_PATH:-cache/LimiX-16M/LimiX-16M.ckpt}
MODEL_REPO_ID=${MODEL_REPO_ID:-}
MODEL_FILENAME=${MODEL_FILENAME:-}
CONFIG_PATH=${CONFIG_PATH:-}
OUT_DIR=${OUT_DIR:-result/LimiX_official_classification_amd}
WORKERS=${WORKERS:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
TRAIN_SHARD_ROWS=${TRAIN_SHARD_ROWS:-12000}
TEST_BATCH_ROWS=${TEST_BATCH_ROWS:-128}
MIN_TRAIN_SHARD_ROWS=${MIN_TRAIN_SHARD_ROWS:-2000}
MIN_TEST_BATCH_ROWS=${MIN_TEST_BATCH_ROWS:-8}

CMD=(
  "${PYTHON}" "${SCRIPT}"
  --root "${ROOT}"
  --benchmarks "${BENCHMARKS}"
  --model-path "${MODEL_PATH}"
  --out-dir "${OUT_DIR}"
  --workers "${WORKERS}"
  --gpus "${GPUS}"
  --train-shard-rows "${TRAIN_SHARD_ROWS}"
  --test-batch-rows "${TEST_BATCH_ROWS}"
  --min-train-shard-rows "${MIN_TRAIN_SHARD_ROWS}"
  --min-test-batch-rows "${MIN_TEST_BATCH_ROWS}"
  --test-size 0.2
  --verbose
)

if [[ -n "${MODEL_REPO_ID}" ]]; then
  CMD+=(--model-repo-id "${MODEL_REPO_ID}")
fi

if [[ -n "${MODEL_FILENAME}" ]]; then
  CMD+=(--model-filename "${MODEL_FILENAME}")
fi

if [[ -n "${CONFIG_PATH}" ]]; then
  CMD+=(--config-path "${CONFIG_PATH}")
fi

"${CMD[@]}"
