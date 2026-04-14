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
BASE_BENCHMARKS=${BASE_BENCHMARKS:-openml_cc18_csv=dataset/openml_cc18_72,tabarena_cls=dataset/tabarena/cls,tabzilla_csv=dataset/tabzilla35,talent_cls=dataset/talent_cls}
MODEL_PATH=${MODEL_PATH:-ckpt/LimiX-16M.ckpt}
CONFIG_PATH=${CONFIG_PATH:-}
OUT_DIR=${OUT_DIR:-result/LimiX_official_classification_amd_skip_all_v3}
WORKERS=${WORKERS:-8}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
SKIP_DATASET_NAMES=${SKIP_DATASET_NAMES:-}
SKIP_DATASET_NAMES_FILE=${SKIP_DATASET_NAMES_FILE:-skip.txt}
ONLY_V2_FAILS=${ONLY_V2_FAILS:-1}
FAIL_SOURCE=${FAIL_SOURCE:-result/LimiX_official_classification_amd_skip_all_v2/all_results.csv}

TEMP_FAIL_ROOT=""
cleanup() {
  if [[ -n "${TEMP_FAIL_ROOT}" && -d "${TEMP_FAIL_ROOT}" ]]; then
    rm -rf "${TEMP_FAIL_ROOT}"
  fi
}
trap cleanup EXIT

BENCHMARKS=${BENCHMARKS:-}
if [[ -z "${BENCHMARKS}" ]]; then
  if [[ "${ONLY_V2_FAILS}" == "1" ]]; then
    TEMP_FAIL_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/limix_v3_fail_only.XXXXXX")
    BENCHMARKS=$(
      "${PYTHON}" - "${FAIL_SOURCE}" "${TEMP_FAIL_ROOT}" <<'PY'
import csv
import os
import pathlib
import sys

fail_source = pathlib.Path(sys.argv[1])
temp_root = pathlib.Path(sys.argv[2])

if not fail_source.exists():
    raise SystemExit(f"FAIL_SOURCE does not exist: {fail_source}")

rows = []
with fail_source.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        if row.get("status") == "fail":
            rows.append(row)

if not rows:
    raise SystemExit(f"No failed rows found in: {fail_source}")

benchmark_names = []
for row in rows:
    benchmark = row["benchmark"]
    dataset_dir = pathlib.Path(row["dataset_dir"])
    dataset_name = row["dataset_name"]
    out_dir = temp_root / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    if benchmark not in benchmark_names:
        benchmark_names.append(benchmark)

    candidates = [
        dataset_dir / f"{dataset_name}.csv",
        dataset_dir / f"{dataset_name}_train.csv",
        dataset_dir / f"{dataset_name}_test.csv",
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise SystemExit(f"Could not locate source csv for {benchmark}/{dataset_name} under {dataset_dir}")

    for src in existing:
        dst = out_dir / src.name
        try:
            os.symlink(src, dst)
        except FileExistsError:
            pass

print(",".join(f"{benchmark}={temp_root / benchmark}" for benchmark in benchmark_names), end="")
PY
    )
  else
    BENCHMARKS=${BASE_BENCHMARKS}
  fi
fi

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
