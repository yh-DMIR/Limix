#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DEFAULT_BENCHMARKS = [
    "openml_cc18_csv=../limix/openml_cc18_csv",
    "tabarena_cls=dataset/tabarena/cls",
    "tabzilla_csv=../limix/tabzilla_csv",
    "talent_csv=../limix/talent_csv",
]
TARGET_CANDIDATES = ["target", "label", "class", "y", "TARGET", "Label", "Class", "Y"]


@dataclass
class DatasetTask:
    benchmark: str
    dataset_id: str
    dataset_name: str
    dataset_dir: str
    single_csv: Optional[str]
    train_csv: Optional[str]
    test_csv: Optional[str]


@dataclass
class ResultRow:
    benchmark: str
    dataset_id: str
    dataset_dir: str
    dataset_name: str
    n_train: int
    n_test: int
    n_features: int
    n_classes: Optional[int]
    accuracy: Optional[float]
    f1_macro: Optional[float]
    logloss: Optional[float]
    auc_ovo: Optional[float]
    predict_seconds: float
    train_shard_rows: int
    test_batch_rows: int
    num_train_shards: int
    status: str
    error: Optional[str]


def clear_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def sanitize_dataset_id(value: str) -> str:
    match = re.search(r"(OpenML-ID-\d+)", value)
    return match.group(1) if match else Path(value).stem


def infer_target_column(df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    return df.columns[-1]


def parse_benchmark_specs(root: Path, specs: List[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    for spec in specs:
        if "=" in spec:
            name, rel_path = spec.split("=", 1)
            benchmark_name = name.strip()
            benchmark_path = Path(rel_path.strip())
        else:
            benchmark_path = Path(spec.strip())
            benchmark_name = benchmark_path.name

        if not benchmark_path.is_absolute():
            benchmark_path = (root / benchmark_path).resolve()
        else:
            benchmark_path = benchmark_path.resolve()
        parsed.append((benchmark_name, benchmark_path))
    return parsed


def discover_benchmark_tasks(benchmark_name: str, benchmark_dir: Path) -> List[DatasetTask]:
    tasks: List[DatasetTask] = []
    seen_keys: set[tuple[str, str]] = set()

    for train_csv in sorted(benchmark_dir.rglob("*_train.csv")):
        dataset_name = train_csv.stem[:-6]
        test_csv = train_csv.with_name(f"{dataset_name}_test.csv")
        task = DatasetTask(
            benchmark=benchmark_name,
            dataset_id=sanitize_dataset_id(str(train_csv)),
            dataset_name=dataset_name,
            dataset_dir=train_csv.parent.as_posix(),
            single_csv=None,
            train_csv=str(train_csv),
            test_csv=str(test_csv) if test_csv.exists() else None,
        )
        task_key = (task.dataset_dir, task.dataset_name)
        if task_key not in seen_keys:
            seen_keys.add(task_key)
            tasks.append(task)

    for csv_path in sorted(benchmark_dir.rglob("*.csv")):
        if csv_path.name.endswith("_train.csv") or csv_path.name.endswith("_test.csv"):
            continue
        task = DatasetTask(
            benchmark=benchmark_name,
            dataset_id=sanitize_dataset_id(str(csv_path)),
            dataset_name=csv_path.stem,
            dataset_dir=csv_path.parent.as_posix(),
            single_csv=str(csv_path),
            train_csv=None,
            test_csv=None,
        )
        task_key = (task.dataset_dir, task.dataset_name)
        if task_key not in seen_keys:
            seen_keys.add(task_key)
            tasks.append(task)

    return tasks


def build_tasks(root: Path, benchmark_specs: List[str]) -> Tuple[List[DatasetTask], Dict[str, int]]:
    tasks: List[DatasetTask] = []
    discovered: Dict[str, int] = {}
    for benchmark_name, benchmark_dir in parse_benchmark_specs(root, benchmark_specs):
        benchmark_tasks = discover_benchmark_tasks(benchmark_name, benchmark_dir) if benchmark_dir.exists() else []
        tasks.extend(benchmark_tasks)
        discovered[benchmark_name] = len(benchmark_tasks)
    return tasks, discovered


def shard_items(items: List[DatasetTask], num_workers: int, worker_id: int) -> List[DatasetTask]:
    return items[worker_id::num_workers]


def normalize_labels(labels: pd.Series) -> pd.Series:
    return labels.astype("string").fillna("__MISSING_LABEL__")


def load_dataset(
    task: DatasetTask,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if task.single_csv is not None:
        df = pd.read_csv(task.single_csv)
        target_col = infer_target_column(df)
        df = df.dropna(subset=[target_col])
        X = df.drop(columns=[target_col])
        y = normalize_labels(df[target_col])
        if len(df) < 2:
            raise ValueError("Not enough valid rows after dropping missing targets.")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
            )
        return X_train.reset_index(drop=True), y_train.reset_index(drop=True), X_test.reset_index(drop=True), y_test.reset_index(drop=True)

    if task.train_csv is None:
        raise ValueError(f"Task {task.dataset_name} does not contain readable csv inputs.")

    train_df = pd.read_csv(task.train_csv)
    target_col = infer_target_column(train_df)
    train_df = train_df.dropna(subset=[target_col])

    if task.test_csv is not None:
        test_df = pd.read_csv(task.test_csv)
        test_target_col = infer_target_column(test_df)
        test_df = test_df.dropna(subset=[test_target_col])
    else:
        try:
            train_df, test_df = train_test_split(
                train_df,
                test_size=test_size,
                random_state=random_state,
                stratify=train_df[target_col],
            )
        except ValueError:
            train_df, test_df = train_test_split(
                train_df,
                test_size=test_size,
                random_state=random_state,
            )
        test_target_col = target_col

    X_train = train_df.drop(columns=[target_col]).reset_index(drop=True)
    y_train = normalize_labels(train_df[target_col]).reset_index(drop=True)
    X_test = test_df.drop(columns=[test_target_col]).reset_index(drop=True)
    y_test = normalize_labels(test_df[test_target_col]).reset_index(drop=True)
    return X_train, y_train, X_test, y_test


def build_stratified_row_shards(y_encoded: np.ndarray, shard_rows: int, random_state: int) -> List[np.ndarray]:
    if shard_rows <= 0 or len(y_encoded) <= shard_rows:
        return [np.arange(len(y_encoded), dtype=np.int64)]

    num_shards = max(1, math.ceil(len(y_encoded) / shard_rows))
    buckets: List[List[int]] = [[] for _ in range(num_shards)]
    rng = np.random.default_rng(random_state)

    for cls in np.unique(y_encoded):
        cls_indices = np.where(y_encoded == cls)[0]
        rng.shuffle(cls_indices)
        for pos, idx in enumerate(cls_indices):
            buckets[pos % num_shards].append(int(idx))

    shards = [np.asarray(bucket, dtype=np.int64) for bucket in buckets if bucket]
    while len(shards) > 1:
        bad_shard_idx = next(
            (idx for idx, shard in enumerate(shards) if len(np.unique(y_encoded[shard])) < 2),
            None,
        )
        if bad_shard_idx is None:
            break
        merge_target_idx = max(
            (idx for idx in range(len(shards)) if idx != bad_shard_idx),
            key=lambda idx: len(shards[idx]),
        )
        shards[merge_target_idx] = np.concatenate([shards[merge_target_idx], shards[bad_shard_idx]])
        del shards[bad_shard_idx]

    return shards


def is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    oom_markers = [
        "out of memory",
        "cuda out of memory",
        "hip out of memory",
        "cublas_status_alloc_failed",
        "miopenstatusallocfailed",
    ]
    return any(marker in message for marker in oom_markers)


def safe_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
    try:
        if y_prob.shape[1] <= 1:
            return None
        if len(np.unique(y_true)) > 2:
            return float(roc_auc_score(y_true, y_prob, multi_class="ovo", labels=np.arange(y_prob.shape[1])))
        return float(roc_auc_score(y_true, y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]))
    except Exception:
        return None


def predict_proba_with_shards(
    clf,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    global_classes: np.ndarray,
    random_state: int,
    train_shard_rows: int,
    test_batch_rows: int,
) -> Tuple[np.ndarray, int]:
    global_encoder = LabelEncoder()
    global_encoder.fit(global_classes)
    y_train_encoded = global_encoder.transform(y_train.to_numpy())
    n_classes = len(global_encoder.classes_)
    train_shards = build_stratified_row_shards(y_train_encoded, train_shard_rows, random_state)
    test_batch_rows = max(1, test_batch_rows)

    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    total_train_rows = max(1, len(y_train_encoded))
    outputs: List[np.ndarray] = []

    for test_start in range(0, len(X_test_np), test_batch_rows):
        test_end = min(test_start + test_batch_rows, len(X_test_np))
        X_test_batch = X_test_np[test_start:test_end]
        batch_scores = np.zeros((len(X_test_batch), n_classes), dtype=np.float32)
        weight_sum = 0.0

        for shard_indices in train_shards:
            X_train_shard = X_train_np[shard_indices]
            y_train_shard = y_train_encoded[shard_indices]
            shard_weight = float(len(shard_indices) / total_train_rows)
            shard_output = clf.predict(X_train_shard, y_train_shard, X_test_batch, task_type="Classification")
            shard_class_indices = np.asarray(clf.classes, dtype=np.int64)
            batch_scores[:, shard_class_indices] += shard_output * shard_weight
            weight_sum += shard_weight
            clear_torch_cache()

        if weight_sum > 0:
            batch_scores /= weight_sum

        row_sums = batch_scores.sum(axis=1, keepdims=True)
        zero_rows = np.isclose(row_sums.squeeze(-1), 0.0)
        if np.any(zero_rows):
            batch_scores[zero_rows] = 1.0 / n_classes
            row_sums = batch_scores.sum(axis=1, keepdims=True)
        outputs.append(batch_scores / row_sums)

    return np.vstack(outputs), len(train_shards)


def evaluate_one_dataset(
    clf,
    task: DatasetTask,
    test_size: float,
    random_state: int,
    train_shard_rows: int,
    test_batch_rows: int,
    min_train_shard_rows: int,
    min_test_batch_rows: int,
    verbose: bool,
) -> ResultRow:
    X_train, y_train, X_test, y_test = load_dataset(task, test_size=test_size, random_state=random_state)
    all_labels = pd.concat([y_train, y_test], ignore_index=True)
    metric_encoder = LabelEncoder()
    metric_encoder.fit(all_labels.to_numpy())
    y_test_encoded = metric_encoder.transform(y_test.to_numpy())
    n_classes = len(metric_encoder.classes_)

    if len(X_train) < 2:
        raise ValueError("Training split contains fewer than 2 rows.")
    if n_classes < 2:
        raise ValueError("Dataset contains fewer than 2 classes.")

    cur_train_shard_rows = max(1, train_shard_rows)
    cur_test_batch_rows = max(1, test_batch_rows)

    while True:
        try:
            t0 = time.time()
            y_prob, num_train_shards = predict_proba_with_shards(
                clf=clf,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                global_classes=metric_encoder.classes_,
                random_state=random_state,
                train_shard_rows=cur_train_shard_rows,
                test_batch_rows=cur_test_batch_rows,
            )
            predict_seconds = time.time() - t0
            break
        except Exception as exc:
            clear_torch_cache()
            if not is_oom_error(exc):
                raise

            can_shrink_test = cur_test_batch_rows > max(1, min_test_batch_rows)
            can_shrink_train = cur_train_shard_rows > max(1, min_train_shard_rows)
            if not can_shrink_test and not can_shrink_train:
                raise

            if can_shrink_test and (cur_test_batch_rows >= cur_train_shard_rows or not can_shrink_train):
                cur_test_batch_rows = max(min_test_batch_rows, cur_test_batch_rows // 2)
            elif can_shrink_train:
                cur_train_shard_rows = max(min_train_shard_rows, cur_train_shard_rows // 2)

            if verbose:
                print(
                    f"[oom-retry] {task.benchmark}/{task.dataset_name} "
                    f"train_shard_rows={cur_train_shard_rows} test_batch_rows={cur_test_batch_rows}"
                )

    y_pred = np.argmax(y_prob, axis=1)
    ll = float(log_loss(y_test_encoded, y_prob, labels=np.arange(y_prob.shape[1])))
    auc_ovo = safe_auc_score(y_test_encoded, y_prob)

    return ResultRow(
        benchmark=task.benchmark,
        dataset_id=task.dataset_id,
        dataset_dir=task.dataset_dir,
        dataset_name=task.dataset_name,
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        n_features=int(X_train.shape[1]),
        n_classes=int(n_classes),
        accuracy=float(accuracy_score(y_test_encoded, y_pred)),
        f1_macro=float(f1_score(y_test_encoded, y_pred, average="macro")),
        logloss=ll,
        auc_ovo=auc_ovo,
        predict_seconds=float(predict_seconds),
        train_shard_rows=int(cur_train_shard_rows),
        test_batch_rows=int(cur_test_batch_rows),
        num_train_shards=int(num_train_shards),
        status="ok",
        error=None,
    )


def default_config_path(root: Path, model_path: str) -> Path:
    lower_name = Path(model_path).name.lower()
    if "2m" in lower_name:
        return (root / "config" / "cls_default_2M_retrieval.json").resolve()
    return (root / "config" / "cls_default_16M_retrieval.json").resolve()


def infer_model_repo_id(model_filename: str, model_repo_id: Optional[str]) -> str:
    if model_repo_id:
        return model_repo_id
    lower_name = model_filename.lower()
    if "2m" in lower_name:
        return "stableai-org/LimiX-2M"
    return "stableai-org/LimiX-16M"


def resolve_model_path(
    root: Path,
    model_path: str,
    model_repo_id: Optional[str],
    model_filename: Optional[str],
) -> Path:
    requested_path = Path(model_path)
    if not requested_path.is_absolute():
        requested_path = (root / requested_path).resolve()
    else:
        requested_path = requested_path.resolve()

    if requested_path.exists():
        return requested_path

    from utils.utils import download_model

    target_filename = model_filename or requested_path.name
    repo_id = infer_model_repo_id(target_filename, model_repo_id)
    requested_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = download_model(repo_id=repo_id, filename=target_filename, save_path=str(requested_path.parent))
    return Path(downloaded_path).resolve()


def run_worker(
    worker_id: int,
    gpu_id: int,
    task_items: List[DatasetTask],
    ready_queue,
    start_event,
    worker_out_csv: str,
    model_path: str,
    config_path: str,
    test_size: float,
    random_state: int,
    train_shard_rows: int,
    test_batch_rows: int,
    min_train_shard_rows: int,
    min_test_batch_rows: int,
    verbose: bool,
) -> None:
    try:
        gpu_id_str = str(gpu_id)
        os.environ["ROCR_VISIBLE_DEVICES"] = gpu_id_str
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("GPU_DEVICE_ORDINAL", None)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        import torch
        from inference.predictor import LimiXPredictor

        if not torch.cuda.is_available():
            raise RuntimeError("GPU backend is not available in this worker.")

        clf = LimiXPredictor(
            device=torch.device("cuda:0"),
            model_path=model_path,
            inference_config=config_path,
            inference_with_DDP=False,
        )

        ready_queue.put(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "status": "ready",
                "assigned_count": len(task_items),
            }
        )
        start_event.wait()

        rows: List[ResultRow] = []
        for task in task_items:
            try:
                row = evaluate_one_dataset(
                    clf=clf,
                    task=task,
                    test_size=test_size,
                    random_state=random_state,
                    train_shard_rows=train_shard_rows,
                    test_batch_rows=test_batch_rows,
                    min_train_shard_rows=min_train_shard_rows,
                    min_test_batch_rows=min_test_batch_rows,
                    verbose=verbose,
                )
            except Exception as exc:
                row = ResultRow(
                    benchmark=task.benchmark,
                    dataset_id=task.dataset_id,
                    dataset_dir=task.dataset_dir,
                    dataset_name=task.dataset_name,
                    n_train=0,
                    n_test=0,
                    n_features=0,
                    n_classes=None,
                    accuracy=None,
                    f1_macro=None,
                    logloss=None,
                    auc_ovo=None,
                    predict_seconds=0.0,
                    train_shard_rows=0,
                    test_batch_rows=0,
                    num_train_shards=0,
                    status="fail",
                    error=f"{type(exc).__name__}: {exc}",
                )

            rows.append(row)

            if verbose:
                if row.status == "ok":
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] [ok] "
                        f"{task.benchmark}/{task.dataset_name} acc={row.accuracy:.6f} "
                        f"f1={row.f1_macro:.6f} shards={row.num_train_shards}"
                    )
                else:
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] [fail] "
                        f"{task.benchmark}/{task.dataset_name} error={row.error}"
                    )
            clear_torch_cache()

        worker_df = pd.DataFrame([asdict(row) for row in rows]) if rows else pd.DataFrame(columns=list(ResultRow.__annotations__.keys()))
        worker_df.to_csv(worker_out_csv, index=False)
    except Exception:
        try:
            ready_queue.put(
                {
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "status": "crash",
                    "error": traceback.format_exc(),
                }
            )
        except Exception:
            pass
        pd.DataFrame(
            [
                {
                    "benchmark": "__worker__",
                    "dataset_id": f"__WORKER_CRASH__{worker_id}",
                    "dataset_dir": "__worker__",
                    "dataset_name": f"__WORKER_CRASH__{worker_id}",
                    "n_train": 0,
                    "n_test": 0,
                    "n_features": 0,
                    "n_classes": None,
                    "accuracy": None,
                    "f1_macro": None,
                    "logloss": None,
                    "auc_ovo": None,
                    "predict_seconds": 0.0,
                    "train_shard_rows": 0,
                    "test_batch_rows": 0,
                    "num_train_shards": 0,
                    "status": "fail",
                    "error": traceback.format_exc(),
                }
            ]
        ).to_csv(worker_out_csv, index=False)


def collect_worker_outputs(out_dir: Path, workers: int) -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    for worker_id in range(workers):
        worker_csv = out_dir / f"worker_{worker_id}.csv"
        if worker_csv.exists():
            try:
                dfs.append(pd.read_csv(worker_csv))
            except pd.errors.EmptyDataError:
                continue
    return dfs


def write_summary(summary_path: Path, result_df: pd.DataFrame, discovered_datasets: int, wall_seconds: float) -> None:
    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()

    lines = [
        f"discovered_datasets: {discovered_datasets}",
        f"processed_datasets: {len(result_df)}",
        f"ok_count: {len(ok_df)}",
        f"fail_count: {len(failed_df)}",
        f"wall_seconds: {wall_seconds:.3f}",
    ]

    if len(ok_df):
        lines.extend(
            [
                f"mean_accuracy: {ok_df['accuracy'].mean():.6f}",
                f"mean_f1_macro: {ok_df['f1_macro'].mean():.6f}",
                f"mean_logloss: {ok_df['logloss'].mean():.6f}",
                f"mean_auc_ovo: {ok_df['auc_ovo'].dropna().mean():.6f}" if ok_df["auc_ovo"].notna().any() else "mean_auc_ovo: NaN",
            ]
        )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_gpu_list(gpus: str) -> List[int]:
    return [int(x.strip()) for x in gpus.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official LimiX classification benchmarks on AMD GPUs.")
    parser.add_argument("--root", type=str, default=".", help="Project root used to resolve relative benchmark/config paths.")
    parser.add_argument("--benchmarks", type=str, default=",".join(DEFAULT_BENCHMARKS), help="Comma-separated benchmark specs.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the official LimiX checkpoint.")
    parser.add_argument("--model-repo-id", type=str, default=None, help="Optional Hugging Face repo id used when --model-path is missing locally.")
    parser.add_argument("--model-filename", type=str, default=None, help="Optional checkpoint filename used during auto-download.")
    parser.add_argument("--config-path", type=str, default=None, help="Optional classification retrieval config path.")
    parser.add_argument("--out-dir", type=str, default="result/LimiX_official_classification_amd", help="Output directory.")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes.")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="Comma-separated AMD GPU ids.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio for single-csv datasets.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splits and sharding.")
    parser.add_argument("--train-shard-rows", type=int, default=12000, help="Initial train rows per inference shard.")
    parser.add_argument("--test-batch-rows", type=int, default=128, help="Initial test rows per inference batch.")
    parser.add_argument("--min-train-shard-rows", type=int, default=2000, help="Lower bound when shrinking train shards after OOM.")
    parser.add_argument("--min-test-batch-rows", type=int, default=8, help="Lower bound when shrinking test batches after OOM.")
    parser.add_argument("--limit-datasets", type=int, default=0, help="Debug-only cap on number of datasets to process.")
    parser.add_argument("--verbose", action="store_true", help="Print per-dataset worker logs.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    benchmark_specs = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    resolved_model_path = resolve_model_path(
        root=root,
        model_path=args.model_path,
        model_repo_id=args.model_repo_id,
        model_filename=args.model_filename,
    )
    config_path = Path(args.config_path).resolve() if args.config_path else default_config_path(root, str(resolved_model_path))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks, discovered = build_tasks(root, benchmark_specs)
    if args.limit_datasets > 0:
        tasks = tasks[:args.limit_datasets]

    gpus = parse_gpu_list(args.gpus)
    workers = min(args.workers, len(gpus)) if gpus else 0
    if workers <= 0:
        raise ValueError("No GPU workers available. Check --workers and --gpus.")

    ready_queue = mp.Queue()
    start_event = mp.Event()
    processes: List[mp.Process] = []

    worker_assignments = [shard_items(tasks, workers, worker_id) for worker_id in range(workers)]
    start_time = time.time()

    for worker_id in range(workers):
        process = mp.Process(
            target=run_worker,
            kwargs={
                "worker_id": worker_id,
                "gpu_id": gpus[worker_id],
                "task_items": worker_assignments[worker_id],
                "ready_queue": ready_queue,
                "start_event": start_event,
                "worker_out_csv": str(out_dir / f"worker_{worker_id}.csv"),
                "model_path": str(resolved_model_path),
                "config_path": str(config_path),
                "test_size": args.test_size,
                "random_state": args.random_state,
                "train_shard_rows": args.train_shard_rows,
                "test_batch_rows": args.test_batch_rows,
                "min_train_shard_rows": args.min_train_shard_rows,
                "min_test_batch_rows": args.min_test_batch_rows,
                "verbose": args.verbose,
            },
        )
        process.start()
        processes.append(process)

    ready_status: List[dict] = []
    for _ in range(workers):
        ready_status.append(ready_queue.get())

    crashed = [item for item in ready_status if item.get("status") == "crash"]
    if crashed:
        for process in processes:
            process.join(timeout=1)
        raise RuntimeError(f"Worker initialization failed: {crashed[0].get('error', 'unknown error')}")

    start_event.set()

    for process in processes:
        process.join()

    worker_dfs = collect_worker_outputs(out_dir, workers)
    result_df = pd.concat(worker_dfs, ignore_index=True) if worker_dfs else pd.DataFrame(columns=list(ResultRow.__annotations__.keys()))
    result_df.to_csv(out_dir / "all_results.csv", index=False)

    wall_seconds = time.time() - start_time
    write_summary(out_dir / "summary.txt", result_df, discovered_datasets=len(tasks), wall_seconds=wall_seconds)

    meta = {
        "root": str(root),
        "benchmarks": benchmark_specs,
        "model_path": str(resolved_model_path),
        "config_path": str(config_path),
        "model_repo_id": args.model_repo_id,
        "model_filename": args.model_filename,
        "workers": workers,
        "gpus": gpus[:workers],
        "test_size": args.test_size,
        "random_state": args.random_state,
        "train_shard_rows": args.train_shard_rows,
        "test_batch_rows": args.test_batch_rows,
        "min_train_shard_rows": args.min_train_shard_rows,
        "min_test_batch_rows": args.min_test_batch_rows,
        "discovered_per_benchmark": discovered,
        "wall_seconds": wall_seconds,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Finished. results={out_dir / 'all_results.csv'} summary={out_dir / 'summary.txt'}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
