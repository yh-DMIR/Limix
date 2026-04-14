#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
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
from sklearn.metrics import accuracy_score, f1_score, log_loss
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
    predict_seconds: float
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


def normalize_skip_name(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""

    basename = Path(text.replace("\\", "/")).name
    if basename.lower().endswith(".csv"):
        basename = basename[:-4]

    lower_name = basename.lower()
    if lower_name.endswith("_train"):
        basename = basename[:-6]
    elif lower_name.endswith("_test"):
        basename = basename[:-5]

    return basename.strip().lower()


def parse_skip_name_tokens(raw_value: str) -> List[str]:
    return [token.strip() for token in re.split(r"[\r\n,]+", raw_value) if token.strip()]


def load_skip_dataset_names(root: Path, skip_dataset_names: str, skip_dataset_names_file: Optional[str]) -> set[str]:
    names: set[str] = set()

    for token in parse_skip_name_tokens(skip_dataset_names):
        normalized = normalize_skip_name(token)
        if normalized:
            names.add(normalized)

    if skip_dataset_names_file:
        file_path = Path(skip_dataset_names_file)
        if not file_path.is_absolute():
            file_path = (root / file_path).resolve()
        else:
            file_path = file_path.resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Skip dataset names file does not exist: {file_path}")
        for token in parse_skip_name_tokens(file_path.read_text(encoding="utf-8")):
            normalized = normalize_skip_name(token)
            if normalized:
                names.add(normalized)

    return names


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


def task_matches_skip(task: DatasetTask, skip_dataset_names: set[str]) -> bool:
    if not skip_dataset_names:
        return False

    candidates = {
        normalize_skip_name(task.dataset_name),
        normalize_skip_name(task.dataset_id),
        normalize_skip_name(task.single_csv or ""),
        normalize_skip_name(task.train_csv or ""),
        normalize_skip_name(task.test_csv or ""),
    }
    return any(candidate and candidate in skip_dataset_names for candidate in candidates)


def build_tasks(
    root: Path,
    benchmark_specs: List[str],
    skip_dataset_names: set[str],
) -> Tuple[List[DatasetTask], Dict[str, int], List[DatasetTask]]:
    tasks: List[DatasetTask] = []
    discovered: Dict[str, int] = {}
    skipped: List[DatasetTask] = []
    for benchmark_name, benchmark_dir in parse_benchmark_specs(root, benchmark_specs):
        benchmark_tasks = discover_benchmark_tasks(benchmark_name, benchmark_dir) if benchmark_dir.exists() else []
        discovered[benchmark_name] = len(benchmark_tasks)
        for task in benchmark_tasks:
            if task_matches_skip(task, skip_dataset_names):
                skipped.append(task)
                continue
            tasks.append(task)
    return tasks, discovered, skipped


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


def predict_proba_full_context(
    clf,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    metric_encoder: LabelEncoder,
) -> np.ndarray:
    n_classes = len(metric_encoder.classes_)

    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_train_encoded = metric_encoder.transform(y_train.to_numpy()).astype(np.int64, copy=False)

    shard_output = clf.predict(X_train_np, y_train_encoded, X_test_np, task_type="Classification")
    shard_output = np.asarray(shard_output, dtype=np.float32)

    if shard_output.ndim != 2 or shard_output.shape[0] != len(X_test_np):
        raise ValueError(
            "Unexpected prediction output shape: "
            f"{shard_output.shape}, expected ({len(X_test_np)}, n_classes)"
        )

    model_classes = np.asarray(clf.classes, dtype=np.int64)
    if model_classes.ndim != 1:
        raise ValueError(f"Unexpected classes shape: {model_classes.shape}")
    if np.any(model_classes < 0) or np.any(model_classes >= n_classes):
        raise ValueError(f"Predicted classes out of range: {model_classes}")

    batch_scores = np.zeros((len(X_test_np), n_classes), dtype=np.float32)
    batch_scores[:, model_classes] = shard_output

    row_sums = batch_scores.sum(axis=1, keepdims=True)
    zero_rows = np.isclose(row_sums.squeeze(-1), 0.0)
    if np.any(zero_rows):
        batch_scores[zero_rows] = 1.0 / n_classes
        row_sums = batch_scores.sum(axis=1, keepdims=True)
    return batch_scores / row_sums


def evaluate_one_dataset(
    clf,
    task: DatasetTask,
    test_size: float,
    random_state: int,
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

    t0 = time.time()
    y_prob = predict_proba_full_context(
        clf=clf,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        metric_encoder=metric_encoder,
    )
    predict_seconds = time.time() - t0

    y_pred = np.argmax(y_prob, axis=1)
    ll = float(log_loss(y_test_encoded, y_prob, labels=np.arange(y_prob.shape[1])))
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
        predict_seconds=float(predict_seconds),
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


def init_worker_output(worker_out_csv: str) -> None:
    columns = list(ResultRow.__annotations__.keys())
    pd.DataFrame(columns=columns).to_csv(worker_out_csv, index=False)


def append_worker_row(worker_out_csv: str, row: ResultRow) -> None:
    pd.DataFrame([asdict(row)]).to_csv(worker_out_csv, mode="a", header=False, index=False)


def run_worker(
    worker_id: int,
    gpu_id: int,
    task_queue,
    ready_queue,
    start_event,
    worker_out_csv: str,
    model_path: str,
    config_path: str,
    test_size: float,
    random_state: int,
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
            }
        )
        start_event.wait()

        init_worker_output(worker_out_csv)
        while True:
            task = task_queue.get()
            if task is None:
                break

            try:
                row = evaluate_one_dataset(
                    clf=clf,
                    task=task,
                    test_size=test_size,
                    random_state=random_state,
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
                    predict_seconds=0.0,
                    status="fail",
                    error=f"{type(exc).__name__}: {exc}",
                )

            append_worker_row(worker_out_csv, row)

            if verbose:
                if row.status == "ok":
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] [ok] "
                        f"{task.benchmark}/{task.dataset_name} acc={row.accuracy:.6f} "
                        f"f1={row.f1_macro:.6f}"
                    )
                else:
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] [fail] "
                        f"{task.benchmark}/{task.dataset_name} error={row.error}"
                    )
            clear_torch_cache()
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
                    "predict_seconds": 0.0,
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


def task_identity(task: DatasetTask) -> tuple[str, str, str, str]:
    return (task.benchmark, task.dataset_id, task.dataset_dir, task.dataset_name)


def result_identity(row: pd.Series) -> tuple[str, str, str, str]:
    return (
        str(row["benchmark"]),
        str(row["dataset_id"]),
        str(row["dataset_dir"]),
        str(row["dataset_name"]),
    )


def build_missing_result_rows(
    tasks: List[DatasetTask],
    result_df: pd.DataFrame,
    worker_exitcodes: List[int],
) -> pd.DataFrame:
    if result_df.empty:
        existing_keys: set[tuple[str, str, str, str]] = set()
    else:
        existing_keys = {result_identity(row) for _, row in result_df.iterrows()}

    missing_rows = []
    exitcode_text = ",".join(str(code) for code in worker_exitcodes)
    error_message = (
        "Missing worker result: task was discovered but no worker wrote a result row. "
        f"worker_exitcodes={exitcode_text}"
    )
    for task in tasks:
        if task_identity(task) in existing_keys:
            continue
        missing_rows.append(
            {
                "benchmark": task.benchmark,
                "dataset_id": task.dataset_id,
                "dataset_dir": task.dataset_dir,
                "dataset_name": task.dataset_name,
                "n_train": 0,
                "n_test": 0,
                "n_features": 0,
                "n_classes": None,
                "accuracy": None,
                "f1_macro": None,
                "logloss": None,
                "predict_seconds": 0.0,
                "status": "fail",
                "error": error_message,
            }
        )

    return pd.DataFrame(missing_rows, columns=list(ResultRow.__annotations__.keys()))


def build_task_table(
    runnable_tasks: List[DatasetTask],
    skipped_tasks: List[DatasetTask],
    result_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows: List[dict] = []

    for task in runnable_tasks:
        rows.append(
            {
                "benchmark": task.benchmark,
                "dataset_id": task.dataset_id,
                "dataset_dir": task.dataset_dir,
                "dataset_name": task.dataset_name,
                "single_csv": task.single_csv,
                "train_csv": task.train_csv,
                "test_csv": task.test_csv,
                "task_status": "pending",
            }
        )

    for task in skipped_tasks:
        rows.append(
            {
                "benchmark": task.benchmark,
                "dataset_id": task.dataset_id,
                "dataset_dir": task.dataset_dir,
                "dataset_name": task.dataset_name,
                "single_csv": task.single_csv,
                "train_csv": task.train_csv,
                "test_csv": task.test_csv,
                "task_status": "skipped",
            }
        )

    task_df = pd.DataFrame(rows)
    if result_df is None or task_df.empty or result_df.empty:
        return task_df

    merged = task_df.merge(
        result_df,
        how="left",
        on=["benchmark", "dataset_id", "dataset_dir", "dataset_name"],
        suffixes=("", "_result"),
    )
    has_result = merged["status"].notna()
    merged.loc[has_result, "task_status"] = merged.loc[has_result, "status"]
    return merged


def write_summary(
    summary_path: Path,
    result_df: pd.DataFrame,
    discovered_datasets: int,
    skipped_datasets: int,
    wall_seconds: float,
) -> None:
    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()

    lines = [
        f"discovered_datasets: {discovered_datasets}",
        f"skipped_datasets: {skipped_datasets}",
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
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for train/test splits.")
    parser.add_argument("--limit-datasets", type=int, default=0, help="Debug-only cap on number of datasets to process.")
    parser.add_argument(
        "--skip-dataset-names",
        type=str,
        default="",
        help="Comma-separated dataset names to skip. Matches dataset name / csv basename, ignoring .csv and _train/_test suffixes.",
    )
    parser.add_argument(
        "--skip-dataset-names-file",
        type=str,
        default=None,
        help="Optional text file listing dataset names to skip, separated by commas or newlines.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-dataset worker logs.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    benchmark_specs = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    skip_dataset_names = load_skip_dataset_names(root, args.skip_dataset_names, args.skip_dataset_names_file)
    resolved_model_path = resolve_model_path(
        root=root,
        model_path=args.model_path,
        model_repo_id=args.model_repo_id,
        model_filename=args.model_filename,
    )
    config_path = Path(args.config_path).resolve() if args.config_path else default_config_path(root, str(resolved_model_path))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks, discovered, skipped_datasets = build_tasks(root, benchmark_specs, skip_dataset_names)
    discovered_total = sum(discovered.values())
    if args.limit_datasets > 0:
        tasks = tasks[:args.limit_datasets]

    task_table_path = out_dir / "task_table.csv"
    build_task_table(tasks, skipped_datasets).to_csv(task_table_path, index=False)

    gpus = parse_gpu_list(args.gpus)
    workers = min(args.workers, len(gpus)) if gpus else 0
    if workers <= 0:
        raise ValueError("No GPU workers available. Check --workers and --gpus.")

    task_queue = mp.Queue()
    ready_queue = mp.Queue()
    start_event = mp.Event()
    processes: List[mp.Process] = []

    for task in tasks:
        task_queue.put(task)
    for _ in range(workers):
        task_queue.put(None)

    start_time = time.time()

    for worker_id in range(workers):
        process = mp.Process(
            target=run_worker,
            kwargs={
                "worker_id": worker_id,
                "gpu_id": gpus[worker_id],
                "task_queue": task_queue,
                "ready_queue": ready_queue,
                "start_event": start_event,
                "worker_out_csv": str(out_dir / f"worker_{worker_id}.csv"),
                "model_path": str(resolved_model_path),
                "config_path": str(config_path),
                "test_size": args.test_size,
                "random_state": args.random_state,
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

    worker_exitcodes = [int(process.exitcode) if process.exitcode is not None else -999 for process in processes]

    worker_dfs = collect_worker_outputs(out_dir, workers)
    result_df = pd.concat(worker_dfs, ignore_index=True) if worker_dfs else pd.DataFrame(columns=list(ResultRow.__annotations__.keys()))
    missing_result_df = build_missing_result_rows(tasks, result_df, worker_exitcodes)
    if not missing_result_df.empty:
        result_df = pd.concat([result_df, missing_result_df], ignore_index=True)
    result_df.to_csv(out_dir / "all_results.csv", index=False)
    build_task_table(tasks, skipped_datasets, result_df=result_df).to_csv(task_table_path, index=False)

    wall_seconds = time.time() - start_time
    write_summary(
        out_dir / "summary.txt",
        result_df,
        discovered_datasets=discovered_total,
        skipped_datasets=len(skipped_datasets),
        wall_seconds=wall_seconds,
    )

    meta = {
        "root": str(root),
        "benchmarks": benchmark_specs,
        "model_path": str(resolved_model_path),
        "config_path": str(config_path),
        "model_repo_id": args.model_repo_id,
        "model_filename": args.model_filename,
        "workers": workers,
        "gpus": gpus[:workers],
        "worker_exitcodes": worker_exitcodes,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "skip_dataset_names_requested": sorted(skip_dataset_names),
        "skipped_datasets": [f"{task.benchmark}/{task.dataset_name}" for task in skipped_datasets],
        "discovered_per_benchmark": discovered,
        "discovered_total": discovered_total,
        "processed_total": len(tasks),
        "missing_result_count": int(len(missing_result_df)),
        "wall_seconds": wall_seconds,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Finished. results={out_dir / 'all_results.csv'} summary={out_dir / 'summary.txt'}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
