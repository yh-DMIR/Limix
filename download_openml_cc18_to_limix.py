import os
import numpy as np
import pandas as pd
import openml
from sklearn.preprocessing import OrdinalEncoder
from openml.tasks import TaskType

# ========== 配置区 ==========
OUT_ROOT = "dataset/limix/openml_cc18_csv"   # 输出给 LimiX 的 data_dir
SUITE_NAME = "OpenML-CC18"           # 基准套件名
SUITE_ID_FALLBACK = 99               # OpenML-CC18 通常是 suite id=99（若按名取不到就用它）
REPEAT = 0                           # 用哪个 repeat
FOLD = 0                             # 用哪个 fold
SAMPLE = 0                           # 通常为 0
# ============================

os.makedirs(OUT_ROOT, exist_ok=True)


def encode_categoricals_fit_on_train(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    对 object/category/bool 列做 OrdinalEncoder:
    - 仅在训练集拟合
    - 测试集未见类别 => -1
    数值列转为 float（无法转的变 NaN）
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    cat_cols = []
    for c in X_train.columns:
        dt = X_train[c].dtype
        if dt == "object" or str(dt).startswith("category") or dt == "bool":
            cat_cols.append(c)

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        tr = X_train[cat_cols].astype("string").fillna("__MISSING__")
        te = X_test[cat_cols].astype("string").fillna("__MISSING__")
        X_train[cat_cols] = enc.fit_transform(tr).astype(np.int32)
        X_test[cat_cols] = enc.transform(te).astype(np.int32)

    # 数值列尽量转 float
    for c in X_train.columns:
        if c not in cat_cols:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    return X_train, X_test


def save_limix_csv(ds_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series):
    out_dir = os.path.join(OUT_ROOT, ds_name)
    os.makedirs(out_dir, exist_ok=True)

    train_df = X_train.copy()
    test_df = X_test.copy()

    train_df["y"] = y_train.values
    test_df["y"] = y_test.values

    train_path = os.path.join(out_dir, f"{ds_name}_train.csv")
    test_path = os.path.join(out_dir, f"{ds_name}_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)


def get_suite():
    # 优先按 name 取；失败就用 fallback id
    try:
        suite = openml.study.get_suite(SUITE_NAME)
        return suite
    except Exception:
        return openml.study.get_suite(SUITE_ID_FALLBACK)


def main():
    suite = get_suite()
    task_ids = list(suite.tasks)

    print(f"Suite: {suite.name} | #tasks={len(task_ids)} | out={OUT_ROOT}")
    print(f"Using split: repeat={REPEAT}, fold={FOLD}, sample={SAMPLE}")

    ok = 0
    fail = 0

    for tid in task_ids:
        try:
            task = openml.tasks.get_task(tid)

            if task.task_type != TaskType.SUPERVISED_CLASSIFICATION:
                print(f"[skip] task {tid}: task_type={task.task_type} (not classification)")
                continue

            did = task.dataset_id
            ds_name = f"OpenML-ID-{did}"

            # 取数据（dataframe）
            dataset = openml.datasets.get_dataset(did, download_data=True)
            target_name = task.target_name  # 官方目标列名（比 default_target_attribute 更可靠）

            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target_name)
            if y is None:
                print(f"[skip] {ds_name}: no y")
                continue

            X = X.dropna(axis=1, how="all")
            if X.shape[1] == 0:
                print(f"[fail] {ds_name}: 0 features after dropping all-NaN columns")
                fail += 1
                continue

            # 用 task 提供的官方划分 indices（标准 benchmark）
            train_idx, test_idx = task.get_train_test_split_indices(
                repeat=REPEAT, fold=FOLD, sample=SAMPLE
            )

            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_test = X.iloc[test_idx].reset_index(drop=True)
            y_series = pd.Series(y).reset_index(drop=True)
            y_train = y_series.iloc[train_idx].reset_index(drop=True)
            y_test = y_series.iloc[test_idx].reset_index(drop=True)

            # 编码类别列
            X_train_enc, X_test_enc = encode_categoricals_fit_on_train(X_train, X_test)

            save_limix_csv(ds_name, X_train_enc, y_train, X_test_enc, y_test)

            print(f"[ok] {ds_name}: train={len(X_train_enc)} test={len(X_test_enc)} -> {os.path.join(OUT_ROOT, ds_name)}")
            ok += 1

        except Exception as e:
            print(f"[fail] task {tid}: {e}")
            fail += 1

    print(f"\nDone. ok={ok}, fail={fail}, out={OUT_ROOT}")


if __name__ == "__main__":
    main()
