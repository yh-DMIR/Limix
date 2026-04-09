import os
import re
import openml
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# 你的 Dataset Name 列（OpenML-ID-xxxx）
DATASET_NAMES = [
    "OpenML-ID-23512", "OpenML-ID-4134", "OpenML-ID-470", "OpenML-ID-1493",
    "OpenML-ID-1459", "OpenML-ID-41027", "OpenML-ID-40981", "OpenML-ID-934",
    "OpenML-ID-1565", "OpenML-ID-41150", "OpenML-ID-41159", "OpenML-ID-846",
    "OpenML-ID-1169", "OpenML-ID-41147", "OpenML-ID-41143", "OpenML-ID-1567",
    "OpenML-ID-999", "OpenML-ID-10", "OpenML-ID-11", "OpenML-ID-14",
    "OpenML-ID-22", "OpenML-ID-29", "OpenML-ID-27", "OpenML-ID-31",
    "OpenML-ID-46", "OpenML-ID-54", "OpenML-ID-333", "OpenML-ID-1067",
    "OpenML-ID-1468", "OpenML-ID-1494", "OpenML-ID-43973", "OpenML-ID-1043",
    "OpenML-ID-43945", "OpenML-ID-1486", "OpenML-ID-42825", "OpenML-ID-4538"
]

# LimiX CSV 输出目录（喂给 inference_classifier.py 的 data_dir）
OUT_ROOT = "dataset/limix/tabzilla_csv"
os.makedirs(OUT_ROOT, exist_ok=True)

TEST_SIZE = 0.2
RANDOM_SEED = 42


def _encode_categoricals_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    将 object/category/bool 列做“训练集拟合 -> 测试集映射”的整数编码
    测试集未见类别 => -1
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    cat_cols = []
    for c in X_train.columns:
        dt = X_train[c].dtype
        if dt == "object" or str(dt).startswith("category") or dt == "bool":
            cat_cols.append(c)

    for c in cat_cols:
        tr = X_train[c].astype("string").fillna("__MISSING__")
        te = X_test[c].astype("string").fillna("__MISSING__")

        uniq = pd.Index(tr.unique())
        mapping = pd.Series(np.arange(len(uniq), dtype=np.int32), index=uniq)

        X_train[c] = tr.map(mapping).astype(np.int32)
        X_test[c] = te.map(mapping).fillna(-1).astype(np.int32)

    # 数值列尽量转 float（缺失变 NaN）
    for c in X_train.columns:
        if c not in cat_cols:
            X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    return X_train, X_test


def export_limix_csv(dataset_name: str, X: pd.DataFrame, y: pd.Series, out_root: str):
    out_dir = os.path.join(out_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    stratify = y
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=stratify
        )
    except Exception:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=None
        )

    X_tr_enc, X_te_enc = _encode_categoricals_train_test(X_tr, X_te)

    train_df = X_tr_enc.copy()
    test_df = X_te_enc.copy()
    train_df["y"] = y_tr.values
    test_df["y"] = y_te.values

    train_csv = os.path.join(out_dir, f"{dataset_name}_train.csv")
    test_csv = os.path.join(out_dir, f"{dataset_name}_test.csv")
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return len(train_df), len(test_df)


def main():
    for name in DATASET_NAMES:
        m = re.search(r"OpenML-ID-(\d+)", name)
        if not m:
            print(f"[skip] invalid name: {name}")
            continue
        did = int(m.group(1))

        try:
            print(f"\n📥 Fetching {name} (id={did}) ...")
            dataset = openml.datasets.get_dataset(
                did,
                download_data=True,
                download_all_files=False,   # 不用拉额外文件
            )

            target = getattr(dataset, "default_target_attribute", None)
            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target)
            if y is None:
                print(f"[skip] {name}: no target")
                continue

            y = pd.Series(y)
            if y.nunique(dropna=True) < 2:
                print(f"[skip] {name}: y has <2 classes")
                continue

            # 清理：删除全空列，防止 0 feature
            X = X.dropna(axis=1, how="all")
            if X.shape[1] == 0:
                print(f"[fail] {name}: 0 features after dropping all-NaN columns")
                continue

            tr_n, te_n = export_limix_csv(name, X, y, OUT_ROOT)
            print(f"[ok] {name}: train={tr_n} test={te_n} -> {os.path.join(OUT_ROOT, name)}")

        except Exception as e:
            print(f"[fail] {name}: {e}")

    print("\n🎉 Done.")


if __name__ == "__main__":
    main()
