import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import openml

# =========================
# 配置（沿用你原脚本的风格）
# =========================
SUITE_ID = 99  # OpenML-CC18
OUT_ROOT = Path("dataset/limix/openml_cc18_csv")  # 只输出 CSV（给 LimiX 的 data_dir）

RANDOM_SEED = 42
TEST_SIZE = 0.20
STRATIFY = True
# =========================


def is_missing_cat(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype("string").str.len().fillna(0) == 0)


def encode_numerical(dfN: pd.DataFrame) -> pd.DataFrame:
    """数值：转 float；缺失保留 NaN"""
    if dfN.shape[1] == 0:
        return pd.DataFrame(index=dfN.index)
    out = dfN.apply(pd.to_numeric, errors="coerce")
    return out


def encode_categorical_full(dfC: pd.DataFrame) -> pd.DataFrame:
    """
    类别：用 pandas.Categorical 在“全量数据”上建码表，保证 train/test 一致
    缺失 -> -1
    """
    if dfC.shape[1] == 0:
        return pd.DataFrame(index=dfC.index)

    out = pd.DataFrame(index=dfC.index)
    for col in dfC.columns:
        s = dfC[col].astype("string")
        miss = is_missing_cat(dfC[col])
        s[miss] = pd.NA
        cat = pd.Categorical(s)
        out[col] = cat.codes.astype(np.int32)  # 缺失 -> -1
    return out


def convert_one_dataset(did: int):
    name = f"OpenML-ID-{did}"
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 取 OpenML 元信息：target + categorical_indicator（和你原脚本一致）:contentReference[oaicite:4]{index=4}
    ds = openml.datasets.get_dataset(did, download_data=True)

    target_name = ds.default_target_attribute
    if not target_name:
        # 少数数据可能没有 default_target_attribute，跳过
        print(f"[skip] {name}: no default_target_attribute")
        return False

    X_meta, y_meta, cat_ind, attr_names = ds.get_data(dataset_format="dataframe", target=target_name)
    feat_names = list(attr_names)
    cat_mask = np.array(cat_ind, dtype=bool)

    # 用 meta 的列名从 X_meta 取特征（最稳）
    X = X_meta.copy()
    y = pd.Series(y_meta)

    # 删除 y 缺失样本（沿用你原脚本思路）:contentReference[oaicite:5]{index=5}
    y_str = y.astype("string")
    y_miss = y_str.isna() | (y_str.str.len().fillna(0) == 0)
    if int(y_miss.sum()) > 0:
        X = X.loc[~y_miss].reset_index(drop=True)
        y_str = y_str.loc[~y_miss].reset_index(drop=True)

    # 删除全空特征列，避免 0 features
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        print(f"[skip] {name}: 0 features after drop all-NaN columns")
        return False

    # 根据 cat_mask 划分数值/类别列（按 meta 顺序对齐）
    # 注意：X 里可能少了某些列（被 drop 掉了），要同步过滤
    cols_in_x = list(X.columns)
    cat_cols = []
    num_cols = []
    for c in feat_names:
        if c not in cols_in_x:
            continue
        is_cat = bool(cat_mask[feat_names.index(c)])
        (cat_cols if is_cat else num_cols).append(c)

    dfN = X[num_cols] if num_cols else pd.DataFrame(index=X.index)
    dfC = X[cat_cols] if cat_cols else pd.DataFrame(index=X.index)

    # y 编码成 0..K-1（LimiX 分类器需要这种形式）
    y_codes, y_classes = pd.factorize(y_str, sort=True)
    y_codes = y_codes.astype(np.int64)

    # train/test split（和你原脚本一致）:contentReference[oaicite:6]{index=6}
    idx_all = np.arange(len(X))
    strat = y_codes if (STRATIFY and len(np.unique(y_codes)) > 1) else None
    idx_train, idx_test = train_test_split(
        idx_all, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=strat
    )

    # 编码特征（全量编码确保一致）
    N_all = encode_numerical(dfN)
    C_all = encode_categorical_full(dfC)

    # 组织成 LimiX CSV：特征列 + 最后一列 y
    X_all = pd.concat(
        [
            N_all.add_prefix("num_"),
            C_all.add_prefix("cat_"),
        ],
        axis=1
    )

    train_df = X_all.iloc[idx_train].copy()
    test_df = X_all.iloc[idx_test].copy()
    train_df["y"] = y_codes[idx_train]
    test_df["y"] = y_codes[idx_test]

    train_path = out_dir / f"{name}_train.csv"
    test_path = out_dir / f"{name}_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # 可选：写一个 info（不影响你“只要 CSV”）
    info = {
        "benchmark": "OpenML-CC18",
        "suite_id": SUITE_ID,
        "openml_id": did,
        "target_name": target_name,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_num_features": int(N_all.shape[1]),
        "n_cat_features": int(C_all.shape[1]),
        "y_classes": [str(x) for x in list(y_classes)],
        "split": {"random_seed": RANDOM_SEED, "test_size": TEST_SIZE, "stratify": bool(strat is not None)},
    }
    with open(out_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"[ok] {name}: train={len(train_df)} test={len(test_df)} -> {out_dir}")
    return True


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    suite = openml.study.get_suite(SUITE_ID)
    dataset_ids = list(suite.data)  # 关键：用 suite.data（照你原脚本）:contentReference[oaicite:7]{index=7}
    print(f"Suite: OpenML-CC18 | #datasets={len(dataset_ids)} | out={OUT_ROOT}")

    ok = 0
    skip = 0
    for did in dataset_ids:
        try:
            if convert_one_dataset(int(did)):
                ok += 1
            else:
                skip += 1
        except Exception as e:
            print(f"[fail] OpenML-ID-{did}: {e}")
            skip += 1

    print(f"\nDone. ok={ok}, skipped/failed={skip}, out={OUT_ROOT}")


if __name__ == "__main__":
    main()
