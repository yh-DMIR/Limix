import pandas as pd
import numpy as np
import os
from pathlib import Path
import openml
from sklearn.model_selection import train_test_split
import time
from scipy import sparse

# 创建保存目录
save_dir = Path("./dataset/limix/tabzilla27")
save_dir.mkdir(parents=True, exist_ok=True)

# 问题数据集专门处理
problem_datasets = [
    ("ada_agnostic", 389),  # 这个有稀疏矩阵问题
]

# 其他数据集
other_datasets = [
    ("artificial-characters", 1497),
    ("mfeat-fourier", 14),
    ("credit-g", 31),
    ("colic", 25),
    ("jasmine", 847),
    ("mfeat-zernike", 23),
    ("elevators", 40671),
    ("qsar-biodeg", 1494),
    ("monks-problems-2", 334),
    ("Australian", 40981),
    ("cnae-9", 1467),
    ("balance-scale", 11),
    ("credit-approval", 29),
    ("electricity", 151),
    ("phoneme", 1489),
    ("heart-h", 51),
    ("socmob", 1479),
    ("jungle_chess_2pcs_raw_endgame_complete", 41027),
    ("GesturePhaseSegmentationProcessed", 4538),
    ("Bioresponse", 4135),
    ("profb", 1470),
    ("splice", 46),
    ("SpeedDating", 40536),
    ("nomao", 1486),
    ("kc1", 1067),
    ("vehicle", 54)
]


def handle_sparse_special(dataset_name, dataset_id):
    """专门处理稀疏矩阵数据集"""
    try:
        print(f"专门处理稀疏矩阵: {dataset_name} (ID: {dataset_id})")

        # 获取数据集
        dataset = openml.datasets.get_dataset(dataset_id)

        # 使用array格式获取数据，避免DataFrame的稀疏问题
        data = dataset.get_data(dataset_format="array")

        if len(data) == 2:
            X, y = data
        else:
            # 有些数据集返回格式不同
            X, y, _, _ = dataset.get_data(dataset_format="dataframe")

            # 强制转换为numpy数组
            if hasattr(X, 'toarray'):  # 稀疏矩阵
                X = X.toarray()
            elif isinstance(X, pd.DataFrame):
                X = X.to_numpy()

        # 确保X是密集数组
        if sparse.issparse(X):
            X = X.toarray()

        # 确保y是一维数组
        if hasattr(y, 'shape') and len(y.shape) > 1:
            y = y.flatten()

        print(f"  数据形状: X={X.shape}, y={y.shape}")

        # 创建DataFrame
        if X.ndim == 2:
            n_features = X.shape[1]
            columns = [f'feature_{i}' for i in range(n_features)]
            X_df = pd.DataFrame(X, columns=columns)
        else:
            print(f"  错误: X的维度不是2D: {X.ndim}")
            return False, None

        # 创建目标Series
        y_series = pd.Series(y, name='target')

        # 合并数据
        df = pd.concat([X_df, y_series], axis=1)

        # 简单的NaN处理（用0填充）
        df = df.fillna(0)

        # 分割数据集
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

        # 创建文件夹
        folder_name = f"{dataset_id}_{dataset_name}"
        folder_path = save_dir / folder_name
        folder_path.mkdir(exist_ok=True)

        # 保存文件
        train_df.to_csv(folder_path / f"{folder_name}_train.csv", index=False)
        test_df.to_csv(folder_path / f"{folder_name}_test.csv", index=False)

        print(f"  ✓ 保存成功: {len(train_df)}训练, {len(test_df)}测试")
        return True, folder_name

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False, None


def download_normal_dataset(dataset_name, dataset_id):
    """下载普通数据集"""
    try:
        print(f"处理: {dataset_name} (ID: {dataset_id})")

        # 获取数据集
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data(dataset_format="dataframe")

        # 确保是DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 重置索引
        X = X.reset_index(drop=True)
        if isinstance(y, pd.Series):
            y = y.reset_index(drop=True)

        # 合并数据
        data = pd.concat([X, y], axis=1)
        data.columns = list(X.columns) + ['target']

        # 简单的NaN处理
        for col in data.columns:
            if data[col].dtype.kind in 'iufc':  # 数值类型
                data[col] = data[col].fillna(0)
            else:
                data[col] = data[col].fillna('missing')

        # 转换分类变量为数值
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = pd.factorize(data[col])[0]

        # 分割数据集
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        # 创建文件夹
        folder_name = f"{dataset_id}_{dataset_name}"
        folder_path = save_dir / folder_name
        folder_path.mkdir(exist_ok=True)

        # 保存文件
        train_data.to_csv(folder_path / f"{folder_name}_train.csv", index=False)
        test_data.to_csv(folder_path / f"{folder_name}_test.csv", index=False)

        print(f"  ✓ 保存: {len(train_data)}训练, {len(test_data)}测试")
        return True, folder_name

    except Exception as e:
        print(f"  ✗ 错误: {e}")
        return False, None


# 主程序
print(f"下载 {len(problem_datasets) + len(other_datasets)} 个数据集")
print(f"保存到: {save_dir}")
print("-" * 50)

success_folders = []

# 首先处理有问题的数据集
print("\n处理稀疏矩阵数据集:")
for name, did in problem_datasets:
    success, folder = handle_sparse_special(name, did)
    if success:
        success_folders.append(folder)
    time.sleep(1)

# # 处理其他数据集
# print("\n处理普通数据集:")
# for i, (name, did) in enumerate(other_datasets, 1):
#     print(f"\n[{i}/{len(other_datasets)}] ", end="")
#     success, folder = download_normal_dataset(name, did)
#     if success:
#         success_folders.append(folder)
#     time.sleep(1)
#
# # 生成结果
# print(f"\n" + "=" * 50)
# print(f"完成! 成功: {len(success_folders)}/{len(problem_datasets) + len(other_datasets)}")
#
# if success_folders:
#     # 保存清单
#     with open(save_dir / "datasets.txt", "w") as f:
#         for folder in success_folders:
#             f.write(f"{folder}\n")
#
#     print(f"\n清单: {save_dir}/datasets.txt")
#     print(f"\n使用命令:")
#     print(f"python inference_classifier.py --data_dir {save_dir}")