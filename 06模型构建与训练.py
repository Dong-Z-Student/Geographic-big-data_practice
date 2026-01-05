# -*- coding: utf-8 -*-
import json
import os
from typing import List, Set, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error


# 设置中文字体，确保中文正常显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ===== 数据加载 =====
def load_jsonl_to_dataframe(input_jsonl_path: str) -> pd.DataFrame:
    """
    读取 JSON 文件
    """
    rows = []
    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    print(f"[INFO] 数据加载: {df.shape}")
    return df


# ==== 特征变量与目标变量准备 ====
def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: Set[str]
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    构建 X / y，并返回特征列名列表
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' 不在数据列中。")

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"[INFO] 特征变量数量: {len(feature_cols)}")

    # 仅在特征工程阶段做必要处理：长尾变量对数化 + 将 price_range 视为类别特征
    X = df[feature_cols].copy()

    # 1) 长尾变量：log1p（避免 log(0)）
    log1p_cols = [
        "betweenness_centrality",
        "competitor_density",
        "distance_to_strip",
        "review_count",
        "topic_1",
    ]
    for col in log1p_cols:
        if col in X.columns:
            # 保守处理：强制数值化；负值视为缺失并置 0，再做 log1p
            s = pd.to_numeric(X[col], errors="coerce")
            s = s.where(s >= 0, np.nan).fillna(0.0)
            X[col] = np.log1p(s)

    # 2) price_range：离散档位，作为类别特征（不对 cat_*** 的 0/1 指示变量做额外处理）
    if "price_range" in X.columns:
        X["price_range"] = X["price_range"].astype("category")

    y = df[target_col]
    return X, y, feature_cols


# ==== 划分数据集 ====
def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    按比例划分训练/验证/测试集
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1.0 - train_ratio),
        random_state=random_state
    )
    # val:test = val_ratio:test_ratio
    val_portion = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1.0 - val_portion),
        random_state=random_state
    )

    print(f"[INFO] 训练集大小: {X_train.shape[0]}")
    print(f"[INFO] 验证集大小: {X_val.shape[0]}")
    print(f"[INFO] 测试集大小: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ==== 模型训练 ====
def train_lightgbm_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
    n_jobs: int = -1,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    early_stopping_rounds: int = 50
) -> lgb.LGBMRegressor:
    """
    训练 LightGBM 回归模型
    """
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=random_state,
        n_jobs=n_jobs
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]
    )
    print("[INFO] 训练完成")
    return model


# ==== 精度评估 ====
def evaluate_rmse(
    model: lgb.LGBMRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[float, pd.Series]:
    """
    返回 RMSE 与测试集预测值
    """
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"[RESULT] 测试集RMSE = {rmse:.4f}")
    return rmse, pd.Series(y_pred, index=y_test.index)


# ==== 散点图 ====
def plot_pred_vs_true_scatter(
    y_true: pd.Series,
    y_pred: pd.Series,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    绘制预测值与真实值散点图
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([1, 5], [1, 5], "r--", linewidth=1)
    plt.xlabel("真实星级")
    plt.ylabel("预测星级")
    plt.title("预测值与真实值散点图")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"[INFO] 散点图已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ==== 特征重要性柱状图 ====
def plot_feature_importance_topk(
    model: lgb.LGBMRegressor,
    feature_cols: List[str],
    top_k: int = 30,
    save_path: Optional[str] = None,
    show: bool = True
) -> pd.DataFrame:
    """
    绘制 Top-K 特征重要性柱状图
    """
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    top_feat_imp = feat_imp_df.head(top_k)

    plt.figure(figsize=(8, 6))
    plt.barh(
        top_feat_imp["feature"][::-1],
        top_feat_imp["importance"][::-1]
    )
    plt.xlabel("特征重要性")
    plt.title(f"前 {top_k} 特征重要性柱状图")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"[INFO] 特征重要性图已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return feat_imp_df


# ==== main ====
def main(
    input_json: str = r"task5/全特征变量数据.json",
    scatter_save_path: str = r"task6/散点图_连续星级.png",
    importance_save_path: str = r"task6/前30特征重要性柱状图_连续星级.png",
    show_plots: bool = True,
    top_k: int = 30
) -> None:
    # 读取数据
    df = load_jsonl_to_dataframe(input_json)

    # 特征变量
    target_col = "avg_stars"
    exclude_cols = {"business_id", "stars", "avg_stars", "latitude", "longitude"}
    X, y, feature_cols = prepare_features_and_target(df, target_col, exclude_cols)

    # 切分训练集、验证集、测试集
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)

    # 训练
    model = train_lightgbm_regressor(X_train, y_train, X_val, y_val)

    # 精度评估
    rmse, y_pred_test = evaluate_rmse(model, X_test, y_test)

    # 散点图
    plot_pred_vs_true_scatter(
        y_true=y_test,
        y_pred=y_pred_test,
        save_path=scatter_save_path,
        show=show_plots
    )

    # 特征重要性柱状图
    _ = plot_feature_importance_topk(
        model=model,
        feature_cols=feature_cols,
        top_k=top_k,
        save_path=importance_save_path,
        show=show_plots
    )


if __name__ == "__main__":
    main()
