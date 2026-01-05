# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体，确保中文正常显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# =========================
# 基础配置
# =========================
INPUT_JSON = r"task5/全特征变量数据.json"
OUTPUT_DIR = "特征变量分布"
BINS = 50

FEATURES_TO_PLOT = [
    "avg_stars",
    "price_range",
    "stars",
    "review_count",
    "avg_sentiment_polarity",
    "avg_sentiment_subjectivity",
    "topic_0",
    "topic_1",
    "topic_2",
    "walkability_score",
    "competitor_density",
    "drive_accessibility_score",
    "betweenness_centrality",
    "distance_to_strip",
    "sentiment_neighborhood_avg"
]

# =========================
# 工具函数
# =========================
def load_json_lines(path):
    """读取 JSON Lines 文件"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def plot_histogram(df, col, output_dir):
    """绘制并保存单个字段的直方图"""
    series = df[col].dropna()

    if series.empty:
        print(f"[WARN] {col} 为空, 跳过")
        return

    plt.figure(figsize=(6, 4))
    plt.hist(series, bins=BINS)
    plt.title(f"{col} 分布直方图")
    plt.xlabel(col)
    plt.ylabel("数量")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{col}分布直方图.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[OK] 保存直方图: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[INFO] 加载特征变量数据")
    df = load_json_lines(INPUT_JSON)
    print(f"[INFO] 加载数据量 {df.shape}")

    for col in FEATURES_TO_PLOT:
        if col not in df.columns:
            print(f"[WARN] 未找到目标列: {col}, 跳过")
            continue

        plot_histogram(df, col, OUTPUT_DIR)

    print("[DONE] 特征变量分布分析完成")


if __name__ == "__main__":
    main()
