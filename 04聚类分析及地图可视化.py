# -*- coding: utf-8 -*-
"""
04 聚类分析及地图可视化（稳定版）
- 情感极性空间聚类（DBSCAN）：x/y/avg_sentiment_polarity 标准化 + sentiment 权重 w
- 主题聚类：topic_i * avg_sentiment_polarity（aspect）后做同样 DBSCAN
- 地图可视化（Folium）：
  1) 餐厅点（按 stars 着色）
  2) 情感热力图
  3) 情感 Hot/Cold Spots（buffer+dissolve 生成区域，多边形按簇均值正负上色）
  4) topic_i clusters（每个 topic 一个图层）
"""

import json
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from shapely.geometry import Point
from shapely.ops import unary_union
from pyproj import Transformer

import folium
from folium.plugins import HeatMap


# =========================
# 参数区（你主要改这里）
# =========================
INPUT_FILE = "restaurant_features.json"   # 按行 JSON（即 jsonl）
OUTPUT_MAP = "restaurant_spatial_analysis.html"
OUTPUT_CLUSTER_CSV = "restaurant_cluster_labels.csv"


# DBSCAN 参数（在标准化后的空间）
EPS = 0.15
MIN_SAMPLES = 10
SENTIMENT_WEIGHT = 10.0       # w：情感/方面维度的重要性（在标准化空间中）

# 可视化参数
HEATMAP_RADIUS = 15
HEATMAP_BLUR = 20

# 聚类区域生成：对簇点在 UTM 平面做 buffer（米）再 dissolve
BUFFER_M_SENTIMENT = 300     # 情感聚类区域 buffer 半径
BUFFER_M_TOPIC = 300         # 主题聚类区域 buffer 半径


# =========================
# IO
# =========================
def load_json_lines(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


# =========================
# 坐标投影：WGS84 -> UTM18N
# =========================
def project_to_utm18n(df: pd.DataFrame) -> pd.DataFrame:
    """
    Philadelphia 对应 UTM 18N (EPSG:32618)
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
    x, y = transformer.transform(df["longitude"].values, df["latitude"].values)
    df["x_utm"] = x
    df["y_utm"] = y
    return df


# =========================
# DBSCAN：标准化 + sentiment 权重
# =========================
def run_weighted_dbscan(df: pd.DataFrame, value_col: str, weight: float) -> np.ndarray:
    """
    对 [x_utm, y_utm, value_col] 做 StandardScaler，
    并将第三维乘 sqrt(weight)，再 DBSCAN。
    """
    X = df[["x_utm", "y_utm", value_col]].values.astype(float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # 第三维加权（保持欧式距离形式）
    Xs[:, 2] *= np.sqrt(weight)

    model = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric="euclidean")
    labels = model.fit_predict(Xs)
    return labels


# =========================
# buffer + dissolve 生成“聚类区域”
# =========================
def build_buffered_polygons_lonlat(
    df: pd.DataFrame,
    cluster_col: str,
    buffer_m: float,
    utm_crs: str = "EPSG:32618",
    wgs84_crs: str = "EPSG:4326",
) -> dict:
    """
    在 UTM 平面坐标（米）上对每个簇的点做 buffer，再 union。
    将 union 结果转换为 lon/lat GeoJSON-like dict (Polygon/MultiPolygon)。
    """
    transformer_back = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)

    polygons = {}

    for cid, g in df.groupby(cluster_col):
        if cid == -1:
            continue
        if len(g) < 3:
            continue

        buffers = [Point(x, y).buffer(buffer_m) for x, y in zip(g["x_utm"], g["y_utm"])]
        merged = unary_union(buffers)

        def to_geojson(geom):
            if geom.is_empty:
                return None

            if geom.geom_type == "Polygon":
                coords = [transformer_back.transform(x, y) for x, y in geom.exterior.coords]
                return {"type": "Polygon", "coordinates": [coords]}

            if geom.geom_type == "MultiPolygon":
                polys = []
                for poly in geom.geoms:
                    coords = [transformer_back.transform(x, y) for x, y in poly.exterior.coords]
                    polys.append([coords])
                return {"type": "MultiPolygon", "coordinates": polys}

            # 其他类型（不太会出现），忽略
            return None

        geo = to_geojson(merged)
        if geo is not None:
            polygons[int(cid)] = geo

    return polygons


# =========================
# 统计：情感簇均值（用于 hot/cold 着色）
# =========================
def compute_cluster_mean_sentiment(df: pd.DataFrame, cluster_col: str = "sentiment_cluster") -> dict:
    tmp = df[df[cluster_col] != -1].groupby(cluster_col)["avg_sentiment_polarity"].mean()
    return {int(k): float(v) for k, v in tmp.to_dict().items()}


# =========================
# Folium 样式工厂函数（彻底避免 lambda 捕获变量）
# =========================
def make_sentiment_style_func(mean_sent: float):
    """
    mean_sent >= 0 -> blue (Hot)
    mean_sent < 0  -> red (Cold)
    """
    color = "blue" if mean_sent >= 0 else "red"

    def _style(_feature):
        return {
            "fillColor": color,
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.25,
        }

    return _style


def make_topic_style_func():
    def _style(_feature):
        return {
            "fillColor": "purple",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.22,
        }

    return _style


# =========================
# 主函数
# =========================
def main():
    print("[LOAD] data")
    df = load_json_lines(INPUT_FILE)

    required = ["latitude", "longitude", "stars", "avg_sentiment_polarity"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"缺少必要字段：{c}")

    print("[PROJ] WGS84 -> UTM 18N (EPSG:32618)")
    df = project_to_utm18n(df)

    # =========================
    # 1) 情感极性聚类
    # =========================
    print("[CLUSTER] sentiment polarity")
    df["sentiment_cluster"] = run_weighted_dbscan(df, "avg_sentiment_polarity", SENTIMENT_WEIGHT)

    sent_cluster_mean = compute_cluster_mean_sentiment(df, "sentiment_cluster")
    sent_polys = build_buffered_polygons_lonlat(df, "sentiment_cluster", BUFFER_M_SENTIMENT)

    if not sent_polys:
        print("[WARN] sentiment_cluster: 没有生成任何区域（可能簇太小/多数为噪声）。")

    # =========================
    # 2) 主题聚类（topic_i 自动识别）
    # =========================
    topic_cols = sorted(
        [c for c in df.columns if c.startswith("topic_")],
        key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 999999
    )

    topic_polys = {}  # topic -> {cid: geojson}
    for topic in topic_cols:
        aspect_col = f"{topic}_aspect"
        cluster_col = f"{topic}_cluster"

        # 方面质量：讨论强度 * 情感极性
        df[aspect_col] = df[topic].astype(float) * df["avg_sentiment_polarity"].astype(float)

        print(f"[CLUSTER] {topic}")
        df[cluster_col] = run_weighted_dbscan(df, aspect_col, SENTIMENT_WEIGHT)
        polys = build_buffered_polygons_lonlat(df, cluster_col, BUFFER_M_TOPIC)

        if not polys:
            print(f"[WARN] {topic}: 没有生成任何区域（可能簇太小/多数为噪声）。")

        topic_polys[topic] = polys

    # =========================
    # 3) Folium 可视化
    # =========================
    print("[MAP] building map")
    center = [float(df["latitude"].mean()), float(df["longitude"].mean())]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    # ---- A. 基础点图：按 stars 着色 ----
    fg_points = folium.FeatureGroup(name="Restaurants (Stars)")
    for _, r in df.iterrows():
        stars = float(r["stars"])
        if stars >= 4.0:
            fc = "green"
        elif stars >= 3.0:
            fc = "orange"
        else:
            fc = "red"

        folium.CircleMarker(
            location=[float(r["latitude"]), float(r["longitude"])],
            radius=3,
            fill=True,
            fill_opacity=0.7,
            fill_color=fc,
            color=None,
            tooltip=f"stars={stars:.1f}",
        ).add_to(fg_points)
    fg_points.add_to(m)

    # ---- B. 情感热力图 ----
    fg_heat = folium.FeatureGroup(name="Sentiment Heatmap")
    HeatMap(
        data=df[["latitude", "longitude", "avg_sentiment_polarity"]].values.tolist(),
        radius=HEATMAP_RADIUS,
        blur=HEATMAP_BLUR,
    ).add_to(fg_heat)
    fg_heat.add_to(m)

    # ---- C. 情感 Hot/Cold Spots（一定会出现在图层中） ----
    fg_sent = folium.FeatureGroup(name="Sentiment Hot/Cold Spots")
    for cid, geo in sent_polys.items():
        mean_s = sent_cluster_mean.get(cid, 0.0)
        folium.GeoJson(
            data=geo,
            style_function=make_sentiment_style_func(mean_s),
            tooltip=f"sent_cluster={cid}, mean_sent={mean_s:.3f}",
        ).add_to(fg_sent)
    fg_sent.add_to(m)

    # ---- D. 主题聚类图层（topic_i 可切换） ----
    for topic, polys in topic_polys.items():
        fg_topic = folium.FeatureGroup(name=f"{topic} clusters")
        style_func = make_topic_style_func()  # 固定样式，无闭包问题

        for cid, geo in polys.items():
            folium.GeoJson(
                data=geo,
                style_function=style_func,
                tooltip=f"{topic}_cluster={cid}",
            ).add_to(fg_topic)

        fg_topic.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    m.save(OUTPUT_MAP)
    print(f"[DONE] map saved -> {OUTPUT_MAP}")

    # =========================
    # 输出聚类标签（CSV）
    # =========================
    print("[OUTPUT] saving cluster labels (CSV)")

    cluster_cols = ["sentiment_cluster"] + [f"{t}_cluster" for t in topic_cols]

    out_df = df[["business_id"] + cluster_cols].copy()

    out_df.to_csv(
        OUTPUT_CLUSTER_CSV,
        index=False,
        encoding="utf-8"
    )

    print(f"[DONE] cluster labels saved -> {OUTPUT_CLUSTER_CSV}")


if __name__ == "__main__":
    main()
