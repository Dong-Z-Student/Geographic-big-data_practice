# -*- coding: utf-8 -*-
"""
04 聚类分析（未标准化 + 可视化，对照实验）
- 直接在原始量纲 [x_utm, y_utm, value] 上做 DBSCAN
- 保留情感/主题权重 w
- 生成与标准化版本一致的 Folium 地图
"""

import json
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from shapely.geometry import Point
from shapely.ops import unary_union
from pyproj import Transformer

import folium
from folium.plugins import HeatMap


# =========================
# 参数区（⚠️ 未标准化版本专用）
# =========================
INPUT_FILE = "restaurant_features.json"
OUTPUT_MAP = "restaurant_spatial_analysis_no_scaling.html"
OUTPUT_CLUSTER_CSV = "restaurant_cluster_labels_no_scaling.csv"

# —— 情感聚类（原始量纲）——
EPS_SENTIMENT = 400          # 米
MIN_SAMPLES_SENTIMENT = 10
W_SENTIMENT = 50000.0        # 情感维度补偿权重（非常大）

# —— 主题聚类（原始量纲）——
EPS_TOPIC = 400              # 米
MIN_SAMPLES_TOPIC = 10
W_TOPIC = 50000.0

# 可视化参数
HEATMAP_RADIUS = 15
HEATMAP_BLUR = 20

# 聚类区域 buffer（米）
BUFFER_M_SENTIMENT = 100
BUFFER_M_TOPIC = 100


# =========================
# IO
# =========================
def load_json_lines(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


# =========================
# 坐标投影
# =========================
def project_to_utm18n(df):
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32618", always_xy=True
    )
    x, y = transformer.transform(df["longitude"].values, df["latitude"].values)
    df["x_utm"] = x
    df["y_utm"] = y
    return df


# =========================
# DBSCAN（❌ 不做标准化）
# =========================
def run_dbscan_no_scaling(df, value_col, weight, eps, min_samples):
    """
    d = sqrt(dx^2 + dy^2 + w * dv^2)
    """
    X = df[["x_utm", "y_utm", value_col]].values.astype(float)
    X[:, 2] *= np.sqrt(weight)

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean"
    )
    return model.fit_predict(X)


# =========================
# buffer + dissolve 生成区域
# =========================
def build_buffered_polygons_lonlat(df, cluster_col, buffer_m):
    transformer_back = Transformer.from_crs(
        "EPSG:32618", "EPSG:4326", always_xy=True
    )

    polygons = {}
    for cid, g in df.groupby(cluster_col):
        if cid == -1 or len(g) < 3:
            continue

        buffers = [Point(x, y).buffer(buffer_m)
                   for x, y in zip(g["x_utm"], g["y_utm"])]
        merged = unary_union(buffers)

        if merged.is_empty:
            continue

        def to_geojson(geom):
            if geom.geom_type == "Polygon":
                coords = [transformer_back.transform(x, y)
                          for x, y in geom.exterior.coords]
                return {"type": "Polygon", "coordinates": [coords]}
            elif geom.geom_type == "MultiPolygon":
                polys = []
                for poly in geom.geoms:
                    coords = [transformer_back.transform(x, y)
                              for x, y in poly.exterior.coords]
                    polys.append([coords])
                return {"type": "MultiPolygon", "coordinates": polys}
            return None

        geo = to_geojson(merged)
        if geo:
            polygons[int(cid)] = geo

    return polygons


def compute_cluster_mean_sentiment(df):
    tmp = (
        df[df["sentiment_cluster"] != -1]
        .groupby("sentiment_cluster")["avg_sentiment_polarity"]
        .mean()
    )
    return {int(k): float(v) for k, v in tmp.to_dict().items()}


# =========================
# Folium 样式函数
# =========================
def make_sentiment_style_func(mean_sent):
    color = "blue" if mean_sent >= 0 else "red"

    def _style(_):
        return {
            "fillColor": color,
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.25,
        }
    return _style


def make_topic_style_func():
    def _style(_):
        return {
            "fillColor": "purple",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.22,
        }
    return _style


# =========================
# 主流程
# =========================
def main():
    print("[LOAD]")
    df = load_json_lines(INPUT_FILE)
    df = project_to_utm18n(df)

    # ========= 情感聚类 =========
    print("[CLUSTER] sentiment (no scaling)")
    df["sentiment_cluster"] = run_dbscan_no_scaling(
        df,
        value_col="avg_sentiment_polarity",
        weight=W_SENTIMENT,
        eps=EPS_SENTIMENT,
        min_samples=MIN_SAMPLES_SENTIMENT,
    )

    sent_means = compute_cluster_mean_sentiment(df)
    sent_polys = build_buffered_polygons_lonlat(
        df, "sentiment_cluster", BUFFER_M_SENTIMENT
    )

    # ========= 主题聚类 =========
    topic_cols = sorted(
        [c for c in df.columns if c.startswith("topic_")],
        key=lambda x: int(x.split("_")[1])
    )

    topic_polys = {}
    for topic in topic_cols:
        aspect_col = f"{topic}_aspect"
        cluster_col = f"{topic}_cluster"

        df[aspect_col] = df[topic] * df["avg_sentiment_polarity"]

        print(f"[CLUSTER] {topic} (no scaling)")
        df[cluster_col] = run_dbscan_no_scaling(
            df,
            value_col=aspect_col,
            weight=W_TOPIC,
            eps=EPS_TOPIC,
            min_samples=MIN_SAMPLES_TOPIC,
        )

        topic_polys[topic] = build_buffered_polygons_lonlat(
            df, cluster_col, BUFFER_M_TOPIC
        )

    # ========= Folium 地图 =========
    print("[MAP]")
    center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    # A. 餐厅点
    fg_points = folium.FeatureGroup(name="Restaurants (Stars)")
    for _, r in df.iterrows():
        stars = float(r["stars"])
        color = "green" if stars >= 4 else ("orange" if stars >= 3 else "red")
        folium.CircleMarker(
            [r["latitude"], r["longitude"]],
            radius=3,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            color=None,
        ).add_to(fg_points)
    fg_points.add_to(m)

    # B. 情感热力图
    fg_heat = folium.FeatureGroup(name="Sentiment Heatmap")
    HeatMap(
        df[["latitude", "longitude", "avg_sentiment_polarity"]].values.tolist(),
        radius=HEATMAP_RADIUS,
        blur=HEATMAP_BLUR,
    ).add_to(fg_heat)
    fg_heat.add_to(m)

    # C. 情感 Hot / Cold
    fg_sent = folium.FeatureGroup(name="Sentiment Hot/Cold Spots (no scaling)")
    for cid, geo in sent_polys.items():
        folium.GeoJson(
            geo,
            style_function=make_sentiment_style_func(sent_means.get(cid, 0.0)),
            tooltip=f"cluster={cid}, mean={sent_means.get(cid, 0.0):.3f}",
        ).add_to(fg_sent)
    fg_sent.add_to(m)

    # D. 主题聚类
    for topic, polys in topic_polys.items():
        fg_topic = folium.FeatureGroup(name=f"{topic} clusters (no scaling)")
        style_func = make_topic_style_func()
        for cid, geo in polys.items():
            folium.GeoJson(
                geo,
                style_function=style_func,
                tooltip=f"{topic}_cluster={cid}",
            ).add_to(fg_topic)
        fg_topic.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    m.save(OUTPUT_MAP)
    print(f"[DONE] map saved -> {OUTPUT_MAP}")

    # ========= 输出 CSV =========
    out_cols = ["business_id", "sentiment_cluster"] + [f"{t}_cluster" for t in topic_cols]
    df[out_cols].to_csv(OUTPUT_CLUSTER_CSV, index=False, encoding="utf-8")
    print(f"[DONE] clusters saved -> {OUTPUT_CLUSTER_CSV}")

    # 诊断
    print("[DIAG] sentiment_cluster counts:")
    print(df["sentiment_cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main()
