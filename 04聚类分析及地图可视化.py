# -*- coding: utf-8 -*-
"""
04-B 聚类分析 + 地图可视化（非标准化版本，用于与标准化版本对照）
说明：
- 与“标准化版本”保持同样功能与输出结构
- 唯一区别：DBSCAN 使用未标准化的 (x_utm, y_utm, value) 进行聚类
- 仍然使用加权距离：d = sqrt(dx^2 + dy^2 + w * dv^2)
  等价实现：把第三维乘 sqrt(w)，再用欧氏距离
- 不显示簇内点（只显示 Top10/Bottom10 的热点/冷点缓冲多边形）
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
from branca.element import MacroElement
from jinja2 import Template


# =========================
# 参数区（可自行调）
# =========================
INPUT_FILE = "restaurant_features.json"  # 按行 JSON（你前一步输出的餐厅级数据）
OUTPUT_MAP_HTML = "restaurant_spatial_analysis_nostd.html"
OUTPUT_CLUSTER_CSV = "restaurant_cluster_labels_nostd.csv"

HOT_Q = 0.90
COLD_Q = 0.10

WGS84 = "EPSG:4326"
UTM18N = "EPSG:32618"

# —— 情感聚类参数（非标准化空间） —— #
SENT_EPS = 400.0          # 单位：米（因为 x/y 是 UTM 米）
SENT_MIN_SAMPLES = 10
SENT_W = 50000.0         # 非标准化时 w 的量纲很敏感，通常需要比标准化版本大很多/小很多（看你的sent量级）

# —— 主题聚类参数（非标准化空间） —— #
TOPIC_EPS = 400.0         # 单位：米
TOPIC_MIN_SAMPLES = 10
TOPIC_W = 500000.0

# buffer（米），用于把簇内点缓冲成面
BUFFER_M_SENT = 100
BUFFER_M_TOPIC = 100

# heatmap 参数
HEATMAP_RADIUS = 15
HEATMAP_BLUR = 20


# =========================
# 工具：读取按行 JSON
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
# 投影：WGS84 -> UTM18N
# =========================
def project_to_utm18n(df: pd.DataFrame) -> pd.DataFrame:
    transformer = Transformer.from_crs(WGS84, UTM18N, always_xy=True)
    x, y = transformer.transform(df["longitude"].values, df["latitude"].values)
    df["x_utm"] = x
    df["y_utm"] = y
    return df


# =========================
# DBSCAN：非标准化 + 第三维权重
# d = sqrt(dx^2 + dy^2 + w*dv^2)
# 等价：用 [x, y, v*sqrt(w)] 的欧氏距离
# =========================
def run_weighted_dbscan_no_standardize(
    df: pd.DataFrame,
    value_col: str,
    weight: float,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    X = df[["x_utm", "y_utm", value_col]].values.astype(float)
    X[:, 2] *= np.sqrt(weight)

    model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = model.fit_predict(X)
    return labels


# =========================
# polygon：buffer + union（输出 lon/lat GeoJSON）
# =========================
def build_buffered_polygons_lonlat(
    df: pd.DataFrame,
    cluster_col: str,
    buffer_m: float,
) -> dict:
    transformer_back = Transformer.from_crs(UTM18N, WGS84, always_xy=True)
    polygons = {}

    for cid, g in df.groupby(cluster_col):
        cid = int(cid)
        if cid == -1:
            continue
        if len(g) < 3:
            continue

        buffers = [Point(x, y).buffer(buffer_m) for x, y in zip(g["x_utm"], g["y_utm"])]
        merged = unary_union(buffers)
        if merged.is_empty:
            continue

        def to_geojson(geom):
            if geom.geom_type == "Polygon":
                coords = [transformer_back.transform(x, y) for x, y in geom.exterior.coords]
                return {"type": "Polygon", "coordinates": [coords]}
            if geom.geom_type == "MultiPolygon":
                polys = []
                for poly in geom.geoms:
                    coords = [transformer_back.transform(x, y) for x, y in poly.exterior.coords]
                    polys.append([coords])
                return {"type": "MultiPolygon", "coordinates": polys}
            return None

        geo = to_geojson(merged)
        if geo is not None:
            polygons[cid] = geo

    return polygons


def cluster_mean(df: pd.DataFrame, cluster_col: str, value_col: str) -> pd.Series:
    sub = df[df[cluster_col] != -1]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby(cluster_col)[value_col].mean()


def select_hot_cold_clusters(mean_series: pd.Series, hot_q: float, cold_q: float):
    if mean_series.empty:
        return set(), set(), None, None
    hot_thr = float(mean_series.quantile(hot_q))
    cold_thr = float(mean_series.quantile(cold_q))
    hot_cids = set(mean_series[mean_series >= hot_thr].index.astype(int).tolist())
    cold_cids = set(mean_series[mean_series <= cold_thr].index.astype(int).tolist())
    return hot_cids, cold_cids, hot_thr, cold_thr


def make_style(color: str, fill_opacity: float = 0.25):
    def _style(_feature):
        return {
            "fillColor": color,
            "color": "black",
            "weight": 1,
            "fillOpacity": fill_opacity,
        }
    return _style


# =========================
# Stars 5档 + 图例
# =========================
def stars_to_color_5bin(stars: float) -> str:
    if stars < 1.0:
        return "#d73027"
    elif stars < 2.0:
        return "#fc8d59"
    elif stars < 3.0:
        return "#fee08b"
    elif stars < 4.0:
        return "#91cf60"
    else:
        return "#1a9850"


class StarLegend5Bin(MacroElement):
    def __init__(self):
        super().__init__()
        self._template = Template("""
        {% macro html(this, kwargs) %}
        <div style="
            position: fixed;
            bottom: 30px;
            left: 30px;
            z-index: 9999;
            background: white;
            padding: 10px 12px;
            border: 1px solid rgba(0,0,0,0.2);
            border-radius: 6px;
            font-size: 13px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        ">
          <div style="font-weight: 600; margin-bottom: 6px;">Stars Legend</div>
          <div><span style="display:inline-block;width:10px;height:10px;background:#d73027;margin-right:6px;border:1px solid #666;"></span> 0 – 1</div>
          <div><span style="display:inline-block;width:10px;height:10px;background:#fc8d59;margin-right:6px;border:1px solid #666;"></span> 1 – 2</div>
          <div><span style="display:inline-block;width:10px;height:10px;background:#fee08b;margin-right:6px;border:1px solid #666;"></span> 2 – 3</div>
          <div><span style="display:inline-block;width:10px;height:10px;background:#91cf60;margin-right:6px;border:1px solid #666;"></span> 3 – 4</div>
          <div><span style="display:inline-block;width:10px;height:10px;background:#1a9850;margin-right:6px;border:1px solid #666;"></span> 4 – 5</div>
        </div>
        {% endmacro %}
        """)


# =========================
# 主流程
# =========================
def main():
    print("[LOAD] reading data...")
    df = load_json_lines(INPUT_FILE)

    must = ["business_id", "latitude", "longitude", "stars",
            "avg_sentiment_polarity", "avg_sentiment_subjectivity"]
    for c in must:
        if c not in df.columns:
            raise KeyError(f"输入缺少字段：{c}")

    # 自动识别 topic_i
    topic_cols = sorted(
        [c for c in df.columns if c.startswith("topic_") and c.split("_")[1].isdigit()],
        key=lambda x: int(x.split("_")[1])
    )
    print(f"[INFO] topics detected: {topic_cols}")

    # 投影
    df = project_to_utm18n(df)

    # ========= 情感聚类（非标准化） =========
    print("[CLUSTER] sentiment DBSCAN (no-std)...")
    df["sentiment_cluster"] = run_weighted_dbscan_no_standardize(
        df,
        value_col="avg_sentiment_polarity",
        weight=SENT_W,
        eps=SENT_EPS,
        min_samples=SENT_MIN_SAMPLES,
    )

    sent_mean = cluster_mean(df, "sentiment_cluster", "avg_sentiment_polarity")
    sent_hot, sent_cold, sent_hot_thr, sent_cold_thr = select_hot_cold_clusters(sent_mean, HOT_Q, COLD_Q)
    print(f"[SENT] clusters={len(sent_mean)}, hot={len(sent_hot)}, cold={len(sent_cold)}, "
          f"hot_thr={sent_hot_thr}, cold_thr={sent_cold_thr}")

    sent_polys_all = build_buffered_polygons_lonlat(df, "sentiment_cluster", BUFFER_M_SENT)
    sent_polys_hot = {cid: geo for cid, geo in sent_polys_all.items() if cid in sent_hot}
    sent_polys_cold = {cid: geo for cid, geo in sent_polys_all.items() if cid in sent_cold}

    # ========= 主题聚类（非标准化） =========
    topic_pack = {}
    for t in topic_cols:
        aspect_col = f"{t}_aspect"
        cluster_col = f"{t}_cluster"

        # 主题强度 × 情感极性（带正负）
        df[aspect_col] = df[t].astype(float) * df["avg_sentiment_polarity"].astype(float)

        print(f"[CLUSTER] {t} DBSCAN (no-std)...")
        df[cluster_col] = run_weighted_dbscan_no_standardize(
            df,
            value_col=aspect_col,
            weight=TOPIC_W,
            eps=TOPIC_EPS,
            min_samples=TOPIC_MIN_SAMPLES,
        )

        mean_aspect = cluster_mean(df, cluster_col, aspect_col)
        hot_cids, cold_cids, hot_thr, cold_thr = select_hot_cold_clusters(mean_aspect, HOT_Q, COLD_Q)

        polys_all = build_buffered_polygons_lonlat(df, cluster_col, BUFFER_M_TOPIC)
        polys_hot = {cid: geo for cid, geo in polys_all.items() if cid in hot_cids}
        polys_cold = {cid: geo for cid, geo in polys_all.items() if cid in cold_cids}

        topic_pack[t] = {
            "cluster_col": cluster_col,
            "aspect_col": aspect_col,
            "mean_aspect": mean_aspect,
            "polys_hot": polys_hot,
            "polys_cold": polys_cold,
        }

        print(f"[{t}] clusters={len(mean_aspect)}, hot={len(polys_hot)}, cold={len(polys_cold)}")

    # ========= 地图 =========
    print("[MAP] building folium map...")
    center = [float(df["latitude"].mean()), float(df["longitude"].mean())]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    m.get_root().add_child(StarLegend5Bin())

    # a) Stars 点图层（5档颜色）
    fg_points = folium.FeatureGroup(name="Restaurants (Stars)")
    for _, r in df.iterrows():
        folium.CircleMarker(
            location=[float(r["latitude"]), float(r["longitude"])],
            radius=3,
            fill=True,
            fill_opacity=0.75,
            fill_color=stars_to_color_5bin(float(r["stars"])),
            color=None,
            tooltip=f"stars={float(r['stars']):.1f}",
        ).add_to(fg_points)
    fg_points.add_to(m)

    # b) 情感热力图
    fg_heat = folium.FeatureGroup(name="Sentiment Heatmap")
    HeatMap(
        data=df[["latitude", "longitude", "avg_sentiment_polarity"]].values.tolist(),
        radius=HEATMAP_RADIUS,
        blur=HEATMAP_BLUR,
    ).add_to(fg_heat)
    fg_heat.add_to(m)

    # c) 情感 Hot/Cold 多边形（只画 Top10/Bottom10）
    fg_sent_area = folium.FeatureGroup(name="Sentiment Hot/Cold (Top10%/Bottom10%)")

    hot_area_style = make_style("red", fill_opacity=0.25)   # HOT=红
    cold_area_style = make_style("blue", fill_opacity=0.25) # COLD=蓝

    for cid, geo in sent_polys_hot.items():
        mean_v = float(sent_mean.get(cid, 0.0))
        folium.GeoJson(
            data=geo,
            style_function=hot_area_style,
            tooltip=f"HOT sentiment_cluster={cid}, mean_sent={mean_v:.3f}",
        ).add_to(fg_sent_area)

    for cid, geo in sent_polys_cold.items():
        mean_v = float(sent_mean.get(cid, 0.0))
        folium.GeoJson(
            data=geo,
            style_function=cold_area_style,
            tooltip=f"COLD sentiment_cluster={cid}, mean_sent={mean_v:.3f}",
        ).add_to(fg_sent_area)

    fg_sent_area.add_to(m)

    # d) 每个 topic 一个图层：只画 Top10/Bottom10 的 Hot/Cold 多边形
    for t, pack in topic_pack.items():
        fg_t = folium.FeatureGroup(name=f"{t} Hot/Cold (Top10%/Bottom10%)")
        mean_aspect = pack["mean_aspect"]

        for cid, geo in pack["polys_hot"].items():
            mv = float(mean_aspect.get(cid, 0.0))
            folium.GeoJson(
                data=geo,
                style_function=hot_area_style,
                tooltip=f"HOT {t}_cluster={cid}, mean_aspect={mv:.3f}",
            ).add_to(fg_t)

        for cid, geo in pack["polys_cold"].items():
            mv = float(mean_aspect.get(cid, 0.0))
            folium.GeoJson(
                data=geo,
                style_function=cold_area_style,
                tooltip=f"COLD {t}_cluster={cid}, mean_aspect={mv:.3f}",
            ).add_to(fg_t)

        fg_t.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(OUTPUT_MAP_HTML)
    print(f"[DONE] map saved -> {OUTPUT_MAP_HTML}")

    # ========= 输出 CSV =========
    out_cols = ["business_id", "sentiment_cluster"] + [f"{t}_cluster" for t in topic_cols]
    df[out_cols].to_csv(OUTPUT_CLUSTER_CSV, index=False, encoding="utf-8")
    print(f"[DONE] csv saved -> {OUTPUT_CLUSTER_CSV}")

    print("[ALL DONE]")


if __name__ == "__main__":
    main()
