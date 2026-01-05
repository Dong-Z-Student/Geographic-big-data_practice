# -*- coding: utf-8 -*-
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
from branca.element import MacroElement
from jinja2 import Template


# =========================
# 配置
# =========================
INPUT_FILE = "restaurant_features.json"   # 按行 JSON
OUTPUT_MAP_HTML = "restaurant_spatial_analysis_std.html"
OUTPUT_CLUSTER_CSV = "restaurant_cluster_labels_std.csv"

HOT_Q = 0.90
COLD_Q = 0.10

WGS84 = "EPSG:4326"
UTM18N = "EPSG:32618"

# —— 情感聚类参数（标准化后空间） —— #
SENT_EPS = 0.22
SENT_MIN_SAMPLES = 8
SENT_W = 5

# —— 主题聚类参数（标准化后空间） —— #
TOPIC_EPS = 0.18
TOPIC_MIN_SAMPLES = 6
TOPIC_W = 5

BUFFER_M_SENT = 100
BUFFER_M_TOPIC = 100

HEATMAP_RADIUS = 15
HEATMAP_BLUR = 20

POINT_RADIUS = 3


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
# DBSCAN：标准化 + 第三维权重
# =========================
def run_weighted_dbscan_standardized(
    df: pd.DataFrame,
    value_col: str,
    weight: float,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    X = df[["x_utm", "y_utm", value_col]].values.astype(float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs[:, 2] *= np.sqrt(weight)

    model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = model.fit_predict(Xs)
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
        if int(cid) == -1:
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
            polygons[int(cid)] = geo

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


# =========================
# Folium 样式函数
# =========================
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
# Stars 分级（5档）+ 图例
# =========================
def stars_to_color_5bin(stars: float) -> str:
    """
    0-1, 1-2, 2-3, 3-4, 4-5
    """
    if stars < 1.0:
        return "#d73027"  # 深红
    elif stars < 2.0:
        return "#fc8d59"  # 橙红
    elif stars < 3.0:
        return "#fee08b"  # 黄
    elif stars < 4.0:
        return "#91cf60"  # 绿
    else:
        return "#1a9850"  # 深绿


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

    must = ["business_id", "latitude", "longitude", "stars", "avg_sentiment_polarity"]
    for c in must:
        if c not in df.columns:
            raise KeyError(f"输入缺少字段：{c}")

    topic_cols = sorted(
        [c for c in df.columns if c.startswith("topic_") and c.split("_")[1].isdigit()],
        key=lambda x: int(x.split("_")[1])
    )
    print(f"[INFO] topics detected: {topic_cols}")

    df = project_to_utm18n(df)

    # ========= 情感聚类 =========
    print("[CLUSTER] sentiment DBSCAN...")
    df["sentiment_cluster"] = run_weighted_dbscan_standardized(
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

    # ========= 主题聚类 =========
    topic_polys_hotcold = {}
    for t in topic_cols:
        aspect_col = f"{t}_aspect"
        cluster_col = f"{t}_cluster"

        df[aspect_col] = df[t].astype(float) * df["avg_sentiment_polarity"].astype(float)

        print(f"[CLUSTER] {t} DBSCAN...")
        df[cluster_col] = run_weighted_dbscan_standardized(
            df,
            value_col=aspect_col,
            weight=TOPIC_W,
            eps=TOPIC_EPS,
            min_samples=TOPIC_MIN_SAMPLES,
        )

        mean_aspect = cluster_mean(df, cluster_col, aspect_col)
        hot_cids, cold_cids, hot_thr, cold_thr = select_hot_cold_clusters(mean_aspect, HOT_Q, COLD_Q)
        print(f"[{t}] clusters={len(mean_aspect)}, hot={len(hot_cids)}, cold={len(cold_cids)}, "
              f"hot_thr={hot_thr}, cold_thr={cold_thr}")

        polys_all = build_buffered_polygons_lonlat(df, cluster_col, BUFFER_M_TOPIC)
        polys_hot = {cid: geo for cid, geo in polys_all.items() if cid in hot_cids}
        polys_cold = {cid: geo for cid, geo in polys_all.items() if cid in cold_cids}

        topic_polys_hotcold[t] = {
            "hot": polys_hot,
            "cold": polys_cold,
            "mean_aspect": mean_aspect,
            "hot_thr": hot_thr,
            "cold_thr": cold_thr,
        }

    # ========= 地图 =========
    print("[MAP] building folium map...")
    center = [float(df["latitude"].mean()), float(df["longitude"].mean())]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    # 新的 5 档星级图例
    m.get_root().add_child(StarLegend5Bin())

    # a) 点图层：stars 5 档着色
    fg_points = folium.FeatureGroup(name="Restaurants (Stars)")
    for _, r in df.iterrows():
        stars = float(r["stars"])
        color = stars_to_color_5bin(stars)
        folium.CircleMarker(
            location=[float(r["latitude"]), float(r["longitude"])],
            radius=POINT_RADIUS,
            fill=True,
            fill_opacity=0.75,
            fill_color=color,
            color=None,
            tooltip=f"stars={stars:.1f}",
        ).add_to(fg_points)
    fg_points.add_to(m)

    # b) 情感热力图层
    fg_heat = folium.FeatureGroup(name="Sentiment Heatmap")
    HeatMap(
        data=df[["latitude", "longitude", "avg_sentiment_polarity"]].values.tolist(),
        radius=HEATMAP_RADIUS,
        blur=HEATMAP_BLUR,
    ).add_to(fg_heat)
    fg_heat.add_to(m)

    # c) 情感热点/冷点：按你的要求 HOT=红，COLD=蓝
    fg_sent = folium.FeatureGroup(name="Sentiment Hot/Cold (Top10%/Bottom10%)")
    hot_style = make_style("red", fill_opacity=0.25)
    cold_style = make_style("blue", fill_opacity=0.25)

    for cid, geo in sent_polys_hot.items():
        mean_v = float(sent_mean.get(cid, 0.0))
        folium.GeoJson(
            data=geo,
            style_function=hot_style,
            tooltip=f"HOT sent_cluster={cid}, mean_sent={mean_v:.3f}",
        ).add_to(fg_sent)

    for cid, geo in sent_polys_cold.items():
        mean_v = float(sent_mean.get(cid, 0.0))
        folium.GeoJson(
            data=geo,
            style_function=cold_style,
            tooltip=f"COLD sent_cluster={cid}, mean_sent={mean_v:.3f}",
        ).add_to(fg_sent)

    fg_sent.add_to(m)

    # d) 主题图层：同样 HOT=红，COLD=蓝
    for t, pack in topic_polys_hotcold.items():
        fg_t = folium.FeatureGroup(name=f"{t} Hot/Cold (Top10%/Bottom10%)")
        mean_aspect = pack["mean_aspect"]

        for cid, geo in pack["hot"].items():
            mv = float(mean_aspect.get(cid, 0.0))
            folium.GeoJson(
                data=geo,
                style_function=hot_style,
                tooltip=f"HOT {t}_cluster={cid}, mean_aspect={mv:.3f}",
            ).add_to(fg_t)

        for cid, geo in pack["cold"].items():
            mv = float(mean_aspect.get(cid, 0.0))
            folium.GeoJson(
                data=geo,
                style_function=cold_style,
                tooltip=f"COLD {t}_cluster={cid}, mean_aspect={mv:.3f}",
            ).add_to(fg_t)

        fg_t.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(OUTPUT_MAP_HTML)
    print(f"[DONE] map saved -> {OUTPUT_MAP_HTML}")

    # ========= 输出 CSV =========
    out_cols = ["business_id", "sentiment_cluster"] + [f"{t}_cluster" for t in topic_cols]
    out_df = df[out_cols].copy()
    out_df.to_csv(OUTPUT_CLUSTER_CSV, index=False, encoding="utf-8")
    print(f"[DONE] csv saved -> {OUTPUT_CLUSTER_CSV}")


if __name__ == "__main__":
    main()
