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
# 基础配置
# =========================
INPUT_FILE = r"task3/情感极性及主题建模后数据_抽样0.5.json"
OUTPUT_MAP_HTML = r"task4/餐厅聚类分析地图.html"
OUTPUT_CLUSTER_CSV = r"task4/餐厅聚类标签.csv"

HOT_Q = 0.90
COLD_Q = 0.10

WGS84 = "EPSG:4326"
UTM18N = "EPSG:32618"

# 情感聚类参数
SENT_EPS = 0.22
SENT_MIN_SAMPLES = 8
SENT_W = 10

# 主题聚类参数
TOPIC_EPS = 0.18
TOPIC_MIN_SAMPLES = 6
TOPIC_W = 10
TOPIC_NAME_MAP = {
    "topic_0": "服务",
    "topic_1": "味道",
    "topic_2": "菜品",
}

# 缓冲区设置
BUFFER_M_SENT = 500
BUFFER_M_TOPIC = 500

# 热力图设置
HEATMAP_RADIUS = 15
HEATMAP_BLUR = 20

POINT_RADIUS = 3

# =========================
# 工具函数
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


def project_to_utm18n(df: pd.DataFrame) -> pd.DataFrame:
    transformer = Transformer.from_crs(WGS84, UTM18N, always_xy=True)
    x, y = transformer.transform(df["longitude"].values, df["latitude"].values)
    df["x_utm"] = x
    df["y_utm"] = y
    return df


# =========================
# DBSCAN聚类函数
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
# 绘图工具函数
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


def build_merged_region_polygons_lonlat(
    df: pd.DataFrame,
    cluster_col: str,
    buffer_m: float,
    target_cids: set,
) -> list:
    if not target_cids:
        return []

    sub = df[df[cluster_col].isin(list(target_cids))].copy()
    if sub.empty:
        return []

    buffers = [Point(x, y).buffer(buffer_m) for x, y in zip(sub["x_utm"], sub["y_utm"])]
    merged = unary_union(buffers)
    if merged.is_empty:
        return []

    if merged.geom_type == "Polygon":
        geoms = [merged]
    elif merged.geom_type == "MultiPolygon":
        geoms = list(merged.geoms)
    else:
        return []

    transformer_back = Transformer.from_crs(UTM18N, WGS84, always_xy=True)

    def poly_to_geojson(poly):
        coords = [transformer_back.transform(x, y) for x, y in poly.exterior.coords]
        return {"type": "Polygon", "coordinates": [coords]}

    return [poly_to_geojson(g) for g in geoms]


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
# 图例
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
# main
# =========================
def main():
    print("[LOAD] 读取数据")
    df = load_json_lines(INPUT_FILE)

    must = ["business_id", "latitude", "longitude", "avg_stars", "avg_sentiment_polarity"]
    for c in must:
        if c not in df.columns:
            raise KeyError(f"输入缺少字段：{c}")

    topic_cols = sorted(
        [c for c in df.columns if c.startswith("topic_") and c.split("_")[1].isdigit()],
        key=lambda x: int(x.split("_")[1])
    )

    # 三个主题（服务/味道/菜品）
    topic_cols = [t for t in topic_cols if t in TOPIC_NAME_MAP]
    print(f"[INFO] 主题字段如下: {topic_cols}")
    print(f"[INFO] 主题映射如下: { {t: TOPIC_NAME_MAP[t] for t in topic_cols} }")

    df = project_to_utm18n(df)

    # 情感聚类
    print("[CLUSTER] 情感聚类")
    df["sentiment_cluster"] = run_weighted_dbscan_standardized(
        df,
        value_col="avg_sentiment_polarity",
        weight=SENT_W,
        eps=SENT_EPS,
        min_samples=SENT_MIN_SAMPLES,
    )

    sent_mean = cluster_mean(df, "sentiment_cluster", "avg_sentiment_polarity")
    sent_hot, sent_cold, sent_hot_thr, sent_cold_thr = select_hot_cold_clusters(sent_mean, HOT_Q, COLD_Q)
    print(f"[SENT] 聚类数：{len(sent_mean)}, 热点数：{len(sent_hot)}, 冷点数：{len(sent_cold)}, "
          f"热点阈值：{sent_hot_thr}, 冷点阈值：{sent_cold_thr}")

    # 合并
    sent_hot_geos = build_merged_region_polygons_lonlat(df, "sentiment_cluster", BUFFER_M_SENT, sent_hot)
    sent_cold_geos = build_merged_region_polygons_lonlat(df, "sentiment_cluster", BUFFER_M_SENT, sent_cold)

    # 主题聚类
    topic_polys_hotcold = {}
    for t in topic_cols:
        topic_name = TOPIC_NAME_MAP.get(t, t)

        aspect_col = f"{t}_aspect"
        cluster_col = f"{t}_cluster"

        df[aspect_col] = df[t].astype(float) * df["avg_sentiment_polarity"].astype(float)

        print(f"[CLUSTER] 主题 {topic_name} ({t}) 聚类")
        df[cluster_col] = run_weighted_dbscan_standardized(
            df,
            value_col=aspect_col,
            weight=TOPIC_W,
            eps=TOPIC_EPS,
            min_samples=TOPIC_MIN_SAMPLES,
        )

        mean_aspect = cluster_mean(df, cluster_col, aspect_col)
        hot_cids, cold_cids, hot_thr, cold_thr = select_hot_cold_clusters(mean_aspect, HOT_Q, COLD_Q)
        print(f"[{topic_name}/{t}] 聚类数：{len(mean_aspect)}, 热点数：{len(hot_cids)}, 冷点数：{len(cold_cids)}, "
              f"热点阈值：{hot_thr}, 冷点阈值：{cold_thr}")

        # 合并
        hot_merged = build_merged_region_polygons_lonlat(df, cluster_col, BUFFER_M_TOPIC, hot_cids)
        cold_merged = build_merged_region_polygons_lonlat(df, cluster_col, BUFFER_M_TOPIC, cold_cids)

        topic_polys_hotcold[t] = {
            "topic_name": topic_name,
            "hot": hot_merged,
            "cold": cold_merged,
            "mean_aspect": mean_aspect,
            "hot_thr": hot_thr,
            "cold_thr": cold_thr,
        }

    # ========= 地图 =========
    print("[MAP] 绘制地图")
    center = [float(df["latitude"].mean()), float(df["longitude"].mean())]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    m.get_root().add_child(StarLegend5Bin())

    # 点图层：stars 5 档着色
    fg_points = folium.FeatureGroup(name="餐厅点位", show=True)
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

    # 情感热力图层
    fg_heat = folium.FeatureGroup(name="情感分析热力图层", show=False)
    HeatMap(
        data=df[["latitude", "longitude", "avg_sentiment_polarity"]].values.tolist(),
        radius=HEATMAP_RADIUS,
        blur=HEATMAP_BLUR,
    ).add_to(fg_heat)
    fg_heat.add_to(m)

    # 情感热点/冷点
    fg_sent = folium.FeatureGroup(name="情感热点/冷点区域", show=False)
    hot_style = make_style("red", fill_opacity=0.25)
    cold_style = make_style("blue", fill_opacity=0.25)

    for geo in sent_hot_geos:
        folium.GeoJson(
            data=geo,
            style_function=hot_style,
            tooltip="情感 热点",
        ).add_to(fg_sent)

    for geo in sent_cold_geos:
        folium.GeoJson(
            data=geo,
            style_function=cold_style,
            tooltip="情感 冷点",
        ).add_to(fg_sent)

    fg_sent.add_to(m)

    # 主题图层
    for t, pack in topic_polys_hotcold.items():
        topic_name = pack.get("topic_name", t)
        fg_t = folium.FeatureGroup(name=f"主题：{topic_name} 热点/冷点区域", show=False)

        for geo in pack["hot"]:
            folium.GeoJson(
                data=geo,
                style_function=hot_style,
                tooltip=f"{topic_name} 热点",
            ).add_to(fg_t)

        for geo in pack["cold"]:
            folium.GeoJson(
                data=geo,
                style_function=cold_style,
                tooltip=f"{topic_name} 冷点",
            ).add_to(fg_t)

        fg_t.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(OUTPUT_MAP_HTML)
    print(f"[DONE] 地图已保存为 {OUTPUT_MAP_HTML}")

    # 输出 CSV
    out_cols = ["business_id", "sentiment_cluster"] + [f"{t}_cluster" for t in topic_cols]
    out_df = df[out_cols].copy()

    rename_map = {f"{t}_cluster": f"{TOPIC_NAME_MAP.get(t, t)}_cluster" for t in topic_cols}
    out_df = out_df.rename(columns=rename_map)

    out_df.to_csv(OUTPUT_CLUSTER_CSV, index=False, encoding="utf-8")
    print(f"[DONE] 聚类文件已保存为 {OUTPUT_CLUSTER_CSV}")


if __name__ == "__main__":
    main()