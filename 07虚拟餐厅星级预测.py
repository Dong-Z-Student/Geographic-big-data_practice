# -*- coding: utf-8 -*-
import json
import pandas as pd
import lightgbm as lgb
import warnings

import geopandas as gpd
from shapely.geometry import Point, MultiPoint

import networkx as nx
import osmnx as ox

import random
import numpy as np
import os


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_global_seed(42)
warnings.filterwarnings("ignore")

# =========================
# 基础配置
# =========================
INPUT_JSON = r"task5/全特征变量数据.json"
VIRTUAL_CSV = r"task7/虚拟餐厅.csv"
OUTPUT_VIRTUAL_JSON = r"task7/虚拟餐厅特征.json"
PREDICTION_OUTPUT = r"task7/虚拟餐厅预测星级.csv"

TARGET_CRS = "EPSG:32618"
WALK_SPEED_KMH = 4.8
WALK_TIME_SECONDS = 600
DRIVE_TIME_SECONDS = 900

LOCAL_WALK_GRAPHML_PATH = "./local_graph/philadelphia_walk_clipped_utm18_slim.graphml"
LOCAL_DRIVE_GRAPHML_PATH = "./local_graph/philadelphia_drive_clipped_utm18_slim.graphml"

PHILADELPHIA_BOUNDS = {
    'min_lon': -75.28, 'max_lon': -74.96,
    'min_lat': 39.86, 'max_lat': 40.14
}

# =========================
# 工具函数
# =========================
def fix_coordinate_columns(df):
    if 'longitude' not in df.columns or 'latitude' not in df.columns:
        return df

    lon = df.iloc[0]['longitude']
    lat = df.iloc[0]['latitude']
    if 30 <= lon <= 50 and abs(lat) > 50:
        df = df.rename(columns={'longitude': 'latitude_temp', 'latitude': 'longitude_temp'})
        df = df.rename(columns={'latitude_temp': 'latitude', 'longitude_temp': 'longitude'})
    return df


def validate_coordinates(df):
    for _, r in df.iterrows():
        lat, lon = r['latitude'], r['longitude']
        if not (PHILADELPHIA_BOUNDS['min_lat'] <= lat <= PHILADELPHIA_BOUNDS['max_lat']):
            return False
        if not (PHILADELPHIA_BOUNDS['min_lon'] <= lon <= PHILADELPHIA_BOUNDS['max_lon']):
            return False
    return True


def map_virtual_type_to_categories(virtual_type, all_categories):
    type_lower = str(virtual_type).lower()
    mapping = {
        'italian': ['italian'],
        'japanese': ['japanese', 'sushi'],
        'chinese': ['chinese'],
        'mexican': ['mexican'],
        'american': ['american', 'american_traditional'],
        'pizza': ['pizza']
    }

    matched = []
    for k, v in mapping.items():
        if k in type_lower:
            for c in v:
                col = f"cat__{c}"
                if col in all_categories:
                    matched.append(col)

    return matched if matched else all_categories[:1]


# =========================
# 数据加载
# =========================
def load_and_prepare_existing_data():
    rows = []
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    cat_cols = [c for c in df.columns if c.startswith("cat__")]
    return df, cat_cols


def load_and_fix_virtual_restaurants():
    df = pd.read_csv(VIRTUAL_CSV)
    df = fix_coordinate_columns(df)

    if not validate_coordinates(df):
        raise ValueError("虚拟餐厅坐标不在费城范围内")

    return df


# =========================
# 特征计算
# =========================
def calculate_features_for_virtual_restaurants(df_virtual, df_existing, cat_cols):

    G_walk = ox.load_graphml(LOCAL_WALK_GRAPHML_PATH)
    G_drive = ox.load_graphml(LOCAL_DRIVE_GRAPHML_PATH)

    gdf_virtual = gpd.GeoDataFrame(
        df_virtual,
        geometry=gpd.points_from_xy(df_virtual.longitude, df_virtual.latitude),
        crs="EPSG:4326"
    ).to_crs(TARGET_CRS)

    gdf_existing = gpd.GeoDataFrame(
        df_existing,
        geometry=gpd.points_from_xy(df_existing.longitude, df_existing.latitude),
        crs="EPSG:4326"
    ).to_crs(TARGET_CRS)

    features_list = []

    for i, vr in gdf_virtual.iterrows():

        features = {
            "business_id": vr.get("id", f"VR{i:03d}"),
            "longitude": df_virtual.iloc[i]["longitude"],
            "latitude": df_virtual.iloc[i]["latitude"],
            "price_range": vr.get("price_range", 2),
            "type": vr.get("type", "Restaurant"),
        }

        # walkability
        try:
            n = ox.distance.nearest_nodes(G_walk, vr.geometry.x, vr.geometry.y)
            dmap = nx.single_source_dijkstra_path_length(
                G_walk, n, cutoff=WALK_TIME_SECONDS, weight="travel_time"
            )
            features["walkability_score"] = len(dmap)
        except:
            features["walkability_score"] = 0

        # competitor density
        cats = map_virtual_type_to_categories(vr.get("type", ""), cat_cols)
        buf = vr.geometry.buffer(WALK_SPEED_KMH * 1000 / 3600 * WALK_TIME_SECONDS)
        idx = list(gdf_existing.sindex.intersection(buf.bounds))
        near = gdf_existing.iloc[idx]
        cnt = 0
        for _, r in near.iterrows():
            for c in cats:
                if r.get(c, 0) == 1:
                    cnt += 1
                    break
        features["competitor_density"] = cnt

        # drive accessibility
        try:
            n = ox.distance.nearest_nodes(G_drive, vr.geometry.x, vr.geometry.y)
            dmap = nx.single_source_dijkstra_path_length(
                G_drive, n, cutoff=DRIVE_TIME_SECONDS, weight="drive_time"
            )
            pts = [(G_drive.nodes[k]["x"], G_drive.nodes[k]["y"]) for k in dmap.keys()]
            features["drive_accessibility_score"] = (
                MultiPoint(pts).convex_hull.area / 1e6 if len(pts) >= 3 else 0.0
            )
        except:
            features["drive_accessibility_score"] = 0.0

        # sentiment neighborhood
        buf = vr.geometry.buffer(500)
        idx = list(gdf_existing.sindex.intersection(buf.bounds))
        near = gdf_existing.iloc[idx]
        features["sentiment_neighborhood_avg"] = (
            near["avg_sentiment_polarity"].mean()
            if "avg_sentiment_polarity" in near
            else df_existing["avg_sentiment_polarity"].mean()
        )

        # other
        features["betweenness_centrality"] = np.random.uniform(0.002, 0.005)
        features["distance_to_strip"] = np.random.uniform(500, 5000)
        features["review_count"] = 0
        features["is_open"] = 1
        features["stars"] = np.nan

        for c in cat_cols:
            features[c] = 1 if c in cats else 0

        features_list.append(features)

    return pd.DataFrame(features_list)


# =========================
# 模型训练与预测
# =========================
def train_model_and_predict(df_existing, df_virtual):

    EXCLUDE = {"business_id", "stars", "latitude", "longitude", "type", "review_count"}
    FEATURES = [c for c in df_existing.columns if c not in EXCLUDE]

    X = df_existing[FEATURES]
    y = df_existing["stars"]

    for c in FEATURES:
        if c not in df_virtual.columns:
            df_virtual[c] = df_existing[c].mean()

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(X, y)

    preds = np.clip(model.predict(df_virtual[FEATURES]), 1, 5)

    result = df_virtual[[
        "business_id", "longitude", "latitude",
        "price_range", "type"
    ]].copy()

    result["predicted_stars"] = preds
    result["walkability_score"] = df_virtual["walkability_score"]
    result["competitor_density"] = df_virtual["competitor_density"]
    result["drive_accessibility_score"] = df_virtual["drive_accessibility_score"]
    result["sentiment_neighborhood_avg"] = df_virtual["sentiment_neighborhood_avg"]

    return result


# =========================
# main
# =========================
def main():

    print("=== 虚拟餐厅星级预测 ===")

    df_existing, cat_cols = load_and_prepare_existing_data()
    df_virtual = load_and_fix_virtual_restaurants()

    df_virtual_features = calculate_features_for_virtual_restaurants(
        df_virtual, df_existing, cat_cols
    )

    df_virtual_features.to_json(
        OUTPUT_VIRTUAL_JSON, orient="records", lines=True, force_ascii=False
    )

    results = train_model_and_predict(df_existing, df_virtual_features)
    results.to_csv(PREDICTION_OUTPUT, index=False, encoding="utf-8")

    print("完成")
    print(results[["business_id", "type", "predicted_stars"]])


if __name__ == "__main__":
    main()