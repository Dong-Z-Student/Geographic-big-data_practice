# -*- coding: utf-8 -*-
"""
Stage 1 - Walk network analysis (tiled OSMnx download)
Outputs: walkability_score, competitor_density
- Auto-fetch Philadelphia polygon from OSM (Nominatim)
- Download walk network by tiles (polygon clipped) to avoid Overpass connection reset
- Project to UTM 18N (EPSG:32618)
- For each restaurant: nearest node -> 10-min isochrone (600s) -> node count + competitor count

MODIFIED:
- Removed primary_cat_col completely
- Competitor definition:
  Two restaurants are competitors if they share ANY cat__*** column where both == 1

Author: ChatGPT
"""

import os
import time
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

import networkx as nx
import osmnx as ox

warnings.filterwarnings("ignore")


# =========================
# User Config
# =========================
INPUT_json = r"restaurant_features.json"   # TODO: 改成你的输入文件（JSON Lines）
OUTPUT_json = r"restaurants_stage1_walk_scores.json"  # 输出（JSON Lines）

PLACE_NAME = "Philadelphia, Pennsylvania, USA"

# 投影：UTM 18N
TARGET_CRS = "EPSG:32618"

# 步行参数
WALK_SPEED_KMH = 4.8                 # 常用假设：4~5 km/h
WALK_TIME_SECONDS = 600              # 10 min = 600s

# 分块大小（单位：米，基于 UTM 投影后）
TILE_SIZE_M = 2500                   # 2.5km 网格（稳妥；如果仍失败可降到 1500~2000）

# tile 与 polygon 的缓冲（米），避免边缘断裂导致路网不连
TILE_BUFFER_M = 50

# OSMnx 缓存与请求设置
OSMNX_CACHE_DIR = "./cache"
REQUEST_TIMEOUT = 180
RETRIES = 5
BACKOFF_SECONDS = 4

# 可选：换 Overpass endpoint（当某个节点不稳时）
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]


# =========================
# Helpers
# =========================
def configure_osmnx():
    os.makedirs(OSMNX_CACHE_DIR, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = OSMNX_CACHE_DIR
    ox.settings.log_console = True
    ox.settings.timeout = REQUEST_TIMEOUT


def get_place_polygon(place_name: str) -> gpd.GeoDataFrame:
    """
    从 Nominatim 获取 place polygon（可能是 MultiPolygon）
    """
    gdf = ox.geocode_to_gdf(place_name)
    if gdf.empty:
        raise RuntimeError(f"Failed to geocode place: {place_name}")

    # 有时会返回多个对象，取面积最大的那个（更稳）
    gdf = gdf.to_crs("EPSG:4326")
    gdf["__area__"] = gdf.geometry.area
    gdf = gdf.sort_values("__area__", ascending=False).head(1).drop(columns="__area__")
    return gdf


def polygon_to_tiles(poly_utm, tile_size_m: float, buffer_m: float) -> List:
    """
    将 polygon 的外包范围切成规则网格，然后与 polygon 相交生成 tile polygon
    """
    minx, miny, maxx, maxy = poly_utm.bounds
    minx -= buffer_m
    miny -= buffer_m
    maxx += buffer_m
    maxy += buffer_m

    tiles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = box(x, y, x + tile_size_m, y + tile_size_m)
            inter = cell.intersection(poly_utm)
            if not inter.is_empty:
                inter2 = inter.buffer(buffer_m)
                tiles.append(inter2)
            y += tile_size_m
        x += tile_size_m

    cleaned = []
    for t in tiles:
        if t.area >= (tile_size_m * tile_size_m * 0.02):  # 至少占 2% 的tile面积
            cleaned.append(t)

    print(f"[INFO] tiles generated: {len(cleaned)}")
    return cleaned


def try_download_graph_from_polygon(poly_wgs84, network_type="walk") -> nx.MultiDiGraph:
    """
    带重试 + 切换 overpass endpoint 的下载函数
    """
    last_err = None
    for ep in OVERPASS_ENDPOINTS:
        ox.settings.overpass_endpoint = ep
        for k in range(RETRIES):
            try:
                G = ox.graph_from_polygon(
                    poly_wgs84,
                    network_type=network_type,
                    simplify=True,
                    retain_all=False,
                    truncate_by_edge=True,
                )
                return G
            except Exception as e:
                last_err = e
                wait = BACKOFF_SECONDS * (k + 1)
                print(f"[WARN] download failed (endpoint={ep}, retry={k+1}/{RETRIES}): {e}")
                time.sleep(wait)
    raise RuntimeError(f"All retries failed downloading graph. Last error: {last_err}")


def merge_graphs(graphs: List[nx.MultiDiGraph]) -> nx.MultiDiGraph:
    graphs = [g for g in graphs if g is not None and len(g) > 0]
    if not graphs:
        raise RuntimeError("No graphs to merge.")
    return nx.compose_all(graphs)


def add_walk_time_to_edges(G: nx.MultiDiGraph, walk_speed_kmh: float):
    meters_per_sec = (walk_speed_kmh * 1000.0) / 3600.0
    for u, v, k, data in G.edges(keys=True, data=True):
        length_m = data.get("length", None)
        if length_m is None:
            data["travel_time"] = np.nan
        else:
            data["travel_time"] = float(length_m) / meters_per_sec


def build_rest_gdf(df: pd.DataFrame, crs="EPSG:4326") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=crs,
    )


def compute_isochrone_nodes(G: nx.MultiDiGraph, origin_node: int, max_time_s: float) -> List[int]:
    lengths = nx.single_source_dijkstra_path_length(
        G, origin_node, cutoff=max_time_s, weight="travel_time"
    )
    return list(lengths.keys())


def build_cat_bitmask(df: pd.DataFrame, cat_cols: List[str]) -> np.ndarray:
    """
    将每行 cat__*** (0/1) 编成一个 Python int bitmask（91列也没问题）。
    两餐厅竞争：mask[i] & mask[j] != 0
    """
    if not cat_cols:
        return np.zeros(len(df), dtype=object)

    # 确保是 0/1
    mat = df[cat_cols].fillna(0).astype(np.int8).to_numpy()
    masks = np.empty(mat.shape[0], dtype=object)

    for i in range(mat.shape[0]):
        m = 0
        row = mat[i]
        # 把为1的列置位
        # 91列很小，这个循环非常快
        for bit, val in enumerate(row):
            if val:
                m |= (1 << bit)
        masks[i] = m
    return masks


def main():
    configure_osmnx()

    print("[1] Load restaurant data")
    df = pd.read_json(INPUT_json, lines=True, encoding="utf-8")

    required = ["business_id", "latitude", "longitude"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Input missing required column: {c}")

    cat_cols = [c for c in df.columns if c.startswith("cat__")]
    print(f"[INFO] category columns: {len(cat_cols)}")

    # 为 competitor 判定准备 bitmask（核心改动）
    cat_masks = build_cat_bitmask(df, cat_cols)

    rest_gdf = build_rest_gdf(df, crs="EPSG:4326")

    print("[2] Fetch Philadelphia polygon (OSM Nominatim)")
    place_gdf = get_place_polygon(PLACE_NAME)
    place_utm = place_gdf.to_crs(TARGET_CRS)
    place_poly_utm = unary_union(place_utm.geometry.values)

    print("[3] Generate tiles over polygon (UTM)")
    tiles_utm = polygon_to_tiles(place_poly_utm, TILE_SIZE_M, TILE_BUFFER_M)

    print("[4] Download walk network by tiles (Overpass) and merge")
    graphs = []
    for i, tile_utm in enumerate(tiles_utm, start=1):
        print(f"  - tile {i}/{len(tiles_utm)}: download walk network")
        tile_wgs = gpd.GeoSeries([tile_utm], crs=TARGET_CRS).to_crs("EPSG:4326").iloc[0]
        try:
            Gi = try_download_graph_from_polygon(tile_wgs, network_type="walk")
            graphs.append(Gi)
            print(f"    nodes={len(Gi.nodes):,}, edges={len(Gi.edges):,}")
        except Exception as e:
            print(f"    [ERROR] tile failed, skipped: {e}")

    G = merge_graphs(graphs)
    print(f"[INFO] merged graph: nodes={len(G.nodes):,}, edges={len(G.edges):,}")

    print("[5] Project graph to UTM 18N")
    G = ox.project_graph(G, to_crs=TARGET_CRS)

    print("[5.5] Clip merged graph to Philadelphia boundary polygon (UTM)")
    G = ox.truncate.truncate_graph_polygon(G, place_poly_utm, truncate_by_edge=True)
    print(f"[INFO] clipped graph: nodes={len(G.nodes):,}, edges={len(G.edges):,}")

    print("[6] Add walking travel_time to edges")
    add_walk_time_to_edges(G, WALK_SPEED_KMH)

    rest_utm = rest_gdf.to_crs(TARGET_CRS)

    print("[7] Snap restaurants to nearest network nodes")
    xs = rest_utm.geometry.x.to_numpy()
    ys = rest_utm.geometry.y.to_numpy()
    rest_nodes = ox.distance.nearest_nodes(G, X=xs, Y=ys)

    rest_utm["__node__"] = rest_nodes

    # 把“哪些餐厅落在哪个节点”先聚合起来
    node_to_rest_idx: Dict[int, List[int]] = {}
    for idx, n in enumerate(rest_nodes):
        node_to_rest_idx.setdefault(int(n), []).append(idx)

    walkability = np.zeros(len(rest_utm), dtype=int)
    competitor = np.zeros(len(rest_utm), dtype=int)

    print("[8] Compute walkability_score and competitor_density (10-min walk isochrone)")
    for i in range(len(rest_utm)):
        origin = int(rest_nodes[i])
        iso_nodes = compute_isochrone_nodes(G, origin, WALK_TIME_SECONDS)
        walkability[i] = len(iso_nodes)

        # competitor: 统计等时圈内“与我共享任意 cat__***==1 的其他餐厅数量”
        my_mask = cat_masks[i]
        if not my_mask:
            competitor[i] = 0
        else:
            cnt = 0
            for n in iso_nodes:
                for ridx in node_to_rest_idx.get(int(n), []):
                    if ridx == i:
                        continue
                    if (my_mask & cat_masks[ridx]) != 0:
                        cnt += 1
            competitor[i] = cnt

        if (i + 1) % 200 == 0:
            print(f"  processed {i+1}/{len(rest_utm)}")

    out = df.copy()
    out["walkability_score"] = walkability
    out["competitor_density"] = competitor

    print("[9] Save output (JSON Lines)")
    out.to_json(
        OUTPUT_json,
        orient="records",
        lines=True,
        force_ascii=False
    )
    print(f"[OK] wrote: {OUTPUT_json} (rows={len(out)})")


if __name__ == "__main__":
    main()
