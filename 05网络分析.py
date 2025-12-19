# -*- coding: utf-8 -*-
import os
import time
import warnings
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, MultiPoint
from shapely.ops import unary_union

import networkx as nx
import osmnx as ox

warnings.filterwarnings("ignore")


# =========================
# User Config
# =========================
INPUT_json = r"restaurant_features.json"   # TODO: 改成你的输入文件（JSON Lines）
# ✅ 最终只输出一个文件（Stage1 + Stage2 全部字段）
OUTPUT_json = r"restaurants_walk_drive_scores.json"  # 输出（JSON Lines）

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

# 并行线程数（按 unique origin node）
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# ===== 本地路网持久化（最终处理后的 G：已投影 + 已裁剪）=====
LOCAL_GRAPH_DIR = "./local_graph"
LOCAL_WALK_GRAPHML_PATH = os.path.join(LOCAL_GRAPH_DIR, "philadelphia_walk_clipped_utm18.graphml")

# ===== Stage2: Drive config =====
DRIVE_TIME_SECONDS = 900  # 15 min = 900s
LOCAL_DRIVE_GRAPHML_PATH = os.path.join(LOCAL_GRAPH_DIR, "philadelphia_drive_clipped_utm18.graphml")

# Drive 默认限速（km/h）：当 edge 无 maxspeed 或无法解析时按 highway 兜底
# 你可以按需要调整
DRIVE_DEFAULT_SPEED_KMH = {
    "motorway": 100,
    "motorway_link": 60,
    "trunk": 80,
    "trunk_link": 50,
    "primary": 60,
    "primary_link": 45,
    "secondary": 50,
    "secondary_link": 40,
    "tertiary": 40,
    "tertiary_link": 35,
    "residential": 30,
    "unclassified": 30,
    "living_street": 15,
    "service": 20,
    "road": 30,
}
DRIVE_FALLBACK_SPEED_KMH = 30  # 最终兜底


# =========================
# Helpers (Stage1 保持原逻辑；Stage2 在下方新增，不覆盖原函数行为)
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


def _is_no_nodes_polygon_error(e: Exception) -> bool:
    msg = str(e) if e is not None else ""
    return "Found no graph nodes within the requested polygon" in msg


def try_download_graph_from_polygon(poly_wgs84, network_type="walk") -> nx.MultiDiGraph:
    """
    带重试 + 切换 overpass endpoint 的下载函数
    - 若遇到“polygon内无节点”，直接返回 None（不重试，不换endpoint）
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
                # 关键改动：无节点 => 直接退出（不重试）
                if _is_no_nodes_polygon_error(e):
                    print(f"[WARN] tile has no graph nodes, skipped immediately: {e}")
                    return None

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


def compute_isochrone_dists(G: nx.MultiDiGraph, origin_node: int, max_time_s: float) -> Dict[int, float]:
    """
    返回: 可达节点 -> travel_time
    """
    return nx.single_source_dijkstra_path_length(
        G, origin_node, cutoff=max_time_s, weight="travel_time"
    )


def build_cat_bitmask(df: pd.DataFrame, cat_cols: List[str]) -> np.ndarray:
    """
    将每行 cat__*** (0/1) 编成一个 Python int bitmask（91列也没问题）。
    两餐厅竞争：mask[i] & mask[j] != 0
    """
    if not cat_cols:
        return np.zeros(len(df), dtype=object)

    mat = df[cat_cols].fillna(0).astype(np.int8).to_numpy()
    masks = np.empty(mat.shape[0], dtype=object)

    for i in range(mat.shape[0]):
        m = 0
        row = mat[i]
        for bit, val in enumerate(row):
            if val:
                m |= (1 << bit)
        masks[i] = m
    return masks


# ===== GraphML 本地加载/保存（Stage1 已用；Stage2复用）=====
def load_graph_if_exists(graphml_path: str) -> Optional[nx.MultiDiGraph]:
    """
    若本地 graphml 存在，则直接读取并返回；否则返回 None
    """
    if graphml_path and os.path.exists(graphml_path):
        print(f"[INFO] Local graph found, loading: {graphml_path}")
        G_local = ox.load_graphml(graphml_path)

        # 兼容：GraphML 可能把节点ID读成 str，这里尽量转回 int（不影响其它功能）
        try:
            any_node = next(iter(G_local.nodes))
            if isinstance(any_node, str) and any_node.isdigit():
                mapping = {n: int(n) for n in G_local.nodes if isinstance(n, str) and n.isdigit()}
                if len(mapping) == len(G_local.nodes):
                    G_local = nx.relabel_nodes(G_local, mapping, copy=True)
        except Exception:
            pass

        print(f"[INFO] loaded local graph: nodes={len(G_local.nodes):,}, edges={len(G_local.edges):,}")
        return G_local
    return None


def save_graph_local(G: nx.MultiDiGraph, graphml_path: str):
    """
    将最终路网保存为 graphml
    """
    if not graphml_path:
        return
    os.makedirs(os.path.dirname(graphml_path), exist_ok=True)
    ox.save_graphml(G, filepath=graphml_path)
    print(f"[INFO] saved local graph: {graphml_path}")


# =========================
# Stage2 NEW Helpers (Drive)
# =========================
def _pick_highway_value(highway_attr) -> Optional[str]:
    if highway_attr is None:
        return None
    if isinstance(highway_attr, (list, tuple)) and len(highway_attr) > 0:
        # 取第一个（OSMnx常见）
        return str(highway_attr[0])
    return str(highway_attr)


def _parse_maxspeed_to_kmh(maxspeed_attr) -> Optional[float]:
    """
    尝试解析 OSM maxspeed：
    - 可能是 "35", "25 mph", ["35", "45"], "signals", "walk" 等
    """
    if maxspeed_attr is None:
        return None

    candidates = []
    if isinstance(maxspeed_attr, (list, tuple)):
        candidates = [str(x) for x in maxspeed_attr if x is not None]
    else:
        candidates = [str(maxspeed_attr)]

    parsed_vals = []
    for s in candidates:
        s2 = s.strip().lower()
        if not s2:
            continue
        # 过滤明显非数值
        if any(tok in s2 for tok in ["signals", "variable", "none", "walk", "national", "urban", "rural"]):
            continue

        # 提取数字
        import re
        nums = re.findall(r"(\d+(\.\d+)?)", s2)
        if not nums:
            continue
        v = float(nums[0][0])

        # mph 转 km/h
        if "mph" in s2:
            v = v * 1.609344
        parsed_vals.append(v)

    if not parsed_vals:
        return None
    # 多值取最小更保守
    return float(min(parsed_vals))


def add_drive_time_to_edges(G: nx.MultiDiGraph,
                            default_speed_map_kmh: Dict[str, float],
                            fallback_speed_kmh: float):
    """
    为 drive 图的每条边添加 drive_time（秒），权重字段名：drive_time
    逻辑：
      - 优先 maxspeed
      - 否则按 highway 默认表
      - 否则 fallback
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        length_m = data.get("length", None)
        if length_m is None:
            data["drive_time"] = np.nan
            continue

        # 1) maxspeed
        ms = _parse_maxspeed_to_kmh(data.get("maxspeed", None))
        if ms is None or ms <= 0:
            # 2) highway 默认
            hw = _pick_highway_value(data.get("highway", None))
            ms = default_speed_map_kmh.get(hw, None)
        if ms is None or ms <= 0:
            ms = float(fallback_speed_kmh)

        meters_per_sec = (ms * 1000.0) / 3600.0
        data["drive_time"] = float(length_m) / meters_per_sec


def compute_isochrone_dists_drive(G: nx.MultiDiGraph, origin_node: int, max_time_s: float) -> Dict[int, float]:
    """
    drive 的 dijkstra：weight=drive_time
    """
    return nx.single_source_dijkstra_path_length(
        G, origin_node, cutoff=max_time_s, weight="drive_time"
    )


def convex_hull_area_km2_from_nodes(node_ids: List[int],
                                    node_x: Dict[int, float],
                                    node_y: Dict[int, float]) -> float:
    """
    用可达节点点集的凸包面积作为等时圈面积（km²）
    - 点太少/退化时做小 buffer 避免面积为 0
    """
    pts = []
    for n in node_ids:
        x = node_x.get(int(n), None)
        y = node_y.get(int(n), None)
        if x is None or y is None:
            continue
        if np.isnan(x) or np.isnan(y):
            continue
        pts.append((x, y))

    if len(pts) < 3:
        return 0.0

    geom = MultiPoint(pts).convex_hull
    area_m2 = float(geom.area)

    # 退化（共线）时 area=0，做一个极小 buffer（1m）避免全为0
    if area_m2 <= 0:
        area_m2 = float(MultiPoint(pts).buffer(1.0).area)

    return area_m2 / 1e6


def build_drive_graph(place_poly_utm,
                      target_crs: str,
                      graphml_path: str) -> nx.MultiDiGraph:
    """
    Stage2: drive 路网构建逻辑（tile下载 + merge + project + clip + 本地保存）
    若本地存在 graphml_path，则直接 load。
    """
    Gd = load_graph_if_exists(graphml_path)
    if Gd is not None:
        print("[INFO] using local DRIVE graph directly (skip download/merge/project/clip)")
        return Gd

    print("[S2-1] Generate tiles over polygon (UTM) for DRIVE")
    tiles_utm = polygon_to_tiles(place_poly_utm, TILE_SIZE_M, TILE_BUFFER_M)

    print("[S2-2] Download DRIVE network by tiles (Overpass) and merge")
    graphs = []
    for i, tile_utm in enumerate(tiles_utm, start=1):
        print(f"  - tile {i}/{len(tiles_utm)}: download drive network")
        tile_wgs = gpd.GeoSeries([tile_utm], crs=target_crs).to_crs("EPSG:4326").iloc[0]
        try:
            Gi = try_download_graph_from_polygon(tile_wgs, network_type="drive")
            if Gi is None or len(Gi.nodes) == 0:
                print(f"    [WARN] empty graph, skipped.")
                continue
            graphs.append(Gi)
            print(f"    nodes={len(Gi.nodes):,}, edges={len(Gi.edges):,}")
        except Exception as e:
            print(f"    [ERROR] tile failed, skipped: {e}")

    Gd = merge_graphs(graphs)
    print(f"[INFO] merged DRIVE graph: nodes={len(Gd.nodes):,}, edges={len(Gd.edges):,}")

    print("[S2-3] Project DRIVE graph to UTM 18N")
    Gd = ox.project_graph(Gd, to_crs=target_crs)

    print("[S2-4] Clip DRIVE graph to place polygon (UTM)")
    Gd = ox.truncate.truncate_graph_polygon(Gd, place_poly_utm, truncate_by_edge=True)
    print(f"[INFO] clipped DRIVE graph: nodes={len(Gd.nodes):,}, edges={len(Gd.edges):,}")

    save_graph_local(Gd, graphml_path)
    return Gd


# =========================
# Main
# =========================
def main():
    start_time = time.time()

    configure_osmnx()

    print("[1] Load restaurant data")
    df = pd.read_json(INPUT_json, lines=True, encoding="utf-8")

    required = ["business_id", "latitude", "longitude"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Input missing required column: {c}")

    cat_cols = [c for c in df.columns if c.startswith("cat__")]
    print(f"[INFO] category columns: {len(cat_cols)}")

    # competitor 判定 bitmask
    cat_masks = build_cat_bitmask(df, cat_cols)

    rest_gdf = build_rest_gdf(df, crs="EPSG:4326")

    print("[2] Fetch Philadelphia polygon (OSM Nominatim)")
    place_gdf = get_place_polygon(PLACE_NAME)
    place_utm = place_gdf.to_crs(TARGET_CRS)
    place_poly_utm = unary_union(place_utm.geometry.values)

    # =========================
    # Stage 1 (walk) —— 原逻辑保持不变（仅把“写文件”延后到最后一次性写）
    # =========================
    print("\n========== Stage 1: WALK ==========")

    # ===== 优先从本地加载已经“投影+裁剪”的最终路网 =====
    G = load_graph_if_exists(LOCAL_WALK_GRAPHML_PATH)

    if G is None:
        print("[3] Generate tiles over polygon (UTM)")
        tiles_utm = polygon_to_tiles(place_poly_utm, TILE_SIZE_M, TILE_BUFFER_M)

        print("[4] Download walk network by tiles (Overpass) and merge")
        graphs = []
        for i, tile_utm in enumerate(tiles_utm, start=1):
            print(f"  - tile {i}/{len(tiles_utm)}: download walk network")
            tile_wgs = gpd.GeoSeries([tile_utm], crs=TARGET_CRS).to_crs("EPSG:4326").iloc[0]
            try:
                Gi = try_download_graph_from_polygon(tile_wgs, network_type="walk")
                if Gi is None or len(Gi.nodes) == 0:
                    print(f"    [WARN] empty graph, skipped.")
                    continue
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

        # 保存最终路网到本地（下次直接加载）
        save_graph_local(G, LOCAL_WALK_GRAPHML_PATH)
    else:
        print("[INFO] using local graph directly (skip download/merge/project/clip)")

    print("[6] Add walking travel_time to edges")
    add_walk_time_to_edges(G, WALK_SPEED_KMH)

    rest_utm = rest_gdf.to_crs(TARGET_CRS)

    print("[7] Snap restaurants to nearest network nodes")
    xs = rest_utm.geometry.x.to_numpy()
    ys = rest_utm.geometry.y.to_numpy()
    rest_nodes = np.asarray(ox.distance.nearest_nodes(G, X=xs, Y=ys), dtype=np.int64)

    rest_utm["__node__"] = rest_nodes

    # node -> restaurant indices
    node_to_rest_idx: Dict[int, List[int]] = {}
    for idx, n in enumerate(rest_nodes):
        node_to_rest_idx.setdefault(int(n), []).append(idx)

    # 预取 node 坐标，给 isochrone bbox 用（投影后 G.nodes[u]['x'/'y'] 是米）
    node_x: Dict[int, float] = {}
    node_y: Dict[int, float] = {}
    for n, data in G.nodes(data=True):
        node_x[int(n)] = float(data.get("x", np.nan))
        node_y[int(n)] = float(data.get("y", np.nan))

    # sindex 直接返回行索引（不会有 STRtree id 不一致问题）
    rest_sindex = rest_utm.sindex

    walkability = np.zeros(len(rest_utm), dtype=np.int32)
    competitor = np.zeros(len(rest_utm), dtype=np.int32)

    print("[8] Compute walkability_score and competitor_density (10-min walk isochrone) by unique origin nodes (parallel)")

    def process_one_origin(origin_node: int, origin_rest_indices: List[int]) -> Tuple[int, int, Dict[int, int]]:
        dist_map = compute_isochrone_dists(G, origin_node, WALK_TIME_SECONDS)
        if not dist_map:
            return origin_node, 0, {ri: 0 for ri in origin_rest_indices}

        iso_nodes = list(dist_map.keys())
        w_score = len(iso_nodes)

        # bbox for candidates (min/max of reachable node coords)
        xs_ = []
        ys_ = []
        for n in iso_nodes:
            x_ = node_x.get(int(n), None)
            y_ = node_y.get(int(n), None)
            if x_ is not None and y_ is not None:
                if not (np.isnan(x_) or np.isnan(y_)):
                    xs_.append(x_)
                    ys_.append(y_)
        if not xs_:
            return origin_node, w_score, {ri: 0 for ri in origin_rest_indices}

        minx, maxx = min(xs_), max(xs_)
        miny, maxy = min(ys_), max(ys_)
        pad = float(TILE_BUFFER_M)
        bbox = box(minx - pad, miny - pad, maxx + pad, maxy + pad)

        cand_idx = list(rest_sindex.intersection(bbox.bounds))

        reachable_set = []
        for j in cand_idx:
            nj = int(rest_nodes[j])
            if nj in dist_map:
                reachable_set.append(j)

        if not reachable_set:
            return origin_node, w_score, {ri: 0 for ri in origin_rest_indices}

        comp_counts: Dict[int, int] = {}
        for i in origin_rest_indices:
            my_mask = cat_masks[i]
            if not my_mask:
                comp_counts[i] = 0
                continue
            cnt = 0
            for j in reachable_set:
                if j == i:
                    continue
                if (my_mask & cat_masks[j]) != 0:
                    cnt += 1
            comp_counts[i] = cnt

        return origin_node, w_score, comp_counts

    futures = []
    done_nodes = 0
    total_nodes = len(node_to_rest_idx)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for origin_node, idxs in node_to_rest_idx.items():
            futures.append(ex.submit(process_one_origin, int(origin_node), idxs))

        for fut in as_completed(futures):
            origin_node, w_score, comp_counts = fut.result()

            for ri in node_to_rest_idx.get(int(origin_node), []):
                walkability[ri] = int(w_score)

            for ri, c in comp_counts.items():
                competitor[int(ri)] = int(c)

            done_nodes += 1
            if done_nodes % 200 == 0 or done_nodes == total_nodes:
                print(f"  processed origin nodes {done_nodes}/{total_nodes}")

    # ✅ 先写回 df，但不落盘（最终统一落盘）
    df["walkability_score"] = walkability
    df["competitor_density"] = competitor

    t1 = time.time()
    print(f"\n总运行时间：{t1 - start_time:.2f} 秒")

    # =========================
    # Stage 2 (drive) —— 新增
    # =========================
    print("\n========== Stage 2: DRIVE ==========")

    Gd = build_drive_graph(
        place_poly_utm=place_poly_utm,
        target_crs=TARGET_CRS,
        graphml_path=LOCAL_DRIVE_GRAPHML_PATH
    )

    print("[S2-5] Add drive_time to edges (speed by maxspeed/highway)")
    add_drive_time_to_edges(Gd, DRIVE_DEFAULT_SPEED_KMH, DRIVE_FALLBACK_SPEED_KMH)

    print("[S2-6] Snap restaurants to nearest DRIVE nodes")
    # 注意：这里复用 rest_utm 几何，但 nearest_nodes 需要针对 DRIVE 图
    xs2 = rest_utm.geometry.x.to_numpy()
    ys2 = rest_utm.geometry.y.to_numpy()
    drive_rest_nodes = np.asarray(ox.distance.nearest_nodes(Gd, X=xs2, Y=ys2), dtype=np.int64)

    # node -> restaurant indices (drive)
    drive_node_to_rest_idx: Dict[int, List[int]] = {}
    for idx, n in enumerate(drive_rest_nodes):
        drive_node_to_rest_idx.setdefault(int(n), []).append(idx)

    # 预取 drive node 坐标（UTM）
    drive_node_x: Dict[int, float] = {}
    drive_node_y: Dict[int, float] = {}
    for n, data in Gd.nodes(data=True):
        drive_node_x[int(n)] = float(data.get("x", np.nan))
        drive_node_y[int(n)] = float(data.get("y", np.nan))

    drive_score = np.zeros(len(df), dtype=np.float32)

    print("[S2-7] Compute drive_accessibility_score (15-min drive isochrone area, km^2) by unique origin nodes (parallel)")

    def process_one_drive_origin(origin_node: int, origin_rest_indices: List[int]) -> Tuple[int, float]:
        dist_map = compute_isochrone_dists_drive(Gd, origin_node, DRIVE_TIME_SECONDS)
        if not dist_map:
            return origin_node, 0.0

        iso_nodes = list(dist_map.keys())
        area_km2 = convex_hull_area_km2_from_nodes(iso_nodes, drive_node_x, drive_node_y)
        return origin_node, float(area_km2)

    futures2 = []
    done2 = 0
    total2 = len(drive_node_to_rest_idx)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for origin_node, idxs in drive_node_to_rest_idx.items():
            futures2.append(ex.submit(process_one_drive_origin, int(origin_node), idxs))

        for fut in as_completed(futures2):
            origin_node, area_km2 = fut.result()

            for ri in drive_node_to_rest_idx.get(int(origin_node), []):
                drive_score[ri] = float(area_km2)

            done2 += 1
            if done2 % 200 == 0 or done2 == total2:
                print(f"  processed drive origin nodes {done2}/{total2}")

    df["drive_accessibility_score"] = drive_score

    # =========================
    # Final Output (single file)
    # =========================
    print("\n========== Final Output ==========")
    print("[F] Save output (JSON Lines) with ALL fields")
    df.to_json(
        OUTPUT_json,
        orient="records",
        lines=True,
        force_ascii=False
    )
    print(f"[OK] wrote: {OUTPUT_json} (rows={len(df)})")

    t2 = time.time()
    print(f"\n总运行时间：{t2 - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
