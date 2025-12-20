# -*- coding: utf-8 -*-
import os
import time
import warnings
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed  # ✅ 仅新增 ProcessPoolExecutor

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, MultiPoint, LineString, MultiLineString
from shapely.ops import unary_union

import networkx as nx
import osmnx as ox

warnings.filterwarnings("ignore")


# =========================
# User Config
# =========================
INPUT_json = r"restaurant_features.json"   # TODO: 改成你的输入文件（JSON Lines）
# ✅ 最终只输出一个文件（Stage1 + Stage2 + Stage3 全部字段）
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
LOCAL_WALK_GRAPHML_PATH = os.path.join(LOCAL_GRAPH_DIR, "philadelphia_walk_clipped_utm18_slim.graphml")

# ===== Stage2: Drive config =====
DRIVE_TIME_SECONDS = 900  # 15 min = 900s
LOCAL_DRIVE_GRAPHML_PATH = os.path.join(LOCAL_GRAPH_DIR, "philadelphia_drive_clipped_utm18_slim.graphml")

# ✅ South Broad Street sidecar（保存 DRIVE 图时自动生成；Stage3 会优先读取，缺失则自动构建）
SOUTH_BROAD_DRIVE_NODES_PATH = os.path.join(LOCAL_GRAPH_DIR, "south_broad_drive_nodes.json")

# ===== Stage3: Centrality & Strip Distance config =====
# ✅ 预留参数：后续你可把 BC_K 改成 500/1000 等做近似对比；None 表示全节点精确
BC_K: Optional[int] = 1000
BC_SEED = 42  # 仅在 BC_K 不为 None 时用于可复现采样（NetworkX 支持 seed）

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


# =========================
# GraphML 本地加载/保存 —— ✅ 仅修改这一块逻辑
# =========================
def _graph_has_edge_attr(G: nx.MultiDiGraph, attr_name: str) -> bool:
    for _, _, _, data in G.edges(keys=True, data=True):
        if data.get(attr_name, None) is not None:
            return True
    return False


# ✅ 新增：把 GraphML 读回来的 edge 属性（可能是 str）强制转 float
def _coerce_edge_attr_to_float(G: nx.MultiDiGraph, attr: str):
    """
    GraphML 读回来的 edge attr 可能是 str，这里统一转成 float，避免 Dijkstra 出现 int/float + str。
    仅对存在且可解析的值处理；不可解析的置为 NaN。
    """
    if G is None:
        return
    for _, _, _, data in G.edges(keys=True, data=True):
        if attr not in data:
            continue
        v = data.get(attr, None)
        if v is None:
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            continue
        try:
            data[attr] = float(v)
        except Exception:
            data[attr] = np.nan


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

        # ✅ 关键修复：根据图类型把权重字段转回 float（避免 dijkstra 报 int+str）
        fname = os.path.basename(graphml_path).lower()
        if "walk" in fname:
            _coerce_edge_attr_to_float(G_local, "travel_time")
        if "drive" in fname:
            _coerce_edge_attr_to_float(G_local, "drive_time")
            _coerce_edge_attr_to_float(G_local, "length")

        print(f"[INFO] loaded local graph: nodes={len(G_local.nodes):,}, edges={len(G_local.edges):,}")
        return G_local
    return None


def _normalize_osm_name_list(name_attr) -> List[str]:
    if name_attr is None:
        return []
    if isinstance(name_attr, (list, tuple)):
        return [str(x) for x in name_attr if x is not None]
    return [str(name_attr)]


def _extract_south_broad_drive_nodes_if_possible(G: nx.MultiDiGraph, out_path: str):
    """
    只在图里仍有 'name' 字段时尝试提取 South Broad Street 相关节点集合并落盘。
    """
    if not out_path:
        return
    if os.path.exists(out_path):
        return

    nodes_set = set()
    hit_edges = 0

    # 必须包含 broad；prefer south（但允许 Broad St 没写 south）
    keys_must = ["broad"]
    keys_prefer = ["south", "s "]

    for u, v, k, data in G.edges(keys=True, data=True):
        names = _normalize_osm_name_list(data.get("name", None))
        if not names:
            continue
        matched = False
        for nm in names:
            s = nm.strip().lower()
            if not s:
                continue
            if all(kw in s for kw in keys_must):
                if any(kw in s for kw in keys_prefer):
                    matched = True
                else:
                    matched = True
            if matched:
                break

        if matched:
            nodes_set.add(int(u))
            nodes_set.add(int(v))
            hit_edges += 1

    if nodes_set:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        import json
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sorted(nodes_set), f, ensure_ascii=False)
        print(f"[INFO] saved South Broad drive nodes sidecar: {out_path} (nodes={len(nodes_set)}, edges_hit={hit_edges})")
    else:
        print("[WARN] South Broad nodes sidecar not created (no matching edges found).")


def _shrink_graph_copy(G: nx.MultiDiGraph,
                       keep_node_attrs: set,
                       keep_edge_attrs: set,
                       keep_graph_attrs: set) -> nx.MultiDiGraph:
    """
    生成一个“瘦身副本”用于保存，避免污染内存中的原图（保证其它功能不变）。
    """
    Gs = nx.MultiDiGraph()
    # graph attrs
    Gs.graph = {k: v for k, v in (G.graph or {}).items() if k in keep_graph_attrs}

    # nodes
    for n, data in G.nodes(data=True):
        Gs.add_node(n, **{k: v for k, v in (data or {}).items() if k in keep_node_attrs})

    # edges
    for u, v, k, data in G.edges(keys=True, data=True):
        Gs.add_edge(u, v, key=k, **{kk: vv for kk, vv in (data or {}).items() if kk in keep_edge_attrs})

    return Gs


def save_graph_local(G: nx.MultiDiGraph, graphml_path: str):
    """
    ✅ 按我们讨论结果保存：
      - WALK：只保存 travel_time（不保存 length），避免下次重复计算
      - DRIVE：保存 drive_time + length（为后续 distance_to_strip），并删掉 highway/maxspeed/name/geometry/osmid 等无用字段
      - 保存 DRIVE 时若还存在 name，则顺便导出 South Broad nodes sidecar，后续阶段3不依赖 name
    """
    if not graphml_path:
        return
    os.makedirs(os.path.dirname(graphml_path), exist_ok=True)

    fname = os.path.basename(graphml_path).lower()
    is_walk = "walk" in fname
    is_drive = "drive" in fname

    # graph-level
    keep_graph_attrs = {"crs"}

    if is_walk:
        # 只有 travel_time + x/y
        keep_node_attrs = {"x", "y"}
        keep_edge_attrs = {"travel_time"}  # ✅ 不保存 length
        Gs = _shrink_graph_copy(G, keep_node_attrs, keep_edge_attrs, keep_graph_attrs)

    elif is_drive:
        # 保存 drive_time + length + x/y
        _extract_south_broad_drive_nodes_if_possible(G, SOUTH_BROAD_DRIVE_NODES_PATH)

        keep_node_attrs = {"x", "y"}
        keep_edge_attrs = {"drive_time", "length"}  # ✅ drive 必须保留 length
        Gs = _shrink_graph_copy(G, keep_node_attrs, keep_edge_attrs, keep_graph_attrs)

    else:
        # 兜底：不做瘦身
        Gs = G

    ox.save_graphml(Gs, filepath=graphml_path)
    print(f"[INFO] saved local graph (shrunk): {graphml_path}")


# =========================
# Stage2 NEW Helpers (Drive)
# =========================
def _pick_highway_value(highway_attr) -> Optional[str]:
    if highway_attr is None:
        return None
    if isinstance(highway_attr, (list, tuple)) and len(highway_attr) > 0:
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
        if any(tok in s2 for tok in ["signals", "variable", "none", "walk", "national", "urban", "rural"]):
            continue

        import re
        nums = re.findall(r"(\d+(\.\d+)?)", s2)
        if not nums:
            continue
        v = float(nums[0][0])

        if "mph" in s2:
            v = v * 1.609344
        parsed_vals.append(v)

    if not parsed_vals:
        return None
    return float(min(parsed_vals))


def add_drive_time_to_edges(G: nx.MultiDiGraph,
                            default_speed_map_kmh: Dict[str, float],
                            fallback_speed_kmh: float):
    """
    为 drive 图的每条边添加 drive_time（秒），权重字段名：drive_time
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        length_m = data.get("length", None)
        if length_m is None:
            data["drive_time"] = np.nan
            continue

        ms = _parse_maxspeed_to_kmh(data.get("maxspeed", None))
        if ms is None or ms <= 0:
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

    if area_m2 <= 0:
        area_m2 = float(MultiPoint(pts).buffer(1.0).area)

    return area_m2 / 1e6


def build_drive_graph(place_poly_utm,
                      target_crs: str,
                      graphml_path: str) -> nx.MultiDiGraph:
    """
    Stage2: drive 路网构建逻辑（tile下载 + merge + project + clip）
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

    return Gd


# =========================
# ✅ Stage2 进程池 worker（仅新增，不影响 Stage1）
# =========================
_D_G = None
_D_NODE_X = None
_D_NODE_Y = None
_D_CUTOFF = None


def _init_drive_worker(graphml_path: str,
                       cutoff_s: int,
                       default_speed_map: Dict[str, float],
                       fallback_speed_kmh: float):
    """
    每个进程启动时初始化一次 DRIVE 图、node坐标
    ✅ 若图里已带 drive_time（我们保存时会带），则不再重复计算 drive_time
    """
    global _D_G, _D_NODE_X, _D_NODE_Y, _D_CUTOFF

    _D_CUTOFF = int(cutoff_s)

    _D_G = ox.load_graphml(graphml_path)
    # GraphML 节点 id 可能是 str，尽量转回 int
    try:
        any_node = next(iter(_D_G.nodes))
        if isinstance(any_node, str) and any_node.isdigit():
            mapping = {n: int(n) for n in _D_G.nodes if isinstance(n, str) and n.isdigit()}
            if len(mapping) == len(_D_G.nodes):
                _D_G = nx.relabel_nodes(_D_G, mapping, copy=True)
    except Exception:
        pass

    # ✅ 关键修复：worker 内再次保证权重字段是 float（防止 GraphML 读回是 str）
    _coerce_edge_attr_to_float(_D_G, "drive_time")
    _coerce_edge_attr_to_float(_D_G, "length")

    # ✅ 如果缺 drive_time 才计算（兼容旧图）
    if not _graph_has_edge_attr(_D_G, "drive_time"):
        add_drive_time_to_edges(_D_G, default_speed_map, fallback_speed_kmh)
        # 刚算完也确保是 float
        _coerce_edge_attr_to_float(_D_G, "drive_time")

    _D_NODE_X = {}
    _D_NODE_Y = {}
    for n, data in _D_G.nodes(data=True):
        _D_NODE_X[int(n)] = float(data.get("x", np.nan))
        _D_NODE_Y[int(n)] = float(data.get("y", np.nan))


def _drive_task(origin_node: int) -> Tuple[int, float]:
    global _D_G, _D_NODE_X, _D_NODE_Y, _D_CUTOFF

    dist_map = nx.single_source_dijkstra_path_length(
        _D_G, origin_node, cutoff=_D_CUTOFF, weight="drive_time"
    )
    if not dist_map:
        return origin_node, 0.0

    iso_nodes = list(dist_map.keys())
    area_km2 = convex_hull_area_km2_from_nodes(iso_nodes, _D_NODE_X, _D_NODE_Y)
    return origin_node, float(area_km2)


# =========================
# Stage3 Helpers (NEW) —— ✅ 只新增，不改 Stage1/Stage2
# =========================
def _load_strip_nodes_sidecar(path: str) -> Optional[List[int]]:
    if not path or not os.path.exists(path):
        return None
    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        if not isinstance(arr, list):
            return None
        out = []
        for x in arr:
            try:
                out.append(int(x))
            except Exception:
                pass
        return out if out else None
    except Exception:
        return None


def _iter_lines_from_geometry(geom):
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            if isinstance(g, LineString) and not g.is_empty:
                yield g


def _sample_points_on_lines(lines: List[LineString], step_m: float = 30.0, max_points: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """
    在若干 LineString 上按间隔采样点，用于 nearest_edges。
    """
    xs = []
    ys = []
    for ln in lines:
        if ln is None or ln.is_empty:
            continue
        length = float(ln.length)
        if length <= 0:
            continue
        n = int(max(2, min(1 + length / step_m, 3000)))
        for i in range(n):
            if len(xs) >= max_points:
                return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
            d = (i / (n - 1)) * length
            p = ln.interpolate(d)
            xs.append(float(p.x))
            ys.append(float(p.y))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _build_strip_nodes_from_osm_features(Gd: nx.MultiDiGraph,
                                        place_name: str,
                                        target_crs: str,
                                        out_path: str) -> Optional[List[int]]:
    """
    当 sidecar 不存在时：从 OSM 直接抓取 South Broad Street 的线要素，
    然后在 Gd 上用 nearest_edges 把这些线映射到 Gd 的边，得到 strip nodes 集合并落盘。
    不依赖 Gd.edge['name']（因为你的 drive 图是 slim 的）。
    """
    print("[S3-2] South Broad sidecar not found, try building from OSM features (name=South Broad Street) ...")

    candidates = [
        {"name": "South Broad Street"},
        {"name": "S Broad St"},
        {"name": "Broad Street"},
        {"name": "Broad St"},
    ]

    gfeat = None
    for tags in candidates:
        try:
            # OSMnx 2.x: features_from_place
            gf = ox.features_from_place(place_name, tags=tags)
            if gf is not None and len(gf) > 0:
                # 只要线
                gf = gf[gf.geometry.type.isin(["LineString", "MultiLineString"])].copy()
                if len(gf) > 0:
                    gfeat = gf
                    print(f"[S3-2] OSM features hit with tags={tags}, rows={len(gf)}")
                    break
        except Exception:
            continue

    if gfeat is None or len(gfeat) == 0:
        print("[WARN] Failed to fetch South Broad features from OSM (features_from_place).")
        return None

    try:
        gfeat = gfeat.to_crs(target_crs)
    except Exception:
        # 如果 features CRS 不可用，强制认为是 WGS84 再投影
        gfeat = gfeat.set_crs("EPSG:4326", allow_override=True).to_crs(target_crs)

    # 收集所有线
    lines = []
    for geom in gfeat.geometry.values:
        for ln in _iter_lines_from_geometry(geom):
            lines.append(ln)

    if not lines:
        print("[WARN] South Broad features contain no LineString geometries.")
        return None

    xs, ys = _sample_points_on_lines(lines, step_m=30.0, max_points=20000)
    if xs.size == 0:
        print("[WARN] No sample points generated for South Broad geometries.")
        return None

    # 用 sampled points 贴到 drive 图的边
    try:
        us, vs, ks = ox.distance.nearest_edges(Gd, X=xs, Y=ys)
    except Exception as e:
        print(f"[WARN] nearest_edges failed: {e}")
        return None

    strip_nodes = set()
    # nearest_edges 返回的 u/v/k 可能是标量或数组
    if np.isscalar(us):
        strip_nodes.add(int(us))
        strip_nodes.add(int(vs))
    else:
        for u, v in zip(us, vs):
            try:
                strip_nodes.add(int(u))
                strip_nodes.add(int(v))
            except Exception:
                pass

    if not strip_nodes:
        print("[WARN] Could not map South Broad features onto DRIVE graph edges.")
        return None

    # 落盘 sidecar
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        import json
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sorted(strip_nodes), f, ensure_ascii=False)
        print(f"[INFO] saved South Broad drive nodes sidecar (rebuilt): {out_path} (nodes={len(strip_nodes)})")
    except Exception:
        pass

    return sorted(strip_nodes)


def _compute_betweenness_centrality_drive(Gd: nx.MultiDiGraph,
                                         weight: str,
                                         k: Optional[int],
                                         seed: int,
                                         normalized: bool) -> Dict[int, float]:
    """
    按你的要求：
      - drive 图
      - weight=length
      - directed（原图）
      - normalized=False
      - k 参数预留：None => 全节点精确；否则采样近似
    """
    print("[S3-1] Compute betweenness_centrality on DRIVE graph (directed, weight=length, normalized=False)")
    # NetworkX 在 k=None 时不会用到 seed；k!=None 才会用到 seed
    try:
        bc = nx.betweenness_centrality(Gd, k=k, weight=weight, normalized=normalized, seed=seed)
    except TypeError:
        # 兼容旧 networkx 版本：可能不支持 seed 参数
        bc = nx.betweenness_centrality(Gd, k=k, weight=weight, normalized=normalized)
    # 强制 key 为 int
    out = {}
    for n, v in bc.items():
        try:
            out[int(n)] = float(v)
        except Exception:
            pass
    return out


def _compute_distance_to_strip_drive(Gd: nx.MultiDiGraph,
                                     strip_nodes: List[int],
                                     weight: str) -> Dict[int, float]:
    """
    多源 Dijkstra：一次得到全图每个节点到 strip_nodes 最近距离（按 length）。
    """
    print("[S3-3] Multi-source Dijkstra for distance_to_strip on DRIVE graph (weight=length)")
    dist = nx.multi_source_dijkstra_path_length(Gd, sources=set(strip_nodes), weight=weight)
    out = {}
    for n, v in dist.items():
        try:
            out[int(n)] = float(v)
        except Exception:
            pass
    return out


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
    # Stage 1 (walk) —— ✅ 保持线程并行不变
    # =========================
    print("\n========== Stage 1: WALK ==========")

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

    else:
        print("[INFO] using local WALK graph directly (skip download/merge/project/clip)")

    if not _graph_has_edge_attr(G, "travel_time"):
        print("[6] Add walking travel_time to edges")
        add_walk_time_to_edges(G, WALK_SPEED_KMH)
    else:
        print("[6] travel_time already exists in local WALK graph, skip recompute")

    if not os.path.exists(LOCAL_WALK_GRAPHML_PATH):
        save_graph_local(G, LOCAL_WALK_GRAPHML_PATH)

    rest_utm = rest_gdf.to_crs(TARGET_CRS)

    print("[7] Snap restaurants to nearest network nodes")
    xs = rest_utm.geometry.x.to_numpy()
    ys = rest_utm.geometry.y.to_numpy()
    rest_nodes = np.asarray(ox.distance.nearest_nodes(G, X=xs, Y=ys), dtype=np.int64)

    rest_utm["__node__"] = rest_nodes

    node_to_rest_idx: Dict[int, List[int]] = {}
    for idx, n in enumerate(rest_nodes):
        node_to_rest_idx.setdefault(int(n), []).append(idx)

    node_x: Dict[int, float] = {}
    node_y: Dict[int, float] = {}
    for n, data in G.nodes(data=True):
        node_x[int(n)] = float(data.get("x", np.nan))
        node_y[int(n)] = float(data.get("y", np.nan))

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

    df["walkability_score"] = walkability
    df["competitor_density"] = competitor

    t1 = time.time()
    print(f"\n阶段一总运行时间：{t1 - start_time:.2f} 秒")


    # =========================
    # Stage 2 (drive) —— ✅ 改为进程并行
    # =========================
    print("\n========== Stage 2: DRIVE ==========")

    Gd = build_drive_graph(
        place_poly_utm=place_poly_utm,
        target_crs=TARGET_CRS,
        graphml_path=LOCAL_DRIVE_GRAPHML_PATH
    )

    # ✅ 主进程也做一次防护（如果是本地读回，类型可能是 str）
    _coerce_edge_attr_to_float(Gd, "drive_time")
    _coerce_edge_attr_to_float(Gd, "length")

    if not _graph_has_edge_attr(Gd, "drive_time"):
        print("[S2-5] Add drive_time to edges (speed by maxspeed/highway)")
        add_drive_time_to_edges(Gd, DRIVE_DEFAULT_SPEED_KMH, DRIVE_FALLBACK_SPEED_KMH)
    else:
        print("[S2-5] drive_time already exists in local DRIVE graph, skip recompute")

    if not os.path.exists(LOCAL_DRIVE_GRAPHML_PATH):
        save_graph_local(Gd, LOCAL_DRIVE_GRAPHML_PATH)

    print("[S2-6] Snap restaurants to nearest DRIVE nodes")
    xs2 = rest_utm.geometry.x.to_numpy()
    ys2 = rest_utm.geometry.y.to_numpy()
    drive_rest_nodes = np.asarray(ox.distance.nearest_nodes(Gd, X=xs2, Y=ys2), dtype=np.int64)

    drive_node_to_rest_idx: Dict[int, List[int]] = {}
    for idx, n in enumerate(drive_rest_nodes):
        drive_node_to_rest_idx.setdefault(int(n), []).append(idx)

    drive_score = np.zeros(len(df), dtype=np.float32)

    print("[S2-7] Compute drive_accessibility_score (15-min drive isochrone area, km^2) by unique origin nodes (parallel)")

    futures2 = []
    done2 = 0
    total2 = len(drive_node_to_rest_idx)

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_init_drive_worker,
        initargs=(
            LOCAL_DRIVE_GRAPHML_PATH,
            DRIVE_TIME_SECONDS,
            DRIVE_DEFAULT_SPEED_KMH,
            DRIVE_FALLBACK_SPEED_KMH
        ),
    ) as ex:
        for origin_node in drive_node_to_rest_idx.keys():
            futures2.append(ex.submit(_drive_task, int(origin_node)))

        for fut in as_completed(futures2):
            origin_node, area_km2 = fut.result()

            for ri in drive_node_to_rest_idx.get(int(origin_node), []):
                drive_score[ri] = float(area_km2)

            done2 += 1
            if done2 % 200 == 0 or done2 == total2:
                print(f"  processed drive origin nodes {done2}/{total2}")

    df["drive_accessibility_score"] = drive_score

    t2 = time.time()
    print(f"\n阶段二总运行时间：{t2 - t1:.2f} 秒")


    # =========================
    # Stage 3 (NEW) —— ✅ 只新增，不改 Stage1/Stage2
    # =========================
    print("\n========== Stage 3: CENTRALITY & STRIP DIST ==========")

    # [S3-1] Betweenness centrality on DRIVE graph
    # 按你的要求：directed + weight=length + normalized=False + k 预留（默认 None 全节点精确）
    bc_map = _compute_betweenness_centrality_drive(
        Gd, weight="length", k=BC_K, seed=BC_SEED, normalized=False
    )

    # [S3-2] South Broad Street nodes sidecar
    strip_nodes = _load_strip_nodes_sidecar(SOUTH_BROAD_DRIVE_NODES_PATH)
    if strip_nodes is None:
        strip_nodes = _build_strip_nodes_from_osm_features(
            Gd=Gd,
            place_name=PLACE_NAME,
            target_crs=TARGET_CRS,
            out_path=SOUTH_BROAD_DRIVE_NODES_PATH
        )

    # [S3-3] Multi-source dijkstra distance to strip (length)
    dist_to_strip = {}
    if strip_nodes is None or len(strip_nodes) == 0:
        print("[WARN] South Broad strip nodes not available; distance_to_strip will be NaN.")
    else:
        dist_to_strip = _compute_distance_to_strip_drive(Gd, strip_nodes=strip_nodes, weight="length")

    # [S3-4] Map to restaurants by snapped DRIVE node (unique node reuse)
    betweenness_arr = np.full(len(df), np.nan, dtype=np.float64)
    dist_strip_arr = np.full(len(df), np.nan, dtype=np.float64)

    # unique node reuse: 同一 drive node 下的餐厅复用同一个值
    unique_nodes = list(drive_node_to_rest_idx.keys())
    for n in unique_nodes:
        n_int = int(n)
        bc_val = bc_map.get(n_int, 0.0)  # 没找到则按 0
        d_val = dist_to_strip.get(n_int, np.nan)  # 不可达则 NaN

        for ri in drive_node_to_rest_idx.get(n_int, []):
            betweenness_arr[int(ri)] = float(bc_val)
            dist_strip_arr[int(ri)] = float(d_val) if not (d_val is None or np.isnan(d_val)) else np.nan

    df["betweenness_centrality"] = betweenness_arr
    df["distance_to_strip"] = dist_strip_arr

    t3 = time.time()
    print(f"\n阶段三总运行时间：{t3 - t2:.2f} 秒")

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

    t = time.time()
    print(f"\n总运行时间：{t - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
