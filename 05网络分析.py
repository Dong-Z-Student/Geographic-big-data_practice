# -*- coding: utf-8 -*-
import os
import time
import warnings
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, MultiPoint, LineString, MultiLineString
from shapely.ops import unary_union

import networkx as nx
import osmnx as ox

warnings.filterwarnings("ignore")


# =========================
# 基础配置
# =========================
# ====文件路径====
INPUT_json = r"task3/情感极性及主题建模后数据.json"
OUTPUT_json = r"task5/全特征变量数据.json"

# ====研究区域====
PLACE_NAME = "Philadelphia, Pennsylvania, USA"
# ====投影：UTM 18N====
TARGET_CRS = "EPSG:32618"

# ====OSMnx参数====
# 缓存与请求设置
OSMNX_CACHE_DIR = "./cache"
REQUEST_TIMEOUT = 180
RETRIES = 5
BACKOFF_SECONDS = 4
# 可选节点
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

# ====stage1：步行参数====
WALK_SPEED_KMH = 4.8
WALK_TIME_SECONDS = 600
# 分块大小（单位：米，基于 UTM 投影后）
TILE_SIZE_M = 2500                   # 2.5km 网格
# tile 与 polygon 的缓冲（米），避免边缘断裂导致路网不连
TILE_BUFFER_M = 50
# 并行线程数
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)
# 本地化路网（最终处理后的 G：已投影 + 已裁剪）
LOCAL_GRAPH_DIR = "./local_graph"
LOCAL_WALK_GRAPHML_PATH = os.path.join(LOCAL_GRAPH_DIR, "philadelphia_walk_clipped_utm18_slim.graphml")

# ====stage2：驾车参数====
DRIVE_TIME_SECONDS = 900  # 15 min = 900s
LOCAL_DRIVE_GRAPHML_PATH = os.path.join(LOCAL_GRAPH_DIR, "philadelphia_drive_clipped_utm18_slim.graphml")
# 主干道South Broad Street sidecar（保存 DRIVE 图时自动生成；Stage3 会优先读取，缺失则自动构建）
SOUTH_BROAD_DRIVE_NODES_PATH = os.path.join(LOCAL_GRAPH_DIR, "south_broad_drive_nodes.json")
# 道路限速（km/h）
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
DRIVE_FALLBACK_SPEED_KMH = 30

# =====stage3：中心性&到主干道距离=====
# 预留参数：None 表示全节点计算中心性
BC_K: Optional[int] = 1000
BC_SEED = 42


# =========================
# GraphML 本地加载/保存
# =========================
def _graph_has_edge_attr(G: nx.MultiDiGraph, attr_name: str) -> bool:
    for _, _, _, data in G.edges(keys=True, data=True):
        if data.get(attr_name, None) is not None:
            return True
    return False

def _coerce_edge_attr_to_float(G: nx.MultiDiGraph, attr: str):
    """
    GraphML 读回来的 edge attr 统一转成 float，不可解析的置为 NaN。
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
        print(f"[INFO] 找到本地化路网, 加载: {graphml_path}")
        G_local = ox.load_graphml(graphml_path)

        try:
            any_node = next(iter(G_local.nodes))
            if isinstance(any_node, str) and any_node.isdigit():
                mapping = {n: int(n) for n in G_local.nodes if isinstance(n, str) and n.isdigit()}
                if len(mapping) == len(G_local.nodes):
                    G_local = nx.relabel_nodes(G_local, mapping, copy=True)
        except Exception:
            pass

        fname = os.path.basename(graphml_path).lower()
        if "walk" in fname:
            _coerce_edge_attr_to_float(G_local, "travel_time")
        if "drive" in fname:
            _coerce_edge_attr_to_float(G_local, "drive_time")
            _coerce_edge_attr_to_float(G_local, "length")

        print(f"[INFO] 加载了本地化路网: nodes={len(G_local.nodes):,}, edges={len(G_local.edges):,}")
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
    图里仍有 'name' 字段时提取 South Broad Street 相关节点集合并本地化
    """
    if not out_path:
        return
    if os.path.exists(out_path):
        return

    nodes_set = set()
    hit_edges = 0

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
        print(f"[INFO] 保存主干道South Broad驾车节点: {out_path} (nodes={len(nodes_set)}, edges_hit={hit_edges})")
    else:
        print("[WARN] 无法保存主干道South Broad驾车节点（没有匹配边）.")


def _shrink_graph_copy(G: nx.MultiDiGraph,
                       keep_node_attrs: set,
                       keep_edge_attrs: set,
                       keep_graph_attrs: set) -> nx.MultiDiGraph:
    """
    生成一个“瘦身副本”用于保存
    """
    Gs = nx.MultiDiGraph()
    Gs.graph = {k: v for k, v in (G.graph or {}).items() if k in keep_graph_attrs}

    for n, data in G.nodes(data=True):
        Gs.add_node(n, **{k: v for k, v in (data or {}).items() if k in keep_node_attrs})

    for u, v, k, data in G.edges(keys=True, data=True):
        Gs.add_edge(u, v, key=k, **{kk: vv for kk, vv in (data or {}).items() if kk in keep_edge_attrs})

    return Gs


def save_graph_local(G: nx.MultiDiGraph, graphml_path: str):
    """
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

    keep_graph_attrs = {"crs"}

    if is_walk:
        keep_node_attrs = {"x", "y"}
        keep_edge_attrs = {"travel_time"}
        Gs = _shrink_graph_copy(G, keep_node_attrs, keep_edge_attrs, keep_graph_attrs)

    elif is_drive:
        _extract_south_broad_drive_nodes_if_possible(G, SOUTH_BROAD_DRIVE_NODES_PATH)

        keep_node_attrs = {"x", "y"}
        keep_edge_attrs = {"drive_time", "length"}
        Gs = _shrink_graph_copy(G, keep_node_attrs, keep_edge_attrs, keep_graph_attrs)

    else:
        Gs = G

    ox.save_graphml(Gs, filepath=graphml_path)
    print(f"[INFO] 已保存本地路网（瘦身路网）: {graphml_path}")


# =========================
# 工具函数
# =========================
def configure_osmnx():
    os.makedirs(OSMNX_CACHE_DIR, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = OSMNX_CACHE_DIR
    ox.settings.log_console = True
    ox.settings.timeout = REQUEST_TIMEOUT


def get_place_polygon(place_name: str) -> gpd.GeoDataFrame:
    """
    从 Nominatim 获取 place polygon
    """
    gdf = ox.geocode_to_gdf(place_name)
    if gdf.empty:
        raise RuntimeError(f"编码目标区域失败: {place_name}")

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

    print(f"[INFO] 格网生成: {len(cleaned)}")
    return cleaned


def _is_no_nodes_polygon_error(e: Exception) -> bool:
    msg = str(e) if e is not None else ""
    return "多边形内无节点" in msg


def try_download_graph_from_polygon(poly_wgs84, network_type="walk") -> nx.MultiDiGraph:
    """
    带重试 + 切换 overpass endpoint 的下载函数
    - 若遇到“多边形内无节点”，直接返回 None（不重试，不换endpoint）
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
                if _is_no_nodes_polygon_error(e):
                    print(f"[WARN] 格网内无节点直接跳过: {e}")
                    return None

                last_err = e
                wait = BACKOFF_SECONDS * (k + 1)
                print(f"[WARN] 下载失败 (endpoint={ep}, retry={k+1}/{RETRIES}): {e}")
                time.sleep(wait)

    raise RuntimeError(f"所有重试均失败 Last error: {last_err}")


def merge_graphs(graphs: List[nx.MultiDiGraph]) -> nx.MultiDiGraph:
    graphs = [g for g in graphs if g is not None and len(g) > 0]
    if not graphs:
        raise RuntimeError("没有图可合并")
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
# stage1 工具函数
# =========================
def build_walk_graph(place_poly_utm,
                     target_crs: str,
                     graphml_path: str) -> nx.MultiDiGraph:
    """
    walk 路网构建逻辑（tile下载 + merge + project + clip）
    若本地存在 graphml_path，则直接 load
    """
    Gw = load_graph_if_exists(graphml_path)
    if Gw is not None:
        print("[INFO] 直接使用本地步行路网")
        return Gw

    print("[S1-1] 生成格网")
    tiles_utm = polygon_to_tiles(place_poly_utm, TILE_SIZE_M, TILE_BUFFER_M)

    print("[S1-2] 通过格网下载路网并合并")
    graphs = []
    for i, tile_utm in enumerate(tiles_utm, start=1):
        print(f"  - tile {i}/{len(tiles_utm)}: download walk network")
        tile_wgs = gpd.GeoSeries([tile_utm], crs=target_crs).to_crs("EPSG:4326").iloc[0]
        try:
            Gi = try_download_graph_from_polygon(tile_wgs, network_type="walk")
            if Gi is None or len(Gi.nodes) == 0:
                print(f"    [WARN] 跳过空图")
                continue
            graphs.append(Gi)
            print(f"    nodes={len(Gi.nodes):,}, edges={len(Gi.edges):,}")
        except Exception as e:
            print(f"    [ERROR] 跳过失败格网: {e}")

    Gw = merge_graphs(graphs)
    print(f"[INFO] 合并路网完成: nodes={len(Gw.nodes):,}, edges={len(Gw.edges):,}")

    print("[S1-3] 投影路网到UTM 18N")
    Gw = ox.project_graph(Gw, to_crs=target_crs)

    print("[S1-4] 裁剪路网到目标区域边界")
    Gw = ox.truncate.truncate_graph_polygon(Gw, place_poly_utm, truncate_by_edge=True)
    print(f"[INFO] 裁剪路网完成: nodes={len(Gw.nodes):,}, edges={len(Gw.edges):,}")

    return Gw



# =========================
# stage2 工具函数
# =========================
def _pick_highway_value(highway_attr) -> Optional[str]:
    if highway_attr is None:
        return None
    if isinstance(highway_attr, (list, tuple)) and len(highway_attr) > 0:
        return str(highway_attr[0])
    return str(highway_attr)


def _parse_maxspeed_to_kmh(maxspeed_attr) -> Optional[float]:
    """
    解析 OSM maxspeed：
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
    drive 路网构建逻辑（tile下载 + merge + project + clip）
    若本地存在 graphml_path，则直接 load
    """
    Gd = load_graph_if_exists(graphml_path)
    if Gd is not None:
        print("[INFO] 直接使用本地驾车路网")
        return Gd

    print("[S2-1] 生成格网")
    tiles_utm = polygon_to_tiles(place_poly_utm, TILE_SIZE_M, TILE_BUFFER_M)

    print("[S2-2] 通过格网下载路网并合并")
    graphs = []
    for i, tile_utm in enumerate(tiles_utm, start=1):
        print(f"  - tile {i}/{len(tiles_utm)}: download drive network")
        tile_wgs = gpd.GeoSeries([tile_utm], crs=target_crs).to_crs("EPSG:4326").iloc[0]
        try:
            Gi = try_download_graph_from_polygon(tile_wgs, network_type="drive")
            if Gi is None or len(Gi.nodes) == 0:
                print(f"    [WARN] 跳过空图")
                continue
            graphs.append(Gi)
            print(f"    nodes={len(Gi.nodes):,}, edges={len(Gi.edges):,}")
        except Exception as e:
            print(f"    [ERROR] 跳过失败格网: {e}")

    Gd = merge_graphs(graphs)
    print(f"[INFO] 合并路网完成: nodes={len(Gd.nodes):,}, edges={len(Gd.edges):,}")

    print("[S2-3] 投影路网到UTM 18N")
    Gd = ox.project_graph(Gd, to_crs=target_crs)

    print("[S2-4] 裁剪路网到目标区域边界")
    Gd = ox.truncate.truncate_graph_polygon(Gd, place_poly_utm, truncate_by_edge=True)
    print(f"[INFO] 裁剪路网完成: nodes={len(Gd.nodes):,}, edges={len(Gd.edges):,}")

    return Gd


# =========================
# Stage2 进程池 worker
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
    若图里有 drive_time，则不再重复计算 drive_time
    """
    global _D_G, _D_NODE_X, _D_NODE_Y, _D_CUTOFF

    _D_CUTOFF = int(cutoff_s)

    _D_G = ox.load_graphml(graphml_path)
    try:
        any_node = next(iter(_D_G.nodes))
        if isinstance(any_node, str) and any_node.isdigit():
            mapping = {n: int(n) for n in _D_G.nodes if isinstance(n, str) and n.isdigit()}
            if len(mapping) == len(_D_G.nodes):
                _D_G = nx.relabel_nodes(_D_G, mapping, copy=True)
    except Exception:
        pass

    _coerce_edge_attr_to_float(_D_G, "drive_time")
    _coerce_edge_attr_to_float(_D_G, "length")

    # 缺 drive_time 会计算
    if not _graph_has_edge_attr(_D_G, "drive_time"):
        add_drive_time_to_edges(_D_G, default_speed_map, fallback_speed_kmh)
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
# Stage3 工具函数
# =========================
def _load_strip_nodes_sidecar(path: str) -> Optional[List[int]]:
    print("[S3-2] 获取主干道South Broad Stree的节点")
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
    print("[S3-2] 没有找到主干道South Broad, 重新从OSM中构建")

    candidates = [
        {"name": "South Broad Street"},
        {"name": "S Broad St"},
        {"name": "Broad Street"},
        {"name": "Broad St"},
    ]

    gfeat = None
    for tags in candidates:
        try:
            gf = ox.features_from_place(place_name, tags=tags)
            if gf is not None and len(gf) > 0:
                gf = gf[gf.geometry.type.isin(["LineString", "MultiLineString"])].copy()
                if len(gf) > 0:
                    gfeat = gf
                    print(f"[S3-2] OSM features hit with tags={tags}, rows={len(gf)}")
                    break
        except Exception:
            continue

    if gfeat is None or len(gfeat) == 0:
        print("[WARN] 无法从OSM中下载主干道South Broad")
        return None

    try:
        gfeat = gfeat.to_crs(target_crs)
    except Exception:
        gfeat = gfeat.set_crs("EPSG:4326", allow_override=True).to_crs(target_crs)

    lines = []
    for geom in gfeat.geometry.values:
        for ln in _iter_lines_from_geometry(geom):
            lines.append(ln)

    if not lines:
        print("[WARN] 主干道South Broad不是线几何")
        return None

    xs, ys = _sample_points_on_lines(lines, step_m=30.0, max_points=20000)
    if xs.size == 0:
        print("[WARN] 无法为主干道South Broad生成点")
        return None

    try:
        us, vs, ks = ox.distance.nearest_edges(Gd, X=xs, Y=ys)
    except Exception as e:
        print(f"[WARN] 找不到最近边: {e}")
        return None

    strip_nodes = set()
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
        print("[WARN] 无法将主干道South Broad的特征映射到驾车路网的边上。")
        return None

    # 本地化
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        import json
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(sorted(strip_nodes), f, ensure_ascii=False)
        print(f"[INFO] 保存主干道South Broad驾车路网节点: {out_path} (nodes={len(strip_nodes)})")
    except Exception:
        pass

    return sorted(strip_nodes)


def _compute_betweenness_centrality_drive(Gd: nx.MultiDiGraph,
                                         weight: str,
                                         k: Optional[int],
                                         seed: int,
                                         normalized: bool) -> Dict[int, float]:
    """
      - drive 图
      - weight=length
      - directed
      - normalized=False
      - k 参数预留：None => 全节点精确；否则采样近似
    """
    print("[S3-1] 在驾车路网上计算介数中心性(directed, weight=length, normalized=False)")
    # NetworkX 在 k=None 时不会用到 seed；k!=None 才会用到 seed
    try:
        bc = nx.betweenness_centrality(Gd, k=k, weight=weight, normalized=normalized, seed=seed)
    except TypeError:
        bc = nx.betweenness_centrality(Gd, k=k, weight=weight, normalized=normalized)
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
    多源 Dijkstra：一次得到全图每个节点到 strip_nodes 最近距离（按 length）
    """
    print("[S3-3] 在驾车路网上计算到主干道的最短距离(weight=length)")
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

    print("[1] 加载餐厅数据")
    df = pd.read_json(INPUT_json, lines=True, encoding="utf-8")

    required = ["business_id", "latitude", "longitude"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"输入数据缺少关键字段: {c}")

    cat_cols = [c for c in df.columns if c.startswith("cat__")]
    print(f"[INFO] 类别列: {len(cat_cols)}")

    cat_masks = build_cat_bitmask(df, cat_cols)

    rest_gdf = build_rest_gdf(df, crs="EPSG:4326")

    print("[2] 获取Philadelphia多边形")
    place_gdf = get_place_polygon(PLACE_NAME)
    place_utm = place_gdf.to_crs(TARGET_CRS)
    place_poly_utm = unary_union(place_utm.geometry.values)

    # =========================
    # Stage 1 线程并行
    # =========================
    print("\n========== Stage 1: 步行 ==========")
    Gw = build_walk_graph(
        place_poly_utm=place_poly_utm,
        target_crs=TARGET_CRS,
        graphml_path=LOCAL_WALK_GRAPHML_PATH
    )

    if not _graph_has_edge_attr(Gw, "travel_time"):
        print("[S1-5] 添加字段travel_time到边")
        add_walk_time_to_edges(Gw, WALK_SPEED_KMH)
    else:
        print("[S1-5] 字段travel_time已经存在")

    if not os.path.exists(LOCAL_WALK_GRAPHML_PATH):
        save_graph_local(Gw, LOCAL_WALK_GRAPHML_PATH)

    rest_utm = rest_gdf.to_crs(TARGET_CRS)

    print("[S1-6] 匹配餐厅到路网最近节点")
    xs = rest_utm.geometry.x.to_numpy()
    ys = rest_utm.geometry.y.to_numpy()
    rest_nodes = np.asarray(ox.distance.nearest_nodes(Gw, X=xs, Y=ys), dtype=np.int64)

    rest_utm["__node__"] = rest_nodes

    node_to_rest_idx: Dict[int, List[int]] = {}
    for idx, n in enumerate(rest_nodes):
        node_to_rest_idx.setdefault(int(n), []).append(idx)

    node_x: Dict[int, float] = {}
    node_y: Dict[int, float] = {}
    for n, data in Gw.nodes(data=True):
        node_x[int(n)] = float(data.get("x", np.nan))
        node_y[int(n)] = float(data.get("y", np.nan))

    rest_sindex = rest_utm.sindex

    walkability = np.zeros(len(rest_utm), dtype=np.int32)
    competitor = np.zeros(len(rest_utm), dtype=np.int32)

    print("[S1-7] 计算步行可达性评分（十分钟步行等时圈内节点数）和等时圈内竞争对手数量")
    def process_one_origin(origin_node: int, origin_rest_indices: List[int]) -> Tuple[int, int, Dict[int, int]]:
        dist_map = compute_isochrone_dists(Gw, origin_node, WALK_TIME_SECONDS)
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
                print(f"  已处理唯一节点： {done_nodes}/{total_nodes}")

    df["walkability_score"] = walkability
    df["competitor_density"] = competitor

    t1 = time.time()
    print(f"\n阶段一总运行时间：{t1 - start_time:.2f} 秒")


    # =========================
    # Stage 2 进程并行
    # =========================
    print("\n========== Stage 2: 驾车 ==========")
    Gd = build_drive_graph(
        place_poly_utm=place_poly_utm,
        target_crs=TARGET_CRS,
        graphml_path=LOCAL_DRIVE_GRAPHML_PATH
    )

    _coerce_edge_attr_to_float(Gd, "drive_time")
    _coerce_edge_attr_to_float(Gd, "length")

    if not _graph_has_edge_attr(Gd, "drive_time"):
        print("[S2-5] 添加字段drive_time到边")
        add_drive_time_to_edges(Gd, DRIVE_DEFAULT_SPEED_KMH, DRIVE_FALLBACK_SPEED_KMH)
    else:
        print("[S2-5] 字段drive_time已经存在")

    if not os.path.exists(LOCAL_DRIVE_GRAPHML_PATH):
        save_graph_local(Gd, LOCAL_DRIVE_GRAPHML_PATH)

    print("[S2-6] 匹配餐厅到路网最近节点")
    xs2 = rest_utm.geometry.x.to_numpy()
    ys2 = rest_utm.geometry.y.to_numpy()
    drive_rest_nodes = np.asarray(ox.distance.nearest_nodes(Gd, X=xs2, Y=ys2), dtype=np.int64)

    drive_node_to_rest_idx: Dict[int, List[int]] = {}
    for idx, n in enumerate(drive_rest_nodes):
        drive_node_to_rest_idx.setdefault(int(n), []).append(idx)

    drive_score = np.zeros(len(df), dtype=np.float32)

    print("[S2-7] 计算驾车可达性评分（十五分钟驾车等时圈面积）")
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
                print(f"  已处理唯一节点： {done2}/{total2}")

    df["drive_accessibility_score"] = drive_score

    t2 = time.time()
    print(f"\n阶段二总运行时间：{t2 - t1:.2f} 秒")


    # =========================
    # Stage 3
    # =========================
    print("\n========== Stage 3: 介数中心性&到主干道最短距离 ==========")
    #[S3-1] 在驾车路网上计算介数中心性
    bc_map = _compute_betweenness_centrality_drive(
        Gd, weight="length", k=BC_K, seed=BC_SEED, normalized=False
    )
    print(f"阶段三计算介数中心性运行时间：{time.time() - t2:.2f} 秒")

    #[S3-2] 获取主干道South Broad Stree的节点
    strip_nodes = _load_strip_nodes_sidecar(SOUTH_BROAD_DRIVE_NODES_PATH)
    if strip_nodes is None:
        strip_nodes = _build_strip_nodes_from_osm_features(
            Gd=Gd,
            place_name=PLACE_NAME,
            target_crs=TARGET_CRS,
            out_path=SOUTH_BROAD_DRIVE_NODES_PATH
        )

    #[S3-3] 计算到主干道South Broad Stree的最短距离
    dist_to_strip = {}
    if strip_nodes is None or len(strip_nodes) == 0:
        print("[WARN] 主干道South Broad strip 节点不可用,distance_to_strip字段将为NaN.")
    else:
        dist_to_strip = _compute_distance_to_strip_drive(Gd, strip_nodes=strip_nodes, weight="length")

    print("[S3-4] 匹配餐馆到最近驾车路网节点")
    betweenness_arr = np.full(len(df), np.nan, dtype=np.float64)
    dist_strip_arr = np.full(len(df), np.nan, dtype=np.float64)

    # 同一 drive node 下的餐厅复用同一个值
    unique_nodes = list(drive_node_to_rest_idx.keys())
    for n in unique_nodes:
        n_int = int(n)
        bc_val = bc_map.get(n_int, 0.0)
        d_val = dist_to_strip.get(n_int, np.nan)

        for ri in drive_node_to_rest_idx.get(n_int, []):
            betweenness_arr[int(ri)] = float(bc_val)
            dist_strip_arr[int(ri)] = float(d_val) if not (d_val is None or np.isnan(d_val)) else np.nan

    df["betweenness_centrality"] = betweenness_arr
    df["distance_to_strip"] = dist_strip_arr

    t3 = time.time()
    print(f"\n阶段三总运行时间：{t3 - t2:.2f} 秒")


    # =========================
    # Stage 4
    # =========================
    print("\n========== Stage 4: 缓冲区内其余餐厅平均情感极性 ==========")
    if "avg_sentiment_polarity" not in df.columns:
        raise ValueError("输入数据缺失关键字段")

    # 使用已经构建好的 UTM 餐厅点
    sentiment_vals = df["avg_sentiment_polarity"].to_numpy(dtype=float)
    coords_x = rest_utm.geometry.x.to_numpy(dtype=float)
    coords_y = rest_utm.geometry.y.to_numpy(dtype=float)
    # 空间索引
    sindex = rest_utm.sindex

    radius = 500.0  # meters
    radius_sq = radius * radius

    neigh_avg = np.full(len(df), np.nan, dtype=np.float32)

    for i in range(len(df)):
        xi = coords_x[i]
        yi = coords_y[i]

        # bbox 粗筛
        minx = xi - radius
        maxx = xi + radius
        miny = yi - radius
        maxy = yi + radius

        cand_idx = list(sindex.intersection((minx, miny, maxx, maxy)))
        if not cand_idx:
            continue

        vals = []

        for j in cand_idx:
            if j == i:
                continue

            dx = coords_x[j] - xi
            dy = coords_y[j] - yi
            if dx * dx + dy * dy <= radius_sq:
                v = sentiment_vals[j]
                if not np.isnan(v):
                    vals.append(v)

        if vals:
            neigh_avg[i] = float(np.mean(vals))

        if (i + 1) % 500 == 0 or (i + 1) == len(df):
            print(f"  已处理餐厅 {i+1}/{len(df)}")

    df["sentiment_neighborhood_avg"] = neigh_avg

    t4 = time.time()
    print(f"\n阶段四总运行时间：{t4 - t3:.2f} 秒")


    # =========================
    # 最终输出
    # =========================
    print("\n========== 最终输出 ==========")
    print("[Final] 保存输出文件")
    df.to_json(
        OUTPUT_json,
        orient="records",
        lines=True,
        force_ascii=False
    )
    print(f"[Done] 结果已保存为: {OUTPUT_json} (rows={len(df)})")

    print(f"\n总运行时间：{time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
