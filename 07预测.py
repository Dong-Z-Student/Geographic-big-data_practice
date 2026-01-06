import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import time
import warnings
import sys
from typing import List, Dict, Tuple, Optional

import geopandas as gpd
from shapely.geometry import box, MultiPoint, Point, Polygon
from shapely.ops import unary_union

import networkx as nx
import osmnx as ox
import chardet

import random
import numpy as np
import os

# =========================
# 设置随机种子
# =========================
def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_global_seed(42)


warnings.filterwarnings("ignore")

# =========================
# 配置参数
# =========================
INPUT_JSON = "restaurants_walk_drive_scores.json"  # 训练数据特征文件
VIRTUAL_CSV = "virtual_restaurants.csv"  # 虚拟餐厅CSV文件
OUTPUT_VIRTUAL_JSON = "virtual_restaurants_features.json"  # 虚拟餐厅特征输出
PREDICTION_OUTPUT = "virtual_predictions.csv"  # 预测结果输出

PLACE_NAME = "Philadelphia, Pennsylvania, USA"
TARGET_CRS = "EPSG:32618"
WALK_SPEED_KMH = 4.8
WALK_TIME_SECONDS = 600
DRIVE_TIME_SECONDS = 900

# 路径设置
OSMNX_CACHE_DIR = "./cache"
LOCAL_GRAPH_DIR = "./local_graph"
LOCAL_WALK_GRAPHML_PATH = os.path.join(LOCAL_GRAPH_DIR, "philadelphia_walk_clipped_utm18_slim.graphml")
LOCAL_DRIVE_GRAPHML_PATH = os.path.join(LOCAL_GRAPH_DIR, "philadelphia_drive_clipped_utm18_slim.graphml")

# 费城大致范围（用于坐标验证）
PHILADELPHIA_BOUNDS = {
    'min_lon': -75.28, 'max_lon': -74.96,
    'min_lat': 39.86, 'max_lat': 40.14
}


# =========================
# 调试工具
# =========================
def print_debug_info(title, data_dict):
    """打印调试信息"""
    print(f"\n{'=' * 50}")
    print(f"DEBUG: {title}")
    print('=' * 50)
    for key, value in data_dict.items():
        print(f"  {key}: {value}")


# =========================
# 数据验证和修复工具
# =========================
def fix_coordinate_columns(df):
    """修复坐标列：交换纬度和经度"""
    print("[INFO] 检查坐标列...")

    # 检查是否需要交换列
    need_swap = False

    # 检查前几行数据
    for i in range(min(5, len(df))):
        lon = df.iloc[i]['longitude'] if 'longitude' in df.columns else None
        lat = df.iloc[i]['latitude'] if 'latitude' in df.columns else None

        if lon is not None and lat is not None:
            # 判断是否需要交换：如果"经度"看起来像纬度（在30-50之间），"纬度"看起来像经度（绝对值大于50）
            if 30 <= lon <= 50 and abs(lat) > 50:
                print(f"[WARN] 第{i + 1}行：疑似坐标列颠倒 - 经度={lon}, 纬度={lat}")
                need_swap = True
                break

    if need_swap:
        print("[INFO] 检测到坐标列颠倒，正在交换...")
        # 交换列名
        if 'longitude' in df.columns and 'latitude' in df.columns:
            df = df.rename(columns={'longitude': 'latitude_temp', 'latitude': 'longitude_temp'})
            df = df.rename(columns={'latitude_temp': 'latitude', 'longitude_temp': 'longitude'})
            print("[INFO] 坐标列已交换")

    return df


def validate_coordinates(df, lat_col='latitude', lon_col='longitude'):
    """验证坐标是否在合理范围内"""
    print("[INFO] 验证坐标...")

    # 检查坐标是否在费城范围内
    valid_count = 0
    invalid_coords = []

    for idx, row in df.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]
        id_val = row.get('id', f'行{idx}')

        # 基本验证
        if pd.isna(lat) or pd.isna(lon):
            print(f"  [WARN] {id_val}: 坐标为空")
            invalid_coords.append(id_val)
            continue

        if not (-90 <= lat <= 90):
            print(f"  [WARN] {id_val}: 纬度超出范围: {lat}")
            invalid_coords.append(id_val)
            continue

        if not (-180 <= lon <= 180):
            print(f"  [WARN] {id_val}: 经度超出范围: {lon}")
            invalid_coords.append(id_val)
            continue

        # 费城范围验证
        if not (PHILADELPHIA_BOUNDS['min_lon'] <= lon <= PHILADELPHIA_BOUNDS['max_lon']):
            print(
                f"  [WARN] {id_val}: 经度不在费城范围内: {lon} (范围: {PHILADELPHIA_BOUNDS['min_lon']} 到 {PHILADELPHIA_BOUNDS['max_lon']})")
            invalid_coords.append(id_val)
            continue

        if not (PHILADELPHIA_BOUNDS['min_lat'] <= lat <= PHILADELPHIA_BOUNDS['max_lat']):
            print(
                f"  [WARN] {id_val}: 纬度不在费城范围内: {lat} (范围: {PHILADELPHIA_BOUNDS['min_lat']} 到 {PHILADELPHIA_BOUNDS['max_lat']})")
            invalid_coords.append(id_val)
            continue

        valid_count += 1

    print(f"[INFO] 有效坐标: {valid_count}/{len(df)}")
    if invalid_coords:
        print(f"[WARN] 无效坐标的餐厅ID: {invalid_coords}")

    return valid_count == len(df)


def map_virtual_type_to_categories(virtual_type, all_categories):
    """将虚拟餐厅的类型映射到现有餐厅的类别"""
    type_lower = str(virtual_type).lower().strip()

    # 常见类型映射
    type_mapping = {
        'french': ['french'],
        'japanese': ['japanese', 'sushi'],
        'mexican': ['mexican'],
        'thai': ['thai'],
        'burgers': ['burgers'],
        'italian': ['italian'],
        'chinese': ['chinese'],
        'american': ['american', 'american_traditional', 'american_new'],
        'pizza': ['pizza'],
        'sandwiches': ['sandwiches'],
        'bars': ['bars'],
        'seafood': ['seafood'],
        'coffee': ['coffee_&_tea'],
        'breakfast': ['breakfast_&_brunch'],
        'fast food': ['fast_food'],
        'salad': ['salad']
    }

    # 查找匹配的类别
    matched_categories = []

    # 首先检查是否有直接映射
    for key, values in type_mapping.items():
        if key in type_lower:
            for value in values:
                cat_name = f"cat__{value.replace(' ', '_')}"
                if cat_name in all_categories:
                    matched_categories.append(cat_name)

    # 如果没有找到直接映射，尝试模糊匹配
    if not matched_categories:
        for cat in all_categories:
            cat_clean = cat.replace('cat__', '').replace('_', ' ').lower()
            if type_lower in cat_clean or any(word in cat_clean for word in type_lower.split()):
                matched_categories.append(cat)

    # 如果还是没有找到，使用默认类别
    if not matched_categories and all_categories:
        matched_categories = [all_categories[0]]

    return matched_categories


def load_and_prepare_existing_data():
    """加载和准备现有餐厅数据"""
    print("\n[INFO] 加载现有餐厅数据...")

    try:
        # 加载特征文件
        rows = []
        with open(INPUT_JSON, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))

        df_existing = pd.DataFrame(rows)

        print_debug_info("现有数据基本信息", {
            "总行数": len(df_existing),
            "列数": len(df_existing.columns),
            "前20列": list(df_existing.columns)[:20]
        })

        # 检查关键字段
        key_fields = ['stars', 'latitude', 'longitude', 'price_range',
                      'walkability_score', 'competitor_density',
                      'drive_accessibility_score', 'avg_sentiment_polarity']

        field_analysis = {}
        for field in key_fields:
            if field in df_existing.columns:
                non_null = df_existing[field].notna().sum()
                unique = df_existing[field].nunique()
                field_analysis[field] = {
                    "存在": "是",
                    "非空数量": non_null,
                    "唯一值数量": unique,
                    "示例值": df_existing[field].iloc[0] if non_null > 0 else "空"
                }
            else:
                field_analysis[field] = {"存在": "否"}

        print_debug_info("关键字段分析", field_analysis)

        # 检查stars分布
        if 'stars' in df_existing.columns:
            stars_stats = df_existing['stars'].describe()
            print_debug_info("Stars分布统计", {
                "平均值": stars_stats['mean'],
                "标准差": stars_stats['std'],
                "最小值": stars_stats['min'],
                "25%分位数": stars_stats['25%'],
                "中位数": stars_stats['50%'],
                "75%分位数": stars_stats['75%'],
                "最大值": stars_stats['max'],
            })

        # 检查类别字段
        cat_cols = [c for c in df_existing.columns if c.startswith('cat__')]
        print_debug_info("类别字段", {
            "数量": len(cat_cols),
            "前10个": cat_cols[:10] if cat_cols else "无"
        })

        return df_existing, cat_cols

    except Exception as e:
        print(f"[ERROR] 加载现有数据失败: {e}")
        raise


def load_and_fix_virtual_restaurants():
    """加载并修复虚拟餐厅数据"""
    print("\n[INFO] 加载虚拟餐厅数据...")

    if not os.path.exists(VIRTUAL_CSV):
        print(f"[ERROR] 虚拟餐厅文件不存在: {VIRTUAL_CSV}")

        # 创建示例虚拟餐厅数据（使用正确的坐标）
        print("[INFO] 创建示例虚拟餐厅数据...")
        sample_data = {
            'id': ['VR001', 'VR002', 'VR003', 'VR004', 'VR005'],
            'longitude': [-75.1652, -75.1500, -75.1350, -75.1200, -75.1050],
            'latitude': [39.9526, 39.9600, 39.9450, 39.9300, 39.9150],
            'price_range': [1, 2, 3, 4, 2],
            'type': ['Italian', 'Japanese', 'Chinese', 'Mexican', 'American'],
            'remarks': ['示例餐厅1', '示例餐厅2', '示例餐厅3', '示例餐厅4', '示例餐厅5']
        }

        df_virtual = pd.DataFrame(sample_data)
        df_virtual.to_csv(VIRTUAL_CSV, index=False, encoding='utf-8')
        print(f"[INFO] 示例数据已保存到: {VIRTUAL_CSV}")
        return df_virtual

    # 加载CSV文件
    try:
        df_virtual = pd.read_csv(VIRTUAL_CSV, encoding='utf-8')
    except:
        try:
            df_virtual = pd.read_csv(VIRTUAL_CSV, encoding='gbk')
        except:
            df_virtual = pd.read_csv(VIRTUAL_CSV, encoding='latin1')

    print_debug_info("原始虚拟餐厅数据", {
        "餐厅数量": len(df_virtual),
        "列名": list(df_virtual.columns),
        "前几行数据": df_virtual.head().to_dict('records')
    })

    # 修复坐标列
    df_virtual = fix_coordinate_columns(df_virtual)

    # 验证坐标
    is_valid = validate_coordinates(df_virtual)

    if not is_valid:
        print("\n[WARN] 虚拟餐厅坐标存在问题")
        print("[INFO] 手动调整坐标列...")

        # 如果坐标验证失败，尝试手动交换
        if 'longitude' in df_virtual.columns and 'latitude' in df_virtual.columns:
            # 检查是否需要交换
            sample_lon = df_virtual.iloc[0]['longitude']
            sample_lat = df_virtual.iloc[0]['latitude']

            # 如果经度看起来像纬度，纬度看起来像经度，则交换
            if 30 <= sample_lon <= 50 and abs(sample_lat) > 50:
                print("[INFO] 手动交换坐标列...")
                temp = df_virtual['longitude'].copy()
                df_virtual['longitude'] = df_virtual['latitude']
                df_virtual['latitude'] = temp

                # 再次验证
                validate_coordinates(df_virtual)

    print_debug_info("修复后虚拟餐厅数据", {
        "前几行数据": df_virtual[['id', 'longitude', 'latitude', 'type']].head().to_dict('records')
    })

    return df_virtual


def calculate_features_for_virtual_restaurants(df_virtual, df_existing, cat_cols):
    """为虚拟餐厅计算特征"""
    print("\n[INFO] 计算虚拟餐厅特征...")

    # =========================
    # 加载路网图（与05网络分析.py相同的逻辑）
    # =========================
    print("[INFO] 加载路网图...")

    def _coerce_edge_attr_to_float(G, attr):
        """将边属性强制转换为float（修复GraphML读取问题）"""
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

    def load_graph_if_exists(graphml_path: str):
        """从本地加载图，如果存在"""
        if graphml_path and os.path.exists(graphml_path):
            print(f"[INFO] 加载本地图: {graphml_path}")
            G = ox.load_graphml(graphml_path)

            # 兼容：GraphML 可能把节点ID读成 str，这里尽量转回 int
            try:
                any_node = next(iter(G.nodes))
                if isinstance(any_node, str) and any_node.isdigit():
                    mapping = {n: int(n) for n in G.nodes if isinstance(n, str) and n.isdigit()}
                    if len(mapping) == len(G.nodes):
                        G = nx.relabel_nodes(G, mapping, copy=True)
            except Exception:
                pass

            # 关键修复：确保权重字段是float（防止GraphML读回是str）
            fname = os.path.basename(graphml_path).lower()
            if "walk" in fname:
                _coerce_edge_attr_to_float(G, "travel_time")
            if "drive" in fname:
                _coerce_edge_attr_to_float(G, "drive_time")
                _coerce_edge_attr_to_float(G, "length")

            print(f"[INFO] 加载完成: 节点={len(G.nodes):,}, 边={len(G.edges):,}")
            return G
        else:
            print(f"[ERROR] 图文件不存在: {graphml_path}")
            return None

    # 加载walk图
    G_walk = load_graph_if_exists(LOCAL_WALK_GRAPHML_PATH)
    if G_walk is None:
        print("[ERROR] Walk图不存在，请先运行05网络分析.py生成路网")
        return None

    # 加载drive图
    G_drive = load_graph_if_exists(LOCAL_DRIVE_GRAPHML_PATH)
    if G_drive is None:
        print("[ERROR] Drive图不存在，请先运行05网络分析.py生成路网")
        return None

    # 检查drive图是否有drive_time权重
    has_drive_time = False
    for _, _, _, data in G_drive.edges(keys=True, data=True):
        if "drive_time" in data and data["drive_time"] is not None:
            has_drive_time = True
            break

    if not has_drive_time:
        print("[WARN] Drive图缺少drive_time权重，尝试重新计算...")
        # 从05网络分析.py复制速度映射
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

        # 从05网络分析.py复制计算drive_time的函数
        def _pick_highway_value(highway_attr):
            if highway_attr is None:
                return None
            if isinstance(highway_attr, (list, tuple)) and len(highway_attr) > 0:
                return str(highway_attr[0])
            return str(highway_attr)

        def _parse_maxspeed_to_kmh(maxspeed_attr):
            """尝试解析 OSM maxspeed"""
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

        def add_drive_time_to_edges(G, default_speed_map_kmh, fallback_speed_kmh):
            """为drive图的每条边添加drive_time（秒）"""
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

        # 计算drive_time
        add_drive_time_to_edges(G_drive, DRIVE_DEFAULT_SPEED_KMH, DRIVE_FALLBACK_SPEED_KMH)
        print("[INFO] 已重新计算drive_time权重")

    # =========================
    # 转换坐标系统
    # =========================
    print("[INFO] 转换坐标系统...")

    # 虚拟餐厅的GeoDataFrame
    gdf_virtual = gpd.GeoDataFrame(
        df_virtual.copy(),
        geometry=gpd.points_from_xy(df_virtual["longitude"], df_virtual["latitude"]),
        crs="EPSG:4326"
    ).to_crs(TARGET_CRS)

    # 现有餐厅的GeoDataFrame（用于其他特征计算）
    gdf_existing = gpd.GeoDataFrame(
        df_existing.copy(),
        geometry=gpd.points_from_xy(df_existing["longitude"], df_existing["latitude"]),
        crs="EPSG:4326"
    ).to_crs(TARGET_CRS)

    features_list = []

    for i, virtual_row in gdf_virtual.iterrows():
        restaurant_id = virtual_row.get('id', f'VR{i:03d}')
        print(f"\n[INFO] 处理虚拟餐厅 {i + 1}/{len(gdf_virtual)}: {restaurant_id}")

        # 基础特征
        features = {
            'business_id': restaurant_id,
            'longitude': df_virtual.iloc[i]['longitude'],
            'latitude': df_virtual.iloc[i]['latitude'],
            'price_range': virtual_row.get('price_range', 2),
            'type': virtual_row.get('type', 'Restaurant'),
        }

        # =========================
        # 1. WALKABILITY_SCORE: 10分钟步行可达的节点数
        # =========================
        print("  计算walkability_score...")

        # 将虚拟餐厅snap到walk图的最近节点
        x_coord = virtual_row.geometry.x
        y_coord = virtual_row.geometry.y

        # 找到最近的walk节点
        try:
            nearest_walk_node = ox.distance.nearest_nodes(G_walk, x_coord, y_coord)
            print(f"    最近walk节点: {nearest_walk_node}")

            # 计算10分钟步行等时线（与05网络分析.py相同的逻辑）
            walk_dist_map = nx.single_source_dijkstra_path_length(
                G_walk, nearest_walk_node, cutoff=WALK_TIME_SECONDS, weight="travel_time"
            )

            # walkability_score = 等时线内可达的节点数
            walkability = len(walk_dist_map) if walk_dist_map else 0
            features['walkability_score'] = int(walkability)
            print(f"    walkability_score: {walkability}")

        except Exception as e:
            print(f"    [WARN] walkability计算失败: {e}")
            features['walkability_score'] = 0

        # =========================
        # 2. COMPETITOR_DENSITY: 10分钟步行范围内的竞争对手数量
        # =========================
        print("  计算competitor_density...")

        # 将虚拟餐厅类型映射到类别
        virtual_type = virtual_row.get('type', '')
        if pd.isna(virtual_type):
            virtual_type = ''

        matched_categories = map_virtual_type_to_categories(virtual_type, cat_cols)

        competitor_count = 0

        # 如果有walk等时线数据，计算等时线内的竞争对手
        if 'walkability_score' in features and features['walkability_score'] > 0:
            # 我们需要现有餐厅在walk图上的节点映射
            # 这里简化处理：使用空间查询找到在等时线缓冲区内的现有餐厅
            walk_speed_mps = (WALK_SPEED_KMH * 1000) / 3600  # 米/秒
            max_walk_distance = walk_speed_mps * WALK_TIME_SECONDS  # 最大步行距离

            # 创建等时线缓冲区
            walk_buffer = virtual_row.geometry.buffer(max_walk_distance)

            # 查询附近的现有餐厅
            possible_matches_index = list(gdf_existing.sindex.intersection(walk_buffer.bounds))
            possible_matches = gdf_existing.iloc[possible_matches_index]
            precise_matches = possible_matches[possible_matches.intersects(walk_buffer)]

            # 计算竞争对手
            for _, existing_row in precise_matches.iterrows():
                # 检查是否有匹配的类别
                for cat in matched_categories:
                    if cat in existing_row.index and existing_row[cat] == 1:
                        competitor_count += 1
                        break

        features['competitor_density'] = competitor_count
        print(f"    competitor_density: {competitor_count} (匹配类别: {matched_categories})")

        # =========================
        # 3. DRIVE_ACCESSIBILITY_SCORE: 15分钟驾驶可达的凸包面积
        # =========================
        print("  计算drive_accessibility_score...")

        # 将虚拟餐厅snap到drive图的最近节点
        try:
            nearest_drive_node = ox.distance.nearest_nodes(G_drive, x_coord, y_coord)
            print(f"    最近drive节点: {nearest_drive_node}")

            # 测试drive_time权重是否存在且有效
            sample_edges = list(G_drive.edges(data=True))[:5]
            print(f"    示例边权重: {[(u, v, data.get('drive_time', 'None')) for u, v, data in sample_edges]}")

            # 计算15分钟驾驶等时线
            try:
                drive_dist_map = nx.single_source_dijkstra_path_length(
                    G_drive, nearest_drive_node, cutoff=DRIVE_TIME_SECONDS, weight="drive_time"
                )
                print(f"    可达节点数: {len(drive_dist_map)}")

                if not drive_dist_map:
                    print("    [WARN] 无可达节点，可能是权重问题或节点孤立")
                    features['drive_accessibility_score'] = 0.0
                else:
                    # 从05网络分析.py复制的凸包面积计算函数
                    def convex_hull_area_km2_from_nodes(node_ids, node_x, node_y):
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

                        from shapely.geometry import MultiPoint
                        geom = MultiPoint(pts).convex_hull
                        area_m2 = float(geom.area)

                        if area_m2 <= 0:
                            geom_buffer = MultiPoint(pts).buffer(1.0)
                            area_m2 = float(geom_buffer.area) if not geom_buffer.is_empty else 0.0

                        return area_m2 / 1e6  # 转换为平方公里

                    # 获取drive图节点坐标
                    drive_node_x = {}
                    drive_node_y = {}
                    for n, data in G_drive.nodes(data=True):
                        drive_node_x[int(n)] = float(data.get("x", np.nan))
                        drive_node_y[int(n)] = float(data.get("y", np.nan))

                    # 计算凸包面积
                    iso_nodes = list(drive_dist_map.keys())
                    area_km2 = convex_hull_area_km2_from_nodes(iso_nodes, drive_node_x, drive_node_y)
                    features['drive_accessibility_score'] = float(area_km2)
                    print(f"    drive_accessibility_score: {area_km2:.4f} km²")

            except Exception as e:
                print(f"    [ERROR] Dijkstra计算失败: {e}")
                features['drive_accessibility_score'] = 0.0

        except Exception as e:
            print(f"    [ERROR] drive_accessibility计算失败: {e}")
            features['drive_accessibility_score'] = 0.0

        # =========================
        # 4. 计算情感邻域平均值
        # =========================
        print("  计算sentiment_neighborhood_avg...")

        sentiment_radius = 500  # 500米范围
        sentiment_buffer = virtual_row.geometry.buffer(sentiment_radius)

        possible_matches_index_sentiment = list(gdf_existing.sindex.intersection(sentiment_buffer.bounds))
        possible_matches_sentiment = gdf_existing.iloc[possible_matches_index_sentiment]
        precise_matches_sentiment = possible_matches_sentiment[possible_matches_sentiment.intersects(sentiment_buffer)]

        if len(precise_matches_sentiment) > 0 and 'avg_sentiment_polarity' in precise_matches_sentiment.columns:
            sentiments = precise_matches_sentiment['avg_sentiment_polarity'].dropna()
            if len(sentiments) > 0:
                avg_sentiment = float(sentiments.mean())
            else:
                avg_sentiment = float(df_existing['avg_sentiment_polarity'].mean())
        else:
            avg_sentiment = float(df_existing['avg_sentiment_polarity'].mean())

        features['sentiment_neighborhood_avg'] = avg_sentiment
        print(f"    sentiment_neighborhood_avg: {avg_sentiment:.4f}")

        # =========================
        # 5. 其他特征
        # =========================
        # 中心性：靠近市中心的餐厅中心性更高
        center_point = Point(-75.1652, 39.9526)  # 费城大致中心
        distance_to_center = virtual_row.geometry.distance(center_point)
        normalized_distance = min(1.0, distance_to_center / 10000)  # 10公里内归一化

        # 距离中心越近，中心性越高
        betweenness = (1 - normalized_distance) * 0.005 + np.random.uniform(0, 0.001)
        features['betweenness_centrality'] = float(betweenness)

        # 到主要街道的距离：随机但合理
        features['distance_to_strip'] = float(np.random.uniform(500, 5000))

        # =========================
        # 6. 添加类别特征
        # =========================
        for cat in cat_cols:
            if cat in matched_categories:
                features[cat] = 1
            else:
                features[cat] = 0

        # =========================
        # 7. 其他必要字段
        # =========================
        features['review_count'] = 0  # 虚拟餐厅无评论
        features['is_open'] = 1  # 假设开业
        features['stars'] = np.nan  # 虚拟餐厅没有真实评分

        features_list.append(features)

    return pd.DataFrame(features_list)


def train_model_and_predict(df_existing, df_virtual_features):
    """训练模型并进行预测"""
    print("\n" + "=" * 60)
    print("模型训练与预测")
    print("=" * 60)

    # 1. 准备训练数据
    print("[INFO] 准备训练数据...")

    # 排除的列
    EXCLUDE_COLS = {
        'business_id', 'stars', 'latitude', 'longitude', 'type', 'review_count'
    }

    # 获取特征列
    FEATURE_COLS = [c for c in df_existing.columns if c not in EXCLUDE_COLS]
    print(f"[INFO] 特征数量: {len(FEATURE_COLS)}")
    print(f"[INFO] 前10个特征: {FEATURE_COLS[:10]}")

    # 准备训练数据
    X_train = df_existing[FEATURE_COLS]
    y_train = df_existing['stars']

    # 确保虚拟餐厅有所有特征列
    missing_features = []
    for col in FEATURE_COLS:
        if col not in df_virtual_features.columns:
            missing_features.append(col)
            # 使用训练数据的平均值填充
            if col in df_existing.columns:
                mean_val = df_existing[col].mean()
                df_virtual_features[col] = mean_val
            else:
                df_virtual_features[col] = 0

    if missing_features:
        print(f"[WARN] 虚拟餐厅缺少 {len(missing_features)} 个特征，已用默认值填充")

    # 2. 训练模型
    print("[INFO] 训练LightGBM模型...")

    try:
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        model.fit(X_train, y_train)
        print("[INFO] 模型训练完成")

        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': FEATURE_COLS,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n[INFO] 特征重要性Top 15:")
        print(feature_importance.head(15).to_string(index=False))

    except Exception as e:
        print(f"[ERROR] 模型训练失败: {e}")
        return None, None, None

    # 3. 预测
    print("[INFO] 进行预测...")
    X_virtual = df_virtual_features[FEATURE_COLS]
    predictions = model.predict(X_virtual)
    predictions = np.clip(predictions, 1.0, 5.0)

    # 4. 创建结果
    results = pd.DataFrame({
        'business_id': df_virtual_features['business_id'],
        'longitude': df_virtual_features['longitude'],
        'latitude': df_virtual_features['latitude'],
        'price_range': df_virtual_features['price_range'],
        'type': df_virtual_features['type'],
        'predicted_stars': predictions,
        'walkability_score': df_virtual_features['walkability_score'],
        'competitor_density': df_virtual_features['competitor_density'],
        'drive_accessibility_score': df_virtual_features['drive_accessibility_score'],
        'sentiment_neighborhood_avg': df_virtual_features['sentiment_neighborhood_avg'],
    })

    return results, model, feature_importance


def visualize_results(results, df_existing, feature_importance):
    """可视化结果"""

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 主可视化
    fig = plt.figure(figsize=(18, 12))

    # 1.1 预测星级分布
    ax1 = plt.subplot(2, 3, 1)
    if 'stars' in df_existing.columns:
        ax1.hist(df_existing['stars'], bins=10, alpha=0.7, label='训练数据', color='blue', density=True)
    ax1.hist(results['predicted_stars'], bins=10, alpha=0.7, label='虚拟餐厅', color='red', density=True)
    ax1.set_xlabel('星级')
    ax1.set_ylabel('密度')
    ax1.set_title('星级分布对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 1.2 价格等级 vs 预测星级
    ax2 = plt.subplot(2, 3, 2)
    scatter = ax2.scatter(results['price_range'], results['predicted_stars'],
                          c=results['walkability_score'], cmap='viridis',
                          s=100, alpha=0.7, edgecolors='black')
    ax2.set_xlabel('价格等级')
    ax2.set_ylabel('预测星级')
    ax2.set_title('价格等级 vs 预测星级')
    plt.colorbar(scatter, ax=ax2, label='步行可达性')
    ax2.grid(True, alpha=0.3)

    # 1.3 步行可达性 vs 预测星级
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(results['walkability_score'], results['predicted_stars'],
                alpha=0.7, s=80)
    ax3.set_xlabel('步行可达性')
    ax3.set_ylabel('预测星级')
    ax3.set_title('步行可达性 vs 预测星级')
    ax3.grid(True, alpha=0.3)

    # 1.4 竞争密度 vs 预测星级
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(results['competitor_density'], results['predicted_stars'],
                alpha=0.7, s=80)
    ax4.set_xlabel('竞争密度')
    ax4.set_ylabel('预测星级')
    ax4.set_title('竞争密度 vs 预测星级')
    ax4.grid(True, alpha=0.3)

    # 1.5 情感邻域 vs 预测星级
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(results['sentiment_neighborhood_avg'], results['predicted_stars'],
                alpha=0.7, s=80)
    ax5.set_xlabel('情感邻域平均值')
    ax5.set_ylabel('预测星级')
    ax5.set_title('情感邻域 vs 预测星级')
    ax5.grid(True, alpha=0.3)

    # 1.6 特征重要性
    ax6 = plt.subplot(2, 3, 6)
    if feature_importance is not None and len(feature_importance) > 0:
        top_features = feature_importance.head(10)
        ax6.barh(range(len(top_features)), top_features['importance'])
        ax6.set_yticks(range(len(top_features)))
        ax6.set_yticklabels(top_features['feature'])
        ax6.set_xlabel('重要性')
        ax6.set_title('Top 10 特征重要性')
        ax6.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('predictions_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. 地理分布图
    plt.figure(figsize=(12, 10))

    # 绘制现有餐厅的分布
    plt.scatter(df_existing['longitude'], df_existing['latitude'],
                c=df_existing['stars'], cmap='RdYlGn', s=20, alpha=0.3,
                label='现有餐厅')

    # 绘制虚拟餐厅
    scatter = plt.scatter(results['longitude'], results['latitude'],
                          c=results['predicted_stars'], cmap='RdYlGn',
                          s=300, alpha=0.8, edgecolors='black', linewidth=2,
                          label='虚拟餐厅')

    plt.colorbar(scatter, label='星级')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.title('餐厅地理分布')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加虚拟餐厅标注
    for i, row in results.iterrows():
        plt.annotate(f"{row['predicted_stars']:.1f}",
                     (row['longitude'], row['latitude']),
                     fontsize=10, ha='center', va='center',
                     color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig('geo_distribution.png', dpi=600, bbox_inches='tight')
    plt.show()

    # 3. 特征值分布
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    feature_dist_cols = [
        ('walkability_score', '步行可达性'),
        ('competitor_density', '竞争密度'),
        ('drive_accessibility_score', '驾驶可达性'),
        ('sentiment_neighborhood_avg', '情感邻域')
    ]

    for idx, (col, title) in enumerate(feature_dist_cols):
        ax = axes[idx // 2, idx % 2]
        if col in results.columns:
            values = results[col]
            ax.hist(values, bins=15, alpha=0.7, color='skyblue')
            ax.set_xlabel(title)
            ax.set_ylabel('数量')
            ax.set_title(f'{title}分布')
            ax.grid(True, alpha=0.3)

            # 添加统计信息
            stats_text = f"均值: {values.mean():.2f}\n标准差: {values.std():.2f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("=" * 80)
    print("虚拟餐厅特征计算与预测系统 - 修复版")
    print("=" * 80)

    try:
        # 步骤1: 分析现有餐厅数据
        print("\n步骤1: 分析现有餐厅数据")
        df_existing, cat_cols = load_and_prepare_existing_data()

        if df_existing is None:
            print("[ERROR] 无法加载现有餐厅数据")
            return

        # 步骤2: 加载并修复虚拟餐厅数据
        print("\n步骤2: 加载并修复虚拟餐厅数据")
        df_virtual = load_and_fix_virtual_restaurants()

        if df_virtual is None or len(df_virtual) == 0:
            print("[ERROR] 虚拟餐厅数据无效")
            return

        # 步骤3: 计算虚拟餐厅特征
        print("\n步骤3: 计算虚拟餐厅特征")
        df_virtual_features = calculate_features_for_virtual_restaurants(
            df_virtual, df_existing, cat_cols
        )

        # 保存特征
        df_virtual_features.to_json(OUTPUT_VIRTUAL_JSON, orient='records',
                                    lines=True, force_ascii=False)
        print(f"\n[INFO] 虚拟餐厅特征已保存到: {OUTPUT_VIRTUAL_JSON}")

        # 显示特征统计
        print("\n[INFO] 虚拟餐厅特征统计:")
        key_features = ['walkability_score', 'competitor_density',
                        'drive_accessibility_score', 'sentiment_neighborhood_avg',
                        'betweenness_centrality', 'distance_to_strip']

        for feature in key_features:
            if feature in df_virtual_features.columns:
                values = df_virtual_features[feature]
                print(f"  {feature}:")
                print(f"    最小值: {values.min():.2f}, 最大值: {values.max():.2f}")
                print(f"    平均值: {values.mean():.2f}, 标准差: {values.std():.2f}")
                print(f"    非空值: {values.notna().sum()}/{len(values)}")

        # 步骤4: 训练模型和预测
        print("\n步骤4: 训练模型和预测")
        results, model, feature_importance = train_model_and_predict(
            df_existing, df_virtual_features
        )

        if results is None:
            print("[ERROR] 预测失败")
            return

        # 保存预测结果
        results.to_csv(PREDICTION_OUTPUT, index=False, encoding='utf-8')
        print(f"\n[INFO] 预测结果已保存到: {PREDICTION_OUTPUT}")

        # 显示结果
        print("\n" + "=" * 80)
        print("最终预测结果")
        print("=" * 80)
        print(results[['business_id', 'type', 'price_range',
                       'predicted_stars', 'walkability_score',
                       'competitor_density']].to_string(index=False))

        # 步骤5: 可视化
        print("\n步骤5: 生成可视化图表")
        visualize_results(results, df_existing, feature_importance)

        # 总结
        print("\n" + "=" * 80)
        print("处理完成!")
        print("=" * 80)
        print(f"1. 现有餐厅数据分析: {len(df_existing)} 条记录")
        print(f"2. 虚拟餐厅特征计算: {len(df_virtual_features)} 条记录")
        print(f"3. 预测结果生成: {len(results)} 条记录")

        print(f"\n预测结果统计:")
        print(f"  平均预测星级: {results['predicted_stars'].mean():.2f}")
        print(f"  星级范围: {results['predicted_stars'].min():.2f} - {results['predicted_stars'].max():.2f}")
        print(f"  星级标准差: {results['predicted_stars'].std():.2f}")

        print(f"\n特征多样性:")
        for feature in key_features:
            if feature in results.columns:
                unique_vals = results[feature].nunique()
                print(f"  {feature}: {unique_vals} 个不同值")

        print(f"\n文件输出:")
        print(f"  - 虚拟餐厅特征: {OUTPUT_VIRTUAL_JSON}")
        print(f"  - 预测结果: {PREDICTION_OUTPUT}")
        print(f"  - 可视化图表: predictions_summary.png, geo_distribution.png, feature_distributions.png")

    except Exception as e:
        print(f"\n[ERROR] 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()