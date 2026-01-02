# -*- coding: utf-8 -*-
import json
import csv
import math
import time
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from typing import Dict, List, Iterable, Set

import matplotlib.pyplot as plt


# ---------------------------
# 工具函数
# ---------------------------
def split_categories(categories_str: str) -> List[str]:
    """将 categories 字段按逗号拆分并去空格"""
    if not categories_str:
        return []
    return [c.strip() for c in categories_str.split(',') if c.strip()]


def has_exact_category(cats: Iterable[str], target: str) -> bool:
    if not target:
        return False
    t = target.strip().lower()
    for c in cats:
        c_norm = (c or "").strip().lower()
        if c_norm == t:
            return True
    return False


def update_restaurants_cooccurrence(counter: Dict[str, int], cats_set: Set[str]) -> bool:
    """统计 Restaurants 与其它类别的共现次数
    - 仅当商铺 categories 中存在独立的 'Restaurants'时触发。
    - 对“其它类别”按类别字符串计数；每家店对每个类别只计 1 次
    """
    has_restaurants = False
    for c in cats_set:
        if (c or "").strip().lower() == "restaurants":
            has_restaurants = True
            break

    if not has_restaurants:
        return False

    for c in cats_set:
        c_norm = (c or "").strip()
        if not c_norm:
            continue
        if c_norm.lower() == "restaurants":
            continue
        counter[c_norm] += 1

    return True


# ---------------------------
# 分块并行处理
# ---------------------------
def process_chunk(chunk_data: List[str]):
    """处理单个数据块，返回统计结果。
    返回：
      city_stats: {city: [total, open, latlon_abnormal, restaurants_count]}
      global_stats: [global_open, global_total]
      restaurants_cooccur: {category: count}
      restaurants_total: 含 Restaurants 的商铺数量（精确匹配）
      category_counts: {category: shop_count}
    """
    city_stats = defaultdict(lambda: [0, 0, 0, 0])  # [总, 营业, 经纬度异常, categories含Restaurants]
    global_open = 0
    global_total = 0

    restaurants_cooccur = defaultdict(int)
    restaurants_total = 0

    category_counts = defaultdict(int)  # {category: shop_count}

    for line in chunk_data:
        try:
            data = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        except Exception:
            continue

        global_total += 1

        city = data.get('city', 'Unknown') or 'Unknown'
        is_open = data.get('is_open', 0) or 0
        latitude = data.get('latitude', None)
        longitude = data.get('longitude', None)

        categories_str = data.get('categories', '') or ''
        cats = split_categories(categories_str)
        cats_set = set(cats)

        # 类别统计：每家店对每个类别只记一次
        for c in cats_set:
            c_norm = (c or '').strip()
            if c_norm:
                category_counts[c_norm] += 1

        # 城市总数
        city_stats[city][0] += 1

        # 营业统计
        if is_open == 1:
            city_stats[city][1] += 1
            global_open += 1

        # 经纬度异常：None/0/缺失视为异常
        lat_ok = latitude is not None and latitude != 0
        lon_ok = longitude is not None and longitude != 0
        if not (lat_ok and lon_ok):
            city_stats[city][2] += 1

        # categories 字段包含 Restaurants 的商铺数量
        if has_exact_category(cats, 'Restaurants'):
            city_stats[city][3] += 1

        # Restaurants 共现
        if update_restaurants_cooccurrence(restaurants_cooccur, cats_set):
            restaurants_total += 1

    return dict(city_stats), [global_open, global_total], dict(restaurants_cooccur), restaurants_total, dict(category_counts)


def read_file_in_chunks(file_path: str, chunk_size: int = 10000):
    """按行读取文件，生成数据块，避免一次性加载整个文件到内存"""
    chunk = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            chunk.append(line)
            if i % chunk_size == 0:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def batch_chunks(file_path: str, batch_size: int, chunk_size: int):
    """将数据块分批，每批包含 batch_size 个数据块"""
    chunks_batch = []
    chunk_count = 0
    for chunk in read_file_in_chunks(file_path, chunk_size):
        chunks_batch.append(chunk)
        chunk_count += 1
        if chunk_count >= batch_size:
            yield chunks_batch
            chunks_batch = []
            chunk_count = 0
    if chunks_batch:
        yield chunks_batch


def process_chunks_in_batches(file_path: str,
                              num_processes: int,
                              batch_size: int = 4,
                              chunk_size: int = 10000):
    """分批并行处理数据块，降低峰值内存"""
    results = []
    with Pool(processes=num_processes) as pool:
        for batch_num, chunk_batch in enumerate(batch_chunks(file_path, batch_size, chunk_size)):
            print(f"处理第 {batch_num + 1} 批数据（含 {len(chunk_batch)} 个chunk）")
            batch_results = pool.map(process_chunk, chunk_batch)
            results.extend(batch_results)
            del chunk_batch
            del batch_results
    return results


def merge_results(results):
    """合并多个进程的统计结果"""
    merged_cities = defaultdict(lambda: [0, 0, 0, 0])
    global_open = 0
    global_total = 0
    restaurants_cooccur_merged = defaultdict(int)
    restaurants_total = 0
    category_counts_merged = defaultdict(int)

    for city_stats, gstats, rest_co, rest_n, cat_cnt in results:
        for city, arr in city_stats.items():
            for i in range(4):
                merged_cities[city][i] += arr[i]

        global_open += gstats[0]
        global_total += gstats[1]

        for k, v in rest_co.items():
            restaurants_cooccur_merged[k] += v

        restaurants_total += int(rest_n or 0)

        for k, v in cat_cnt.items():
            category_counts_merged[k] += v

    return (
        dict(merged_cities),
        global_open,
        global_total,
        dict(restaurants_cooccur_merged),
        restaurants_total,
        dict(category_counts_merged),
    )


# ---------------------------
# 保存 & 可视化
# ---------------------------
def save_city_stats_csv(city_stats: Dict[str, List[int]],
                        global_open: int,
                        out_csv: str = '城市统计数据.csv'):
    rows = []
    for city, arr in city_stats.items():
        total, open_cnt, latlon_abn, restaurants_cnt = arr
        open_prop = (open_cnt / global_open * 100) if global_open > 0 else 0.0
        rows.append((city, total, open_cnt, open_prop, latlon_abn, restaurants_cnt))
    rows.sort(key=lambda x: x[1], reverse=True)

    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            '城市',
            '商铺数量',
            '正在营业的商铺数量',
            '营业商铺占总营业商铺的比例(%)',
            '经纬度异常的商铺数量',
            'Restaurants商铺数量'
        ])
        for r in rows:
            writer.writerow([r[0], r[1], r[2], f"{r[3]:.2f}", r[4], r[5]])

    print(f"城市统计结果已保存：{out_csv}")


def save_category_stats_csv(category_counts: Dict[str, int],
                            out_csv: str = '类别统计数据.csv'):
    items = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['类别名称', '类别对应商铺数量'])
        for k, v in items:
            writer.writerow([k, v])
    print(f"类别统计结果已保存：{out_csv}")


def save_restaurants_cooccurrence_csv(cooccur: Dict[str, int],
                                      restaurants_shop_total: int,
                                      out_csv: str = 'Restaurants共现类别.csv'):
    """保存 Restaurants 共现统计 CSV：类别名、共同出现次数、共现占比。
    共现占比(%) = 共同出现次数 / 含 Restaurants 的商铺数量 * 100
    """
    items = sorted(cooccur.items(), key=lambda x: x[1], reverse=True)
    denom = restaurants_shop_total if restaurants_shop_total > 0 else 0

    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['类别名', '共同出现次数', '在Restaurants商铺中的占比(%)'])
        for k, v in items:
            ratio = (v / denom * 100) if denom else 0.0
            writer.writerow([k, v, f"{ratio:.4f}"])

    print(f"Restaurants 共现统计已保存：{out_csv}")


def plot_restaurants_cooccurrence_graph(cooccur: Dict[str, int],
                                        target_label: str,
                                        out_png: str,
                                        top_n: int = 40):
    """画 Restaurants 共现强度图。
    - 圆点=类别
    - 连线粗细=共现次数（归一化映射到 linewidth）
    - 圆大小=共现次数（归一化映射到 scatter size）
    """
    items = sorted(cooccur.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not items:
        print(f"[WARN] {target_label} 共现为空，跳过绘图：{out_png}")
        return

    labels = [target_label] + [k for k, _ in items]
    weights = [v for _, v in items]
    w_min, w_max = min(weights), max(weights)

    import numpy as np
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    pos = {target_label: (0.0, 0.0)}
    radius = 1.0
    for i, lab in enumerate(labels[1:], start=1):
        pos[lab] = (radius * math.cos(angles[i]), radius * math.sin(angles[i]))

    plt.figure(figsize=(10, 10), dpi=200)

    # 画边（线粗细=强度）
    for (lab, w) in items:
        if w_max > w_min:
            lw = 0.8 + 7.2 * (w - w_min) / (w_max - w_min)
        else:
            lw = 4.0
        x0, y0 = pos[target_label]
        x1, y1 = pos[lab]
        plt.plot([x0, x1], [y0, y1], linewidth=lw, alpha=0.6)

    # 画节点（圆大小=强度）
    xs = [pos[l][0] for l in labels]
    ys = [pos[l][1] for l in labels]

    sizes = [1400]
    for _, w in items:
        if w_max > 0:
            sizes.append(200 + 1800 * (w / w_max))
        else:
            sizes.append(400)

    plt.scatter(xs, ys, s=sizes, alpha=0.9)

    # 标注
    for l in labels:
        x, y = pos[l]
        plt.text(x, y, l, fontsize=8, ha='center', va='center')

    plt.title(f"Co-occurrence graph with '{target_label}' (Top {top_n})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

    print(f"共现强度图已保存：{out_png}")


# ---------------------------
# main
# ---------------------------
def main(file_path: str,
         num_processes: int = None,
         batch_size: int = 8,
         chunk_size: int = 10000):
    start_time = time.time()

    if num_processes is None:
        num_processes = min(cpu_count(), 8)

    print(f"使用 {num_processes} 个进程分批处理数据...")

    results = process_chunks_in_batches(
        file_path=file_path,
        num_processes=num_processes,
        batch_size=batch_size,
        chunk_size=chunk_size
    )

    city_stats, global_open, global_total, rest_co, rest_total, category_counts = merge_results(results)

    print("\n全局统计：")
    print(f"总商铺数量: {global_total}")
    print(f"正在营业的商铺数量: {global_open}")
    print(f"营业比例(全局): {(global_open / global_total * 100) if global_total else 0:.2f}%")
    print(f"categories 字段包含 Restaurants（精确匹配）的商铺数量: {rest_total}")

    # 城市统计 CSV
    save_city_stats_csv(city_stats, global_open, out_csv=r'task_one/城市统计数据.csv')

    # 类别统计 CSV
    save_category_stats_csv(category_counts, out_csv=r'task_one/类别统计数据.csv')

    # Restaurants 共现
    save_restaurants_cooccurrence_csv(rest_co, rest_total, out_csv=r'task_one/Restaurants共现类别.csv')
    plot_restaurants_cooccurrence_graph(rest_co, target_label='Restaurants', out_png=r'task_one/Restaurants共现强度图.png', top_n=40)

    end_time = time.time()
    print(f"\n总运行时间：{end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    file_path = r"data/yelp_academic_dataset_business.json"
    main(file_path)