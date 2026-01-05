# -*- coding: utf-8 -*-
import json
import re
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# =========================
# 基础配置
# =========================
CITY_TARGET = "philadelphia"

PHILLY_BBOX = {
    "lat_min": 39.85,
    "lat_max": 40.15,
    "lon_min": -75.35,
    "lon_max": -74.95
}

CUSTOM_STOPWORDS = {"food", "place", "restaurant", "one", "really", "also"}
STOPWORDS = ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS)

CATEGORY_VOCAB = [
    "Sandwiches","Bars","American (Traditional)","Pizza","Fast Food",
    "Breakfast & Brunch","American (New)","Burgers","Mexican","Italian",
    "Coffee & Tea","Seafood","Chinese","Salad","Chicken Wings","Cafes",
    "Delis","Specialty Food","Bakeries","Japanese","Sushi Bars","Barbeque",
    "Asian Fusion","Steakhouses","Diners","Mediterranean","Vegetarian",
    "Soup","Tacos","Food Trucks","Thai","Cajun/Creole","Tex-Mex",
    "Vietnamese","Indian","Greek","Buffets","Middle Eastern","Gastropubs",
    "French","Korean","Spanish","Cuban","Canadian (New)","Pakistani","Irish",
    "Hawaiian","Modern European","German","African","Szechuan","Beer Gardens",
    "Puerto Rican","Cantonese","Turkish","Lebanese","Peruvian","Taiwanese",
    "Brazilian","British","Ethiopian","Colombian","Salvadoran","Moroccan",
    "Venezuelan","Dominican","Afghan","Polish","Russian","Persian/Iranian",
    "Basque","Arabic","Mongolian","Argentine","Malaysian","Honduran",
    "Belgian","Indonesian","Himalayan/Nepalese","Haitian","Ukrainian",
    "Burmese","Cambodian","Trinidadian","Sicilian","Egyptian","Armenian",
    "Bangladeshi","Australian","Scandinavian"
]


# =========================
# 工具函数
# =========================
def norm(s):
    return (s or "").strip().lower()

VOCAB_NORM = {norm(c) for c in CATEGORY_VOCAB}

def sanitize_col_name(s):
    return s.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")

def parse_categories(cat_str):
    if not cat_str:
        return []
    return [c.strip() for c in cat_str.split(",") if c.strip()]

def has_exact_restaurants(cats):
    return any(norm(c) == "restaurants" for c in cats)

def in_philly_bbox(lat, lon):
    if lat is None or lon is None:
        return False
    return (
        PHILLY_BBOX["lat_min"] <= lat <= PHILLY_BBOX["lat_max"] and
        PHILLY_BBOX["lon_min"] <= lon <= PHILLY_BBOX["lon_max"]
    )

_text_re = re.compile(r"[^a-z\s]")
def clean_text(text):
    text = (text or "").lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = _text_re.sub(" ", text)
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)

def write_merged_json(out_path, rows_iter):
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows_iter:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =========================
# Business 并行处理
# =========================
def process_business_chunk(lines):
    rows = []
    price_counter = Counter()
    bids = set()

    for line in lines:
        try:
            d = json.loads(line)
        except Exception:
            continue

        if norm(d.get("city")) != CITY_TARGET:
            continue
        if d.get("is_open") != 1:
            continue
        if (d.get("review_count") or 0) < 5:
            continue

        lat, lon = d.get("latitude"), d.get("longitude")
        if not in_philly_bbox(lat, lon):
            continue

        cats = parse_categories(d.get("categories", ""))
        if not has_exact_restaurants(cats):
            continue

        # price_range
        attrs = d.get("attributes") or {}
        pr = attrs.get("RestaurantsPriceRange2")
        pr_int = None
        try:
            pr_int = int(str(pr).strip().strip('"').strip("'"))
            price_counter[pr_int] += 1
        except Exception:
            pr_int = None

        cat_norms = {norm(c) for c in cats if norm(c) != "restaurants"}
        hit_vocab = any(c in VOCAB_NORM for c in cat_norms)

        cat_flags = {}
        for raw in CATEGORY_VOCAB:
            col = f"cat__{sanitize_col_name(raw)}"
            cat_flags[col] = int(norm(raw) in cat_norms)

        cat_flags["cat__Other"] = int(not hit_vocab)

        row = {
            "business_id": d.get("business_id"),
            "price_range": pr_int,
            "stars": d.get("stars"),
            "review_count": d.get("review_count"),
            "latitude": lat,
            "longitude": lon,
            **cat_flags
        }

        rows.append(row)
        bids.add(row["business_id"])

    return rows, price_counter, bids


def chunk_reader(path, chunk_size=10000):
    chunk = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def preprocess_business_parallel(path, chunk_size=10000, n_proc=None):
    if n_proc is None:
        n_proc = max(1, cpu_count() - 1)

    rows_all = []
    price_counter_all = Counter()
    business_id_set = set()

    with Pool(processes=n_proc) as pool:
        for rows, pc, bids in pool.imap_unordered(
            process_business_chunk,
            chunk_reader(path, chunk_size)
        ):
            rows_all.extend(rows)
            price_counter_all.update(pc)
            business_id_set.update(bids)

    mode_price = price_counter_all.most_common(1)[0][0] if price_counter_all else 2
    for r in rows_all:
        if r["price_range"] is None:
            r["price_range"] = mode_price

    business_map = {r["business_id"]: r for r in rows_all}
    return business_map, business_id_set, mode_price


# =========================
# Review → merge 生成器
# =========================
# def iter_merged_reviews(review_path, business_map, business_id_set):
#     with open(review_path, "r", encoding="utf-8") as f:
#         for line in f:
#             try:
#                 d = json.loads(line)
#             except Exception:
#                 continue
#
#             bid = d.get("business_id")
#             if bid not in business_id_set:
#                 continue
#
#             b = business_map.get(bid)
#             if b is None:
#                 continue
#
#             yield {
#                 "review_id": d.get("review_id"),
#                 "text_clean": clean_text(d.get("text", "")),
#                 **b
#             }


def iter_merged_reviews(review_path, business_map, business_id_set):
    """
    在同一个函数里完成：
    1) 第1遍扫描 review：统计每个 business_id 的 avg_stars
    2) 第2遍扫描 review：输出每条 review，并带上 avg_stars
    main 不需要再单独调用统计函数
    """
    from collections import defaultdict

    # -------- pass 1: compute avg_stars per business_id --------
    sum_map = defaultdict(float)
    cnt_map = defaultdict(int)

    with open(review_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue

            bid = d.get("business_id")
            if bid not in business_id_set:
                continue

            s = d.get("stars")
            try:
                s = float(s)
            except Exception:
                continue

            sum_map[bid] += s
            cnt_map[bid] += 1

    avg_stars_map = {}
    for bid, ssum in sum_map.items():
        c = cnt_map.get(bid, 0)
        avg_stars_map[bid] = (ssum / c) if c > 0 else None

    # -------- pass 2: yield merged rows --------
    with open(review_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue

            bid = d.get("business_id")
            if bid not in business_id_set:
                continue

            b = business_map.get(bid)
            if b is None:
                continue

            yield {
                "review_id": d.get("review_id"),
                "text_clean": clean_text(d.get("text", "")),
                "avg_stars": avg_stars_map.get(bid),  # NEW FIELD
                **b
            }




# =========================
# main
# =========================
def main(business_path, review_path):
    start_time = time.time()

    business_map, business_id_set, mode_price = preprocess_business_parallel(
        business_path
    )

    print(f"[Business] 餐厅数量: {len(business_map)}")
    print(f"[Business] price_range 众数: {mode_price}")

    merged_iter = iter_merged_reviews(
        review_path, business_map, business_id_set
    )
    write_merged_json(r"task2/数据预处理后数据.json", merged_iter)

    print("[Done] 合并数据处理完成")
    print(f"结果已保存为：task2/数据预处理后数据.json")
    print(f"\n总运行时间：{time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main(
        business_path="data/yelp_academic_dataset_business.json",
        review_path="data/yelp_academic_dataset_review.json"
    )
