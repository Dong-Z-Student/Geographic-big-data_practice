# -*- coding: utf-8 -*-
import json
import re
import csv
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# =========================
# 配置
# =========================

CITY_TARGET = "philadelphia"

# Philadelphia 矩形框（按你现有代码的范围替换即可）
PHILLY_BBOX = {
    "lat_min": 39.85,
    "lat_max": 40.15,
    "lon_min": -75.35,
    "lon_max": -74.95
}

CUSTOM_STOPWORDS = {"food", "place", "restaurant", "one", "really", "also"}
STOPWORDS = ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS)

# 你的白名单类别（多值）
CATEGORY_VOCAB = [
    "Sandwiches","Bars","American (Traditional)","Pizza","Fast Food","Breakfast & Brunch",
    "American (New)","Burgers","Mexican","Italian","Coffee & Tea","Seafood","Chinese","Salad",
    "Chicken Wings","Cafes","Delis","Specialty Food","Bakeries","Japanese","Sushi Bars","Barbeque",
    "Asian Fusion","Steakhouses","Diners","Mediterranean","Vegetarian","Soup","Tacos","Food Trucks",
    "Thai","Cajun/Creole","Tex-Mex","Vietnamese","Indian","Greek","Buffets","Middle Eastern",
    "Gastropubs","French","Korean","Spanish","Cuban","Canadian (New)","Pakistani","Irish","Hawaiian",
    "Modern European","German","African","Szechuan","Beer Gardens","Puerto Rican","Cantonese","Turkish",
    "Lebanese","Peruvian","Taiwanese","Brazilian","British","Ethiopian","Colombian","Salvadoran",
    "Moroccan","Venezuelan","Dominican","Afghan","Polish","Russian","Persian/Iranian","Basque","Arabic",
    "Mongolian","Argentine","Malaysian","Honduran","Belgian","Indonesian","Himalayan/Nepalese","Haitian",
    "Ukrainian","Burmese","Cambodian","Trinidadian","Sicilian","Egyptian","Armenian","Bangladeshi",
    "Australian","Scandinavian"
]


# =========================
# 工具函数
# =========================

def norm(s: str) -> str:
    return (s or "").strip().lower()

# 规范化后的白名单集合
VOCAB_NORM = {norm(c) for c in CATEGORY_VOCAB}

def sanitize_col_name(raw: str) -> str:
    # 列名不要有空格和斜杠，方便后续建模
    return raw.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")

def parse_categories(cat_str: str):
    if not cat_str:
        return []
    return [c.strip() for c in cat_str.split(",") if c.strip()]

def has_exact_restaurants(cats):
    # 精确 token 匹配（忽略大小写）
    return any(norm(c) == "restaurants" for c in cats)

def in_philly_bbox(lat, lon):
    if lat is None or lon is None:
        return False
    return (
        PHILLY_BBOX["lat_min"] <= lat <= PHILLY_BBOX["lat_max"] and
        PHILLY_BBOX["lon_min"] <= lon <= PHILLY_BBOX["lon_max"]
    )

_text_re = re.compile(r"[^a-z\s]")
def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = _text_re.sub(" ", text)
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)


# =========================
# Business 并行处理（每块）
# =========================

def process_business_chunk(lines):
    """
    返回：
      rows: list[dict]  (business 特征行，price_range 可能为 None)
      price_counter: Counter  (price_range 频数，用于全局众数)
      bids: set[str]  (通过筛选的 business_id)
    """
    rows = []
    price_counter = Counter()
    bids = set()

    for line in lines:
        try:
            d = json.loads(line)
        except Exception:
            continue

        # a) city
        city = norm(d.get("city"))
        if city != CITY_TARGET:
            continue

        # b) is_open
        if d.get("is_open") != 1:
            continue

        # c) review_count >= 5
        if (d.get("review_count") or 0) < 5:
            continue

        # e) bbox
        lat = d.get("latitude")
        lon = d.get("longitude")
        if not in_philly_bbox(lat, lon):
            continue

        # f) categories 必须精确匹配包含 Restaurants
        cats = parse_categories(d.get("categories", ""))
        if not has_exact_restaurants(cats):
            continue

        # d) price_range：允许缺失，后面用众数填
        attrs = d.get("attributes") or {}
        pr = attrs.get("RestaurantsPriceRange2")
        pr_int = None
        try:
            pr_int = int(str(pr).strip().strip('"').strip("'"))
            price_counter[pr_int] += 1
        except Exception:
            pr_int = None

        # multi-hot：仅对你的白名单做 0/1
        cat_norms = {norm(c) for c in cats}
        # 忽略 Restaurants 本身（它只是筛选条件，不作为“餐厅类型”特征）
        cat_norms_wo_rest = {c for c in cat_norms if c != "restaurants"}

        hit_any_vocab = any(c in VOCAB_NORM for c in cat_norms_wo_rest)

        cat_flags = {}
        for raw in CATEGORY_VOCAB:
            key = f"cat__{sanitize_col_name(raw)}"
            cat_flags[key] = int(norm(raw) in cat_norms_wo_rest)

        # 修正后的 Other：只有当“白名单一个都没命中”才是 Other=1
        cat_flags["cat__Other"] = int(not hit_any_vocab)

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


# =========================
# 主流程：business 并行 -> 得到 business_map / business_id_set / mode_price
# =========================

def preprocess_business_parallel(business_path, chunk_size=10000, n_proc=None):
    if n_proc is None:
        n_proc = max(1, cpu_count() - 1)

    rows_all = []
    price_counter_all = Counter()
    business_id_set = set()

    with Pool(processes=n_proc) as pool:
        for rows, pc, bids in pool.imap_unordered(process_business_chunk, chunk_reader(business_path, chunk_size)):
            rows_all.extend(rows)
            price_counter_all.update(pc)
            business_id_set.update(bids)

    # 众数填充：如果极端情况下所有都缺失，给一个安全默认值 2
    mode_price = 2
    if price_counter_all:
        mode_price = price_counter_all.most_common(1)[0][0]

    # 填充缺失
    for r in rows_all:
        if r["price_range"] is None:
            r["price_range"] = mode_price

    # 构建映射，供 review 流式 merge
    business_map = {r["business_id"]: r for r in rows_all}

    return business_map, business_id_set, mode_price


# =========================
# Review 流式：过滤 + 清洗 + 直接 merge 输出 CSV + JSONL
# =========================

def stream_merge_reviews(review_path, business_map, business_id_set,
                         out_csv_path="merged.csv",
                         out_jsonl_path="merged.jsonl"):

    # 先准备 CSV 头
    # 业务字段：从 business_map 任取一条
    sample_business = next(iter(business_map.values()))
    business_fields = list(sample_business.keys())  # 含 business_id

    # review 字段（你要求保留 review_id、business_id、text 清洗后）
    # 注意：business_id 已在 business_fields 中，所以 review 只加 review_id + text_clean
    fieldnames = ["review_id", "text_clean"] + [f for f in business_fields]

    with open(review_path, "r", encoding="utf-8") as fin, \
         open(out_csv_path, "w", encoding="utf-8", newline="") as fcsv, \
         open(out_jsonl_path, "w", encoding="utf-8") as fjsonl:

        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        kept = 0
        for line in fin:
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

            out = {
                "review_id": d.get("review_id"),
                "text_clean": clean_text(d.get("text", "")),
                **b
            }

            writer.writerow(out)
            fjsonl.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept += 1

    return kept


# =========================
# main
# =========================

def main(business_path, review_path):
    start_time = time.time()
    business_map, business_id_set, mode_price = preprocess_business_parallel(
        business_path,
        chunk_size=10000,
        n_proc=max(1, cpu_count() - 1)
    )

    print(f"[Business] 过滤后餐厅数量: {len(business_map)}")
    print(f"[Business] price_range 众数: {mode_price}")

    kept_reviews = stream_merge_reviews(
        review_path,
        business_map=business_map,
        business_id_set=business_id_set,
        out_csv_path="餐厅整体评论数据.csv",
        out_jsonl_path="餐厅整体评论数据.jsonl"
    )

    print(f"[Review] 合并输出 review 行数: {kept_reviews}")
    print("[Output] 餐厅整体评论数据已生成")

    end_time = time.time()
    print(f"\n总运行时间：{end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main(
        business_path="data/yelp_academic_dataset_business.json",
        review_path="data/yelp_academic_dataset_review.json"
    )
