# -*- coding: utf-8 -*-
"""
03 情感 + LDA 主题特征构建（餐厅级）

输入（按行JSON，每行一个review记录）字段固定包含：
"review_id", "text_clean", "business_id", "price_range", "stars", "review_count",
"latitude", "longitude", "cat__***"

满足需求：
1) TextBlob 情感：逐条评论 polarity/subjectivity，按 business 求均值
2) gensim LdaMulticore：可配置训练比例（默认100%），随机抽样
3) num_topics=5；推断主题向量按批处理；导出每个topic的top words供判读
4) 输出餐厅级：business字段 + 所有cat__字段 + avg_sentiment_* + topic_*
5) 输出可选：CSV 与 按行 JSON（扩展名可用 .json）
"""

import json
import csv
import random
import time
from collections import defaultdict
from typing import Dict, Any, List, Iterable, Tuple

from textblob import TextBlob
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore


# =========================
# 固定字段名
# =========================
TEXT_FIELD = "text_clean"   # 按你的要求写死

LDA_EXTRA_STOPWORDS = {
    "good", "great", "just", "really", "nice","best",
    "love", "like", "amazing", "pretty","don","didn","ve","did"
}

# =========================
# 读写工具
# =========================

def iter_json_lines(path: str) -> Iterable[Dict[str, Any]]:
    """逐行读取（每行一个 JSON 对象）。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json_lines(path: str, rows: List[Dict[str, Any]]) -> None:
    """按行写 JSON（可用 .json 后缀，但内容是 line-delimited JSON）。"""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_topic_words_txt(path: str, topic_words: Dict[int, List[Tuple[str, float]]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for k in sorted(topic_words.keys()):
            f.write(f"Topic {k}:\n")
            f.write(", ".join([f"{w}({p:.4f})" for w, p in topic_words[k]]) + "\n\n")


def save_topic_words_json(path: str, topic_words: Dict[int, List[Tuple[str, float]]]) -> None:
    obj = {str(k): [{"word": w, "prob": float(p)} for w, p in v] for k, v in topic_words.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================
# 文本与字段处理
# =========================

def tokenize_for_lda(text: str) -> List[str]:
    """
    text_clean 已预处理，这里仅做：
    - 空格切分
    - 移除 LDA 专用情感泛化词（不影响 TextBlob）
    """
    if not text:
        return []
    return [
        t for t in text.split()
        if t and t not in LDA_EXTRA_STOPWORDS
    ]


def extract_category_columns(sample_record: Dict[str, Any], prefix: str = "cat__") -> List[str]:
    return [k for k in sample_record.keys() if k.startswith(prefix)]


# =========================
# LDA 训练：随机抽样语料（流式 I/O）
# =========================

def build_lda_training_texts(
    input_path: str,
    train_ratio: float = 1.0,
    seed: int = 42,
    min_tokens: int = 5,
    max_docs_cap: int = 0,
) -> List[List[str]]:
    """
    从全量 review 流式读取，按 Bernoulli 随机抽样得到训练文档集合。
    - train_ratio: 1.0 表示100%，0.2 表示约20%
    - max_docs_cap: >0 时最多取这么多条（防止全量时内存吃紧）
    """
    rng = random.Random(seed)
    texts = []
    seen = 0
    kept = 0

    for rec in iter_json_lines(input_path):
        seen += 1
        if train_ratio < 1.0 and rng.random() >= train_ratio:
            continue

        tokens = tokenize_for_lda(str(rec.get(TEXT_FIELD, "")))
        if len(tokens) < min_tokens:
            continue

        texts.append(tokens)
        kept += 1

        if max_docs_cap > 0 and kept >= max_docs_cap:
            break

    print(f"[LDA] scanned={seen:,}, sampled_docs={kept:,}, ratio={train_ratio}")
    return texts


def train_lda_model(
    texts: List[List[str]],
    num_topics: int = 5,
    workers: int = 0,
    passes: int = 5,
    random_state: int = 42,
    no_below: int = 10,
    no_above: float = 0.5,
    keep_n: int = 30000,
) -> Tuple[LdaMulticore, Dictionary]:
    """
    用 gensim LdaMulticore 训练 LDA。
    """
    if workers <= 0:
        import os
        workers = max(1, (os.cpu_count() or 4) - 1)

    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    corpus = [dictionary.doc2bow(t) for t in texts]

    lda = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        workers=workers,
        passes=passes,
        random_state=random_state,
        chunksize=2000
    )
    return lda, dictionary


def get_topic_words(lda: LdaMulticore, topn: int = 15) -> Dict[int, List[Tuple[str, float]]]:
    return {k: lda.show_topic(k, topn=topn) for k in range(lda.num_topics)}


# =========================
# 餐厅级聚合：情感 + 主题向量（流式 I/O + 批处理推断）
# =========================

def dense_topic_vector(lda: LdaMulticore, bow, num_topics: int) -> List[float]:
    dist = lda.get_document_topics(bow, minimum_probability=0.0)
    vec = [0.0] * num_topics
    for tid, p in dist:
        vec[int(tid)] = float(p)
    return vec


def build_restaurant_level_dataset(
    input_path: str,
    lda: LdaMulticore,
    dictionary: Dictionary,
    category_cols: List[str],
    batch_size: int = 2000
) -> List[Dict[str, Any]]:
    """
    第二遍扫描：对每条 review 计算 sentiment + topic 向量（批处理推断），并按 business_id 聚合。
    返回餐厅级 rows。
    """
    business_static: Dict[str, Dict[str, Any]] = {}

    sent_sum_pol = defaultdict(float)
    sent_sum_sub = defaultdict(float)
    sent_cnt = defaultdict(int)

    topic_sum = defaultdict(lambda: [0.0] * lda.num_topics)
    topic_cnt = defaultdict(int)

    batch_bids: List[str] = []
    batch_bows: List[Any] = []

    scanned = 0
    used = 0
    t0 = time.time()

    def flush_batch():
        nonlocal batch_bids, batch_bows
        if not batch_bids:
            return

        for bid, bow in zip(batch_bids, batch_bows):
            vec = dense_topic_vector(lda, bow, lda.num_topics)
            s = topic_sum[bid]
            for i in range(lda.num_topics):
                s[i] += vec[i]
            topic_cnt[bid] += 1

        batch_bids = []
        batch_bows = []

    for rec in iter_json_lines(input_path):
        scanned += 1

        bid = rec.get("business_id")
        if not bid:
            continue

        # 静态字段只保存一次
        if bid not in business_static:
            static = {
                "business_id": bid,
                "price_range": rec.get("price_range"),
                "stars": rec.get("stars"),
                "review_count": rec.get("review_count"),
                "latitude": rec.get("latitude"),
                "longitude": rec.get("longitude"),
            }
            for c in category_cols:
                static[c] = rec.get(c, 0)
            business_static[bid] = static

        text = str(rec.get(TEXT_FIELD, ""))

        # 1) 情感：流式逐条
        try:
            blob = TextBlob(text)
            pol = float(blob.sentiment.polarity)
            sub = float(blob.sentiment.subjectivity)
        except Exception:
            pol, sub = 0.0, 0.0

        sent_sum_pol[bid] += pol
        sent_sum_sub[bid] += sub
        sent_cnt[bid] += 1

        # 2) 主题推断：构造bow放入batch
        tokens = tokenize_for_lda(text)
        bow = dictionary.doc2bow(tokens) if tokens else []
        batch_bids.append(bid)
        batch_bows.append(bow)

        used += 1
        if len(batch_bids) >= batch_size:
            flush_batch()

        if scanned % 200000 == 0:
            elapsed = time.time() - t0
            print(f"[AGG] scanned={scanned:,}, used={used:,}, unique_business={len(business_static):,}, elapsed={elapsed:.1f}s")

    flush_batch()

    # 输出餐厅级
    rows = []
    for bid, static in business_static.items():
        n = sent_cnt.get(bid, 0)
        avg_pol = sent_sum_pol[bid] / n if n > 0 else 0.0
        avg_sub = sent_sum_sub[bid] / n if n > 0 else 0.0

        tc = topic_cnt.get(bid, 0)
        if tc > 0:
            s = topic_sum[bid]
            avg_topic = [v / tc for v in s]
        else:
            avg_topic = [0.0] * lda.num_topics

        out = dict(static)
        out["avg_sentiment_polarity"] = avg_pol
        out["avg_sentiment_subjectivity"] = avg_sub
        for i in range(lda.num_topics):
            out[f"topic_{i}"] = avg_topic[i]

        rows.append(out)

    print(f"[DONE] restaurant_rows={len(rows):,}")
    return rows


# =========================
# main
# =========================

def main():
    start = time.time()

    # ===== 路径配置（改成你的文件名）=====
    INPUT_MERGED_JSON = "餐厅整体评论数据.jsonl"  # 上一步输出：按行JSON（可以叫 .json）
    #OUT_FEATURES_CSV = "restaurant_features.csv"
    OUT_FEATURES_JSON = "restaurant_features.json"  # 按行JSON（扩展名用 .json）

    #OUT_TOPIC_WORDS_TXT = "lda_topic_words.txt"
    OUT_TOPIC_WORDS_JSON = "lda_topic_words.json"

    # ===== 可调参数 =====
    TRAIN_RATIO = 1.0      # 1.0=100%；0.2=20%（随机抽样）
    SEED = 42
    NUM_TOPICS = 5
    TOPN_WORDS = 15

    PASSES = 5
    NO_BELOW = 10
    NO_ABOVE = 0.5
    KEEP_N = 30000
    MAX_DOCS_CAP = 0       # 0=不限制；>0 可限制训练文档数（防内存吃紧）

    INFER_BATCH_SIZE = 100000

    # ===== 读取一条样本确定 cat__ 字段列表 =====
    sample = next(iter_json_lines(INPUT_MERGED_JSON))
    if TEXT_FIELD not in sample:
        raise KeyError(f"输入数据缺少字段 {TEXT_FIELD}")
    category_cols = extract_category_columns(sample, prefix="cat__")
    print(f"[INFO] TEXT_FIELD={TEXT_FIELD}, category_cols={len(category_cols)}")

    # ===== 1) 构造训练语料（流式 + 随机抽样）=====
    t0 = time.time()
    lda_texts = build_lda_training_texts(
        input_path=INPUT_MERGED_JSON,
        train_ratio=TRAIN_RATIO,
        seed=SEED,
        min_tokens=5,
        max_docs_cap=MAX_DOCS_CAP
    )
    print(f"[TIME] build training texts: {time.time() - t0:.1f}s")

    # ===== 2) 训练 LDA（gensim 多核）=====
    t1 = time.time()
    lda, dictionary = train_lda_model(
        texts=lda_texts,
        num_topics=NUM_TOPICS,
        workers=0,  # 自动CPU-1
        passes=PASSES,
        random_state=SEED,
        no_below=NO_BELOW,
        no_above=NO_ABOVE,
        keep_n=KEEP_N
    )
    print(f"[TIME] train LDA: {time.time() - t1:.1f}s")

    # ===== 3) 导出主题词（供人工判读）=====
    topic_words = get_topic_words(lda, topn=TOPN_WORDS)
    #save_topic_words_txt(OUT_TOPIC_WORDS_TXT, topic_words)
    save_topic_words_json(OUT_TOPIC_WORDS_JSON, topic_words)
    print(f"[INFO] topic words saved: {OUT_TOPIC_WORDS_JSON}")

    # ===== 4) 第二遍扫描：情感（流式）+ 主题（批推断）→ 餐厅级聚合 =====
    t2 = time.time()
    restaurant_rows = build_restaurant_level_dataset(
        input_path=INPUT_MERGED_JSON,
        lda=lda,
        dictionary=dictionary,
        category_cols=category_cols,
        batch_size=INFER_BATCH_SIZE
    )
    print(f"[TIME] build restaurant features: {time.time() - t2:.1f}s")

    """
    # ===== 5) 可选输出（你可在 main 中注释掉以减少运行时间）=====
    base_fields = ["business_id", "price_range", "stars", "review_count", "latitude", "longitude"]
    out_fields = base_fields + category_cols + [
        "avg_sentiment_polarity", "avg_sentiment_subjectivity"
    ] + [f"topic_{i}" for i in range(NUM_TOPICS)]
    # 保存 CSV
    write_csv(OUT_FEATURES_CSV, restaurant_rows, out_fields)
    """

    # 保存按行 JSON（扩展名用 .json）
    write_json_lines(OUT_FEATURES_JSON, restaurant_rows)

    #print(f"[OUTPUT] {OUT_FEATURES_CSV}")
    print(f"[OUTPUT] {OUT_FEATURES_JSON}")
    print("[DONE] all finished.")
    print(f"[Total] 总运行时间: {time.time() - start:.2f} 秒")


if __name__ == "__main__":
    main()
