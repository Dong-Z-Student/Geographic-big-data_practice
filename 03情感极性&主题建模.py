# -*- coding: utf-8 -*-
import json
import random
import time
import os
import tempfile
import multiprocessing as mp
from collections import defaultdict
from typing import Dict, Any, List, Iterable, Tuple

from textblob import TextBlob
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore

TEXT_FIELD = "text_clean"

LDA_EXTRA_STOPWORDS = {
    "good", "great", "just", "really", "nice", "best",
    "love", "like", "amazing", "pretty", "don", "didn", "ve", "did"
}

# =========================
# 读写工具
# =========================
def iter_json_lines(path: str) -> Iterable[Dict[str, Any]]:
    """逐行读取"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def write_json_lines(path: str, rows: List[Dict[str, Any]]) -> None:
    """按行写 JSON"""
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
    - 空格切分
    - 移除 LDA 停用词
    """
    if not text:
        return []
    return [t for t in text.split() if t and t not in LDA_EXTRA_STOPWORDS]


def extract_category_columns(sample_record: Dict[str, Any], prefix: str = "cat__") -> List[str]:
    return [k for k in sample_record.keys() if k.startswith(prefix)]


# =========================
# LDA 训练：随机抽样语料
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


# ============================================================
# 餐厅级聚合：I/O 单线程 + 计算多进程 + 主进程 reduce
# ============================================================
# worker 全局
_W_LDA = None
_W_DICT = None
_W_NUM_TOPICS = 0

def _worker_init(lda_path: str, dict_path: str, num_topics: int):
    """worker 启动时加载模型/词典"""
    global _W_LDA, _W_DICT, _W_NUM_TOPICS
    _W_LDA = LdaMulticore.load(lda_path)
    _W_DICT = Dictionary.load(dict_path)
    _W_NUM_TOPICS = int(num_topics)

def _dense_topic_vector_worker(bow) -> List[float]:
    dist = _W_LDA.get_document_topics(bow, minimum_probability=0.0)
    vec = [0.0] * _W_NUM_TOPICS
    for tid, p in dist:
        vec[int(tid)] = float(p)
    return vec

def _process_chunk(args):
    """
    处理一个 chunk（list[dict]），并在 worker 内部对 business_id 做局部聚合。
    返回：局部聚合结果
    """
    chunk, category_cols = args

    # 1) 静态字段
    static_map: Dict[str, Dict[str, Any]] = {}

    # 2) 情感局部聚合
    sent_sum_pol = defaultdict(float)
    sent_sum_sub = defaultdict(float)
    sent_cnt = defaultdict(int)

    # 3) topic 局部聚合
    topic_sum = defaultdict(lambda: [0.0] * _W_NUM_TOPICS)
    topic_cnt = defaultdict(int)

    for rec in chunk:
        bid = rec.get("business_id")
        if not bid:
            continue

        if bid not in static_map:
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
            static_map[bid] = static

        text = str(rec.get(TEXT_FIELD, ""))

        # sentiment
        try:
            blob = TextBlob(text)
            pol = float(blob.sentiment.polarity)
            sub = float(blob.sentiment.subjectivity)
        except Exception:
            pol, sub = 0.0, 0.0

        sent_sum_pol[bid] += pol
        sent_sum_sub[bid] += sub
        sent_cnt[bid] += 1

        # topic
        tokens = tokenize_for_lda(text)
        bow = _W_DICT.doc2bow(tokens) if tokens else []
        vec = _dense_topic_vector_worker(bow)
        s = topic_sum[bid]
        for i in range(_W_NUM_TOPICS):
            s[i] += vec[i]
        topic_cnt[bid] += 1

    topic_sum = {k: v for k, v in topic_sum.items()}

    return static_map, dict(sent_sum_pol), dict(sent_sum_sub), dict(sent_cnt), topic_sum, dict(topic_cnt)


def build_restaurant_level_dataset_parallel(
    input_path: str,
    lda: LdaMulticore,
    dictionary: Dictionary,
    category_cols: List[str],
    chunk_size: int = 10000,
    workers: int = 0
) -> List[Dict[str, Any]]:
    """
    并行版第二遍扫描：对每条 review 计算 sentiment + topic 向量，并按 business_id 聚合。
    - chunk_size：主进程每次读多少条组成一个 chunk 发给 worker
    - workers：并行进程数
    """
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 4) - 1)

    # 将模型/词典保存到临时文件，供 worker initializer 加载
    tmp_dir = tempfile.mkdtemp(prefix="lda_tmp_")
    lda_path = os.path.join(tmp_dir, "lda.model")
    dict_path = os.path.join(tmp_dir, "dict.gensim")

    lda.save(lda_path)
    dictionary.save(dict_path)

    def chunk_generator() -> Iterable[Tuple[List[Dict[str, Any]], List[str]]]:
        buf = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                buf.append(rec)
                if len(buf) >= chunk_size:
                    yield (buf, category_cols)
                    buf = []
            if buf:
                yield (buf, category_cols)

    # 主进程全局聚合容器（reduce）
    business_static: Dict[str, Dict[str, Any]] = {}

    sent_sum_pol = defaultdict(float)
    sent_sum_sub = defaultdict(float)
    sent_cnt = defaultdict(int)

    topic_sum = defaultdict(lambda: [0.0] * lda.num_topics)
    topic_cnt = defaultdict(int)

    scanned = 0
    t0 = time.time()

    with mp.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(lda_path, dict_path, lda.num_topics)
    ) as pool:

        for static_map, s_pol, s_sub, s_cnt, t_sum, t_cnt in pool.imap_unordered(
            _process_chunk,
            chunk_generator(),
            chunksize=1
        ):
            scanned += chunk_size

            # 合并静态字段
            for bid, st in static_map.items():
                if bid not in business_static:
                    business_static[bid] = st

            # 合并情感
            for bid, v in s_pol.items():
                sent_sum_pol[bid] += v
            for bid, v in s_sub.items():
                sent_sum_sub[bid] += v
            for bid, v in s_cnt.items():
                sent_cnt[bid] += v

            # 合并主题向量 sum/cnt
            for bid, vec in t_sum.items():
                s = topic_sum[bid]
                for i in range(lda.num_topics):
                    s[i] += vec[i]
            for bid, v in t_cnt.items():
                topic_cnt[bid] += v

            if scanned % 200000 == 0:
                elapsed = time.time() - t0
                print(f"[AGG-P] scanned~{scanned:,}, unique_business={len(business_static):,}, elapsed={elapsed:.1f}s")

    # 输出餐厅级 rows
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

    INPUT_MERGED_JSON = r"task_two/数据预处理后数据.json"
    OUT_FEATURES_JSON = "情感极性及主题建模后数据.json"

    OUT_TOPIC_WORDS_JSON = "主题词.json"

    # 可调参数
    TRAIN_RATIO = 1.0      # 1.0=100%；0.2=20%（随机抽样）
    SEED = 42
    NUM_TOPICS = 3
    TOPN_WORDS = 15

    PASSES = 5
    NO_BELOW = 10
    NO_ABOVE = 0.5
    KEEP_N = 30000
    MAX_DOCS_CAP = 0       # 0=不限制；>0 可限制训练文档数

    # 每个任务发给worker多少条review
    INFER_BATCH_SIZE = 10000
    # 并行 worker 数（0=自动 cpu-1）
    FEATURE_WORKERS = 0

    # 读取一条样本确定 cat__ 字段列表
    sample = next(iter_json_lines(INPUT_MERGED_JSON))
    if TEXT_FIELD not in sample:
        raise KeyError(f"输入数据缺少字段 {TEXT_FIELD}")
    category_cols = extract_category_columns(sample, prefix="cat__")
    print(f"[INFO] TEXT_FIELD={TEXT_FIELD}, category_cols={len(category_cols)}")

    # 1) 构造训练语料（流式 + 随机抽样）
    t0 = time.time()
    lda_texts = build_lda_training_texts(
        input_path=INPUT_MERGED_JSON,
        train_ratio=TRAIN_RATIO,
        seed=SEED,
        min_tokens=5,
        max_docs_cap=MAX_DOCS_CAP
    )
    print(f"[TIME] build training texts: {time.time() - t0:.1f}s")

    # 2) 训练 LDA（gensim 多核）=====
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

    # 3) 导出主题词（供人工判读）
    topic_words = get_topic_words(lda, topn=TOPN_WORDS)
    save_topic_words_json(OUT_TOPIC_WORDS_JSON, topic_words)
    print(f"[INFO] topic words saved: {OUT_TOPIC_WORDS_JSON}")


    # 4) 第二遍扫描（并行）：情感 + 主题 → 餐厅级聚合
    t2 = time.time()
    restaurant_rows = build_restaurant_level_dataset_parallel(
        input_path=INPUT_MERGED_JSON,
        lda=lda,
        dictionary=dictionary,
        category_cols=category_cols,
        chunk_size=INFER_BATCH_SIZE,
        workers=FEATURE_WORKERS
    )
    print(f"[TIME] build restaurant features: {time.time() - t2:.1f}s")

    write_json_lines(OUT_FEATURES_JSON, restaurant_rows)

    print(f"[OUTPUT] {OUT_FEATURES_JSON}")
    print("[DONE] all finished.")
    print(f"[Total] 总运行时间: {time.time() - start:.2f} 秒")


if __name__ == "__main__":
    main()