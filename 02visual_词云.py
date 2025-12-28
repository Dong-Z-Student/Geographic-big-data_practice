import jsonlines
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import string
import os
from collections import Counter
from textblob import TextBlob

# 修改为jsonl文件路径
jsonl_file_path = "餐厅整体评论数据_parallel.jsonl"
positive_cloud_path = "positive_wordcloud.png"
negative_cloud_path = "negative_wordcloud.png"

plt.rcParams["axes.unicode_minus"] = False

ADVANCED_STOPWORDS = {
    # ---- 基础英文停用词 ----
    'a','an','the','and','or','but','is','are','was','were','be','been','am',
    'in','on','at','to','for','of','with','by','about','as','i','you','he',
    'she','it','we','they','this','that','these','those','have','has','had',
    'do','does','did','will','would','can','could','may','might','should','not',

    # ---- 疑问词 ----
    'when','where','why','how','what','who','whom','which','whats',

    # ---- 副词 ----
    'very','really','still','maybe','probably','actually','just','now',
    'then','ever','never','always','often','sometimes','again','even','quite',
    'rather','once','twice','only','over','than','came','going','thats','also','else','both','nashville','good',

    # ---- 连词 / 介词 ----
    'from','into','onto','within','without','across','around',
    'after','before','during','while','since','because','though','although',
    'unless','until','upon','there','other','your','some','them','their','here','another',

    # ---- 常见空洞动词 ----
    'make','take','go','come','get','got','give','gave','put','see','saw',
    'say','said','look','know','think','feel','felt','try','tried',

    # ---- 常见餐厅场景词 ----
    'place','restaurant','food','order','ordered','table','staff','people',
    'menu','service','wait','time','server','experience','location','went',
    'back','day','line','group','area','spot','item','meal','thing','things',

    # ---- 两字母常见词 ----
    'im','ive','youre','theyre','hes','shes','its','id','dont','cant',
    'yes','no','ok','oh','hi','bye'
}


# 删除长度<=3的短词
def remove_short(words):
    return [w for w in words if len(w) > 3]

# 文本清洗
def clean_text(text_list):
    text = " ".join(text_list).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\d+", "", text)  # 去数字
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    # 删除停用词 + 删除短词
    words = [w for w in words if w not in ADVANCED_STOPWORDS]
    words = remove_short(words)

    return words   # 返回清洗后的词列表

# 生成词云（用词频）
def generate_wordcloud(counter, save_path, title, positive=True):
    font_path = os.path.join("C:/", "Windows", "Fonts", "arial.ttf")

    wc = WordCloud(
        font_path=font_path,
        width=900,
        height=650,
        background_color="white",
        max_words=200,
        max_font_size=150,
        prefer_horizontal=0.8,
        colormap="OrRd" if positive else "PuBuGn"
    )

    wc.generate_from_frequencies(counter)

    plt.figure(figsize=(12, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=20, pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# 读取jsonl文件并使用textblob进行情感分析
positive_list = []
negative_list = []

with jsonlines.open(jsonl_file_path, 'r') as reader:
    for obj in reader:
        # 提取text_clean字段内容
        text = obj.get('text_clean', '')
        if text:
            # 使用textblob进行情感分析
            analysis = TextBlob(text)
            # 根据极性判断情感（>0为正面，<0为负面）
            if analysis.sentiment.polarity > 0:
                positive_list.append(text)
            elif analysis.sentiment.polarity < 0:
                negative_list.append(text)

# 后续处理逻辑保持不变
positive_words = clean_text(positive_list)
negative_words = clean_text(negative_list)

positive_counter = Counter(positive_words)
negative_counter = Counter(negative_words)

generate_wordcloud(positive_counter, positive_cloud_path, "Positive Reviews WordCloud", positive=True)
generate_wordcloud(negative_counter, negative_cloud_path, "Negative Reviews WordCloud", positive=False)

print("词云生成完成！请查看生成的两张图片。")