#!/usr/bin/env python
# coding: utf-8

# In[3]:


# ============================================================
# 模块1-0：数据获取（从 Kaggle 原始文件开始）
# 数据源：Kaggle bittlingmayer/amazonreviews（Amazon Polarity）
# 原始格式：fastText 纯文本，__label__1=消极, __label__2=积极
# ============================================================
import bz2
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------- 路径 ----------
RAW_DIR = r"D:\训练"           # 原始 .bz2 所在目录
DATA_DIR = r"D:\训练\data2"    # 解析结果存放目录
os.makedirs(DATA_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(RAW_DIR, "train.ft.txt.bz2")
TEST_PATH = os.path.join(RAW_DIR, "test.ft.txt.bz2")

# ---------- 中文字体（全局设置一次，后面所有图都生效）----------
font_paths = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]
font_path = next((p for p in font_paths if os.path.exists(p)), None)

if font_path:
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    print("已加载中文字体：", font_path)
else:
    print("未找到中文字体，图表将使用英文标签")

# ---------- 看一眼原始文件长什么样 ----------
print("\n===== 原始文件前3行 =====")
for path, name in [(TRAIN_PATH, "train"), (TEST_PATH, "test")]:
    print(f"\n--- {name}.ft.txt.bz2 ---")
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(f"  {line[:110]}...")

# In[4]:


# ============================================================
# 解析 fastText 格式：把 __label__1 / __label__2 转成 0 / 1
# ============================================================
def parse_fasttext(path, name):
    labels, texts = [], []
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            label, text = line.rstrip("\n").split(" ", 1)  #分别装情感标签与文本内容   #在第一个空格处分开
            labels.append(0 if label == "__label__1" else 1)    #__label__1 → 0   __label__2 → 1
            texts.append(text)

    df = pd.DataFrame({"sentiment": labels, "review": texts})       #sentiment情感标签
    print(f"{name}：{len(df):,} 条   "
          f"消极 {(df['sentiment'] == 0).sum():,}   "
          f"积极 {(df['sentiment'] == 1).sum():,}")
    return df


print("开始解析原始数据，约需5分钟...")
t0 = time.time()

train_raw = parse_fasttext(TRAIN_PATH, "训练集(原始)")       #训练集 = 给模型做题和学习的题目  测试集 = 学完以后，用来检查模型真正会不会的题目
test_raw = parse_fasttext(TEST_PATH, "测试集(原始)")

print(f"\n解析完成，总耗时 {time.time() - t0:.0f} 秒")

# In[5]:


# ============================================================
# 原始数据 360 万条，全部训练会很慢，随机抽样控制计算量
# 原始统计数字要记录下来，报告里要用
# ============================================================
N_TRAIN, N_TEST = 200_000, 40_000               #从原始训练集中抽 20万条，从原始测试集中抽 4万条

train = train_raw.sample(n=N_TRAIN, random_state=42).reset_index(drop=True) #从 train_raw 里面随机抽取 20 万条数据
test = test_raw.sample(n=N_TEST, random_state=42).reset_index(drop=True)    #drop=True 原来的旧索引不要保留下来

print("训练集：", train.shape, "  积极占比 %.2f%%" % (train["sentiment"].mean() * 100))     #因为0代表消极,1代表积极.求均值就代表了积极的占比
print("测试集：", test.shape, "  积极占比 %.2f%%" % (test["sentiment"].mean() * 100))

# 存成 CSV，后续模块直接读，不用每次重新解析
train.to_csv(os.path.join(DATA_DIR, "amazon_train.csv"), index=False, encoding="utf-8-sig")
test.to_csv(os.path.join(DATA_DIR, "amazon_test.csv"), index=False, encoding="utf-8-sig")
print("\n已保存：amazon_train.csv / amazon_test.csv")

# 记录原始数据统计（写报告用）
raw_stats = pd.DataFrame({
    "数据集": ["训练集", "测试集"],
    "原始条数": [len(train_raw), len(test_raw)],
    "消极条数": [(train_raw["sentiment"] == 0).sum(), (test_raw["sentiment"] == 0).sum()],
    "积极条数": [(train_raw["sentiment"] == 1).sum(), (test_raw["sentiment"] == 1).sum()],
    "抽样后条数": [len(train), len(test)],
})
raw_stats.to_csv(os.path.join(DATA_DIR, "amazon_raw_stats.csv"),
                 index=False, encoding="utf-8-sig")
print(raw_stats)

# In[6]:


# ============================================================
# 模块1-A：数据认识
# 从这一步开始只读 CSV，不再解析原始文件
# ============================================================
train = pd.read_csv(os.path.join(DATA_DIR, "amazon_train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "amazon_test.csv"))

print("训练集：", train.shape)
print("测试集：", test.shape)
print("列名：", list(train.columns))
print(train.head(3))

# ---------- 情感标签分布 ----------
print("\n===== 情感标签分布（0=消极, 1=积极）=====")
dist = pd.DataFrame({
    "训练集数量": train["sentiment"].value_counts().sort_index(),
    "训练集占比%": (train["sentiment"].value_counts(normalize=True).sort_index() * 100).round(2),    #统计每个值出现了多少次，并把“次数”变成“占比”。
    "测试集数量": test["sentiment"].value_counts().sort_index(),
    "测试集占比%": (test["sentiment"].value_counts(normalize=True).sort_index() * 100).round(2),
})
dist.index = ["消极(0)", "积极(1)"]
print(dist)

# ---------- 数据质量 ----------
print("\n===== 数据质量 =====")
quality = pd.DataFrame({
    "数据类型": train.dtypes.astype(str),       #.dtypes: 查看每列数据类型 ,  .astype: 转换成字符串
    "缺失值": train.isnull().sum(),
    "唯一值数": train.nunique(),
})
print(quality)
print("重复评论数：", train.duplicated(subset=["review"]).sum())  #判断数据是否重复,不重复为False,重复的返回True
print("完全重复行数：", train.duplicated().sum())      #subset=["review"])表示只看review这一列

# ---------- 评论长度 ----------
train["review_len"] = train["review"].str.len()
train["word_count"] = train["review"].str.split().str.len()

print("\n===== 评论长度（字符数）=====")
print(train["review_len"].describe().round(1).to_frame("字符数"))    #to_frame("字符数"): 把这个 Series 转成 DataFrame，并把这一列命名为“字符数”。

print("\n===== 评论长度（词数）=====")
print(train["word_count"].describe().round(1).to_frame("词数"))

print("\n===== 不同情感的平均长度 =====")
print(train.groupby("sentiment")[["review_len", "word_count"]].mean().round(1))

# ---------- 各看一条 ----------    #分别从训练集中找一条消极评论和一条积极评论，然后把它们打印出来看看。
print("\n===== 样例 =====")
for s in [0, 1]:
    r = train[train["sentiment"] == s].iloc[0]      #找到当前类别的所有评论，然后只拿第一条评论出来，存到 r
    tag = "消极" if s == 0 else "积极"
    print(f"\n--- {tag}({s})，{r['review_len']} 字符 ---")
    print(r["review"][:260])

# In[7]:


# ============================================================
# 模块1-B：标签分布 + 评论长度可视化
# ============================================================
plt.close("all")
# 注意：sns.set_theme() 会重置 matplotlib 参数，字体必须在它之后重新设一次
sns.set_theme(style="whitegrid")
if font_path:
    plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5))         #一次性创建一个画布(fiig)和若干个子图区域(axes)。
names = ["消极(0)", "积极(1)"]              #宽与高
colors = ["#E76F51", "#2A9D8F"]

# ---------- 图1：标签分布 ----------    画一张“训练集 vs 测试集的正负评论数量”柱状图。
ax = axes[0]        #表示在第一个子图上操作
xt = np.arange(2)   #给两个类别在图上的横轴位置编号。
w = 0.36            #柱子宽度为0.36
b1 = ax.bar(
    xt - w / 2,         #调整柱子位置,不能让两根柱子完全重叠。
    train["sentiment"].value_counts().sort_index(),         #value_counts()默认按照从多到少排序,而sort_index()指定按索引排序
    w,
    label="训练集",
    color="#9BBBD4"
)
b2 = ax.bar(
    xt + w / 2,         #调整柱子位置,不能让两根柱子完全重叠。
    test["sentiment"].value_counts().sort_index(),
    w,
    label="测试集",
    color="#2364AA"
)
for bs in (b1, b2):                 #在柱子顶端标注数据
    for b in bs:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1200,       #找到中心点,在每根柱子上方标出它具体有多少条评论。
                f"{int(b.get_height()):,}", ha="center", fontsize=9)
ax.set_xticks(xt)           #横坐标的刻度放在 0 和 1 这两个位置。
ax.set_xticklabels(names)   #将0与1 分别替换为  names = ["消极(0)", "积极(1)"]
ax.set_ylabel("评论数量")
ax.set_title("情感标签分布：基本 50:50", loc="left", fontweight="bold")
ax.legend(frameon=False)        #设置图例(不要边框)

# ---------- 图2：按情感分组的长度分布 ----------
ax = axes[1]
for s in [0, 1]:
    ax.hist(train.loc[train["sentiment"] == s, "review_len"], bins=60,      #hist为直方图 ; 找出当前情感类别的评论，然后只拿它们的 review_len。
            alpha=0.6, label=names[s], color=colors[s])     #bins=60 表示分为60个区间;alpha表示透明度
ax.set_xlabel("评论字符数")
ax.set_ylabel("评论数量")
ax.set_title("评论长度分布（按情感）", loc="left", fontweight="bold")
ax.legend(frameon=False)

# ---------- 图3：词数箱线图 ----------
ax = axes[2]
data = [train.loc[train["sentiment"] == s, "word_count"] for s in [0, 1]]           #loc用法: (a,b)先筛选行a,再筛选列b; 还可用于条件筛选: 如df.loc[df["age"] > 20]
bp = ax.boxplot(data, tick_labels=names, patch_artist=True,         #boxplot 为箱线图 ; patch_artist=True 表示让箱子本身可以被填充颜色
                widths=0.5, showfliers=False)               #showfliers=False表示不显示异常点(不是删除!)
for patch, c in zip(bp["boxes"], colors):           #内置函数zip用法: 表示把多个序列按照位置一一配对。
    patch.set_facecolor(c)      #设置填充颜色
    patch.set_alpha(0.7)
ax.set_ylabel("评论词数")
ax.set_title("评论词数箱线图", loc="left", fontweight="bold")

for ax in axes:
    ax.spines["top"].set_visible(False)             #此处循环表示: 把三个图右边和上边的边框去掉。
    ax.spines["right"].set_visible(False)

plt.tight_layout()                  #自动调整三个子图的位置和间距。
plt.show()

# In[8]:


# ============================================================
# 模块2-A：NLTK 环境准备 + 停用词表
# 核心要点：否定词绝不能删
# ============================================================

# 就是目的是想要去掉句子里面一些没啥用的词汇来简化,
# ALL_STOP是标准名单,包含所有的.
# 但是里面包含的一些否定词不能直接去掉,会对句子产生歧义.
# 所以把这些否定词整理成一个集合NEGATIONS,
# 最终取出ALL_STOP里面除了NEGATIONS的其他所有,把这个集合当成最终删除的词集合

import re
import time

import nltk         #一个专门处理自然语言文本的 Python 工具包。

for r in ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab",
          "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(r, quiet=True)

from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# 否定词集合：这些词必须从停用词表里剔除
# 理由：删掉 not 之后，"not good" 和 "good" 就变成同一个意思了
NEGATIONS = {"not", "no", "nor", "never", "none", "cannot", "nothing", "neither",       #把重要的否定词专门列出来。
             "n't", "isn't", "wasn't", "aren't", "weren't", "don't", "doesn't",
             "didn't", "won't", "wouldn't", "can't", "couldn't", "shouldn't",
             "hasn't", "haven't", "hadn't", "mustn't", "needn't", "ain't"}

ALL_STOP = set(stopwords.words("english"))          #完整的英文停用词集合。
STOP_WORDS = ALL_STOP - NEGATIONS                   #从标准英文停用词表里，把否定词排除出去。

print(f"标准停用词：{len(ALL_STOP)} 个")
print(f"剔除否定词后：{len(STOP_WORDS)} 个")
print(f"已保留的否定词：{sorted(NEGATIONS & ALL_STOP)}")


# In[9]:


# ============================================================
# 模块2-B：文本清洗 + 分词 + 去停用词 + 词形还原
# ============================================================
LEMMATIZER = WordNetLemmatizer()        #创建一个“词形还原器”，以后拿它把单词还原成基本形式。
WN_POS = {"J": wordnet.ADJ, "N": wordnet.NOUN,
          "V": wordnet.VERB, "R": wordnet.ADV}   #词形还原不能只看单词本身，还需要知道它是什么词性。

# 缩写展开：顺序很重要，特殊缩写要放在通用的 n't 之前
# 否则 "can't" 会变成 "ca" + "n't"，留下 ca 这种垃圾词
CONTRACTIONS = [
    (r"won't", "will not"), (r"can't", "cannot"), (r"shan't", "shall not"),
    (r"n't", " not"), (r"'re", " are"), (r"'ve", " have"), (r"'ll", " will"),
    (r"'d", " would"), (r"'m", " am"), (r"'s", " is"),
]

HTML_RE = re.compile(r"<.*?>")          # HTML 标签       找出 HTML 标签。
URL_RE = re.compile(r"http\S+|www\.\S+")  # 网址
KEEP_RE = re.compile(r"[^a-z\s]")        # 只保留字母和空格
SPACE_RE = re.compile(r"\s+")            # 多余空格


def expand_contractions(text):      #展开缩写：don't → do not   text为评论文本
    for pat, rep in CONTRACTIONS:
        text = re.sub(pat, rep, text)       #pat指的是缩写格式,rep指的是完整格式 ; #把 text 里面符合 pat 的内容，替换成 rep。
    return text


def clean_text(text):
    """文本清洗：展开缩写 → 小写 → 去HTML → 去网址 → 只留字母 → 去多余空格"""
    text = expand_contractions(text.lower())            #调用前面的那个expand_contractions函数展开缩写,并全部改成小写
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = KEEP_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def lemmatize_tokens(tokens):
    """带词性标注的词形还原
    不加POS会把 was 变成 wa、把 running 变成 running（失败）"""
    tagged = pos_tag(tokens)        # 判断每个单词是什么词性。
    return [LEMMATIZER.lemmatize(w, WN_POS.get(t[0].upper(), wordnet.NOUN))
            for w, t in tagged]     #把 tagged 里的每一个单词拿出来，做词形还原，然后把还原后的所有单词组成一个列表返回。
                                    #把单词变回词典里的基本形式，减少同一个词因为时态、单复数等产生的不同写法(books -> book)

def process(text):
    """完整流程：清洗 → 分词 → 去停用词(保留否定词) → 词形还原"""
    text = clean_text(text)         #文本清理
    tokens = word_tokenize(text)    #自带的分词函数: 把一句英文文本分成一个个单词
    tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]      #去停用词 + 去掉长度为1的词
    return " ".join(lemmatize_tokens(tokens))       #先词形还原,再再 " ".join(...) 把列表重新拼成一句字符串。


# ---------- 效果演示 ----------
print("===== 处理效果示例 =====")
for s in [
    "NOT GOOD. I don't recommend it. It isn't worth $50!",
    "Batteries died within a year ...: However, it doesn't work anymore!!!",
]:
    print(f"\n原文：{s}")
    print(f"清洗：{clean_text(s)}")
    print(f"最终：{process(s)}")

# In[10]:


# ============================================================
# 模块2-C：全量文本处理
# 分块执行 + 每块存盘，万一中断可以断点续跑
# ============================================================
import os

DATA_DIR = r"D:\训练\data2"

train = pd.read_csv(os.path.join(DATA_DIR, "amazon_train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "amazon_test.csv"))

CLEAN_TRAIN = os.path.join(DATA_DIR, "amazon_train_clean.csv")
CLEAN_TEST = os.path.join(DATA_DIR, "amazon_test_clean.csv")
CHUNK = 25_000          #每次处理 25,000 条数据。


def process_dataframe(df, out_path, name):          #df → 要处理的数据; out_path → 清洗后的数据保存在哪里; name:代号
    # 断点续跑：如果已经有中间结果，从断的地方继续
    if os.path.exists(out_path):                    #先查看文件是否存在
        done = pd.read_csv(out_path)
        print(f"[{name}] 发现中间结果 {len(done):,} 条，继续处理")
    else:
        done = pd.DataFrame(columns=list(df.columns) + ["clean_review"])

    start = len(done)
    t0 = time.time()
    for i in range(start, len(df), CHUNK):
        part = df.iloc[i:i + CHUNK].copy()                            #从 df 里面取当前这一批数据出来
        part["clean_review"] = [process(t) for t in part["review"]]   #把这一批数据中 review 列的每一条评论都拿去执行 process()
        done = pd.concat([done, part], ignore_index=True)        #把刚刚处理好的这一批数据，拼到前面已经处理好的数据后面。
        done.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[{name}] {len(done):,}/{len(df):,}   用时 {(time.time()-t0)/60:.1f} 分钟",
              flush=True)
    return done


print("开始处理文本，约需22分钟...\n")
t0 = time.time()

train_clean = process_dataframe(train, CLEAN_TRAIN, "训练集")
test_clean = process_dataframe(test, CLEAN_TEST, "测试集")

print(f"\n全部完成，总耗时 {(time.time()-t0)/60:.1f} 分钟")
print("已保存：amazon_train_clean.csv / amazon_test_clean.csv")

# In[11]:


# ============================================================
# 模块2-D：处理结果检查
# 后续模块直接读这两个 clean CSV，不用再跑22分钟
# ============================================================
train_clean = pd.read_csv(CLEAN_TRAIN)
test_clean = pd.read_csv(CLEAN_TEST)

train_clean["raw_len"] = train_clean["review"].str.split().str.len()
train_clean["clean_len"] = train_clean["clean_review"].str.split().str.len()

print("训练集：", train_clean.shape)
print("测试集：", test_clean.shape)
print("\n平均词数：原始 %.1f → 处理后 %.1f（减少 %.1f%%）" % (
    train_clean["raw_len"].mean(),
    train_clean["clean_len"].mean(),
    (1 - train_clean["clean_len"].mean() / train_clean["raw_len"].mean()) * 100))
print("空文本条数：", (train_clean["clean_review"].str.len() == 0).sum())

print("\n===== 处理前后对比 =====")
for i in [0, 1]:
    print(f"\n--- 第{i+1}条 ---")
    print("原始：", train_clean["review"].iloc[i][:180])
    print("处理后：", train_clean["clean_review"].iloc[i][:180])

# ---------- 词频分析 ----------
from collections import Counter

all_words = " ".join(train_clean["clean_review"]).split()
cnt = Counter(all_words)
print(f"\n总词数 {len(all_words):,}，唯一词 {len(cnt):,}")
print("\n全量 Top20 高频词：")
display(pd.DataFrame(cnt.most_common(20), columns=["词", "出现次数"]))

pos_words = Counter(" ".join(
    train_clean.loc[train_clean["sentiment"] == 1, "clean_review"]).split())
neg_words = Counter(" ".join(
    train_clean.loc[train_clean["sentiment"] == 0, "clean_review"]).split())

print("\n===== 高频词在两类评论中的差异 =====")
rows = []
for w, c in cnt.most_common(15):
    p, n = pos_words.get(w, 0), neg_words.get(w, 0)
    rows.append({"词": w, "积极": p, "消极": n, "消极/积极倍数": round(n / max(p, 1), 2)})
display(pd.DataFrame(rows))

# In[13]:


# ============================================================
# 模块2-E：处理效果可视化
# ============================================================
plt.close("all")
sns.set_theme(style="whitegrid")
if font_path:
    plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ---------- 图1：处理前后词数分布 ----------
ax = axes[0]
ax.hist(train_clean["raw_len"], bins=50, alpha=0.6, label="原始", color="#9BBBD4")
ax.hist(train_clean["clean_len"], bins=50, alpha=0.7, label="处理后", color="#E76F51")
ax.set_xlabel("词数")
ax.set_ylabel("评论数量")
ax.set_title("处理前后词数分布", loc="left", fontweight="bold")
ax.legend(frameon=False)

# ---------- 图2：积极评论高频词 ----------
ax = axes[1]
top = pos_words.most_common(15)[::-1]
ax.barh([w for w, _ in top], [c for _, c in top], color="#2A9D8F", height=0.65)
ax.set_xlabel("出现次数")
ax.set_title("积极评论高频词 Top15", loc="left", fontweight="bold")

# ---------- 图3：消极评论高频词 ----------
ax = axes[2]
top = neg_words.most_common(15)[::-1]           #取出现次数最多的 15 个词,并倒置
ax.barh([w for w, _ in top], [c for _, c in top], color="#E76F51", height=0.65)     #取出单词与次数,作为横纵轴
ax.set_xlabel("出现次数")
ax.set_title("消极评论高频词 Top15", loc="left", fontweight="bold")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

# In[14]:


# ============================================================
# 模块3-A：文本向量化
# 对比三种文本表示方法：词袋 / TF-IDF
# ============================================================
# 把“人能看懂的文字”变成“机器学习模型能计算的数字”，同时比较词袋和 TF-IDF 哪一种文本表示更适合 Amazon 情感分类。
import os
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB

DATA_DIR = r"D:\训练\data2"

train = pd.read_csv(os.path.join(DATA_DIR, "amazon_train_clean.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "amazon_test_clean.csv"))

X_train_text = train["clean_review"]        #X_train_text → 训练用的评论
y_train = train["sentiment"]                #Y_train → 训练用的答案
X_test_text = test["clean_review"]          #X_test_text → 测试用的评论
y_test = test["sentiment"]                  #Y_test → 测试用的答案

print("训练集：", X_train_text.shape, " 测试集：", X_test_text.shape)

# ---------- 用同一句话演示两种方法的差别 ----------
demo = ["this product is not good",
        "this product is not good and not cheap"]

#词袋模型（Bag of Words）  #统计每个词出现了多少次。
#直接统计每个词出现多少次。
#好处是简单、直观、容易解释，而且计算比较快。
#所以它很适合拿来作为一个基准方法：先看看“仅仅统计词频”能做到什么效果。
cv_demo = CountVectorizer(tokenizer=str.split,          #str.split 就是按空格切分。
                          preprocessor=None,            #str.split 就是按空格切分。
                          token_pattern=None,           #不使用 CountVectorizer 默认的正则分词方式
                          lowercase=False)              #不再自动转小写

#TF-IDF模型   #根据词语在当前评论中的出现频率和在全部评论中的普遍程度，给每个词计算重要性权重，并将文本转换成数字向量
#它不是简单数次数，而是：
#给不同词计算不同的重要性权重。
#好处是能够降低那些在大量评论里都很常见的词的影响，更突出具有区分能力的词，因此通常更适合文本分类
tf_demo = TfidfVectorizer(tokenizer=str.split, preprocessor=None,
                          token_pattern=None, lowercase=False)

cvd = cv_demo.fit_transform(demo).toarray()             #用词袋模型把 demo 里的两句话转换成数字矩阵。
tfd = tf_demo.fit_transform(demo).toarray()             #用TF-IDF模型把 demo 里的两句话转换成数字矩阵。
words = cv_demo.get_feature_names_out()                 #把词袋模型建立出来的“词表”取出来。
print("\n===== 同一句话在两种表示下的差异 =====")
print(f"{'词':14s} {'词袋-句1':>10s} {'TFIDF-句1':>11s} {'词袋-句2':>10s} {'TFIDF-句2':>11s}")

for j, w in enumerate(words):                           #把每个词，以及这个词在两句话中的“词袋值”和“TF-IDF值”打印出来。
    print(f"{w:14s} {cvd[0][j]:10.3f} {tfd[0][j]:11.3f} {cvd[1][j]:10.3f} {tfd[1][j]:11.3f}")

# In[15]:

# ============================================================
# 模块3-B：三种方案全量向量化
# 关键：向量化器只在训练集上 fit，测试集只做 transform
#       否则词表会把测试集的信息带进来（数据泄露）
# ============================================================
#创建空字典
results = {}

def build(vec, name, tag):      #vec  → 用哪一种向量化方法 name → 这个方法叫什么 tag  → 保存文件时使用的简称
    t0 = time.time()
    X_tr = vec.fit_transform(X_train_text)   # 用训练集建立词表，并把训练集转换成数字向量。(学习规则 + 转成数字)
    X_te = vec.transform(X_test_text)        # 按照刚才训练集学到的词表和规则，把测试集转换成数字。(只按照训练集学到的规则转成数字)
    dt = time.time() - t0

    print(f"  {name:20s} 词表 {len(vec.vocabulary_):>7,}   "
          f"非零元素 {X_tr.nnz:>10,}   用时 {dt:5.0f}s")
    results[name] = (X_tr, X_te)            #把这一种方法得到的训练向量和测试向量保存到字典里。

    # 存盘，后续模块直接读，不用重新向量化
    sparse.save_npz(os.path.join(DATA_DIR, f"Xtrain_{tag}.npz"), X_tr)
    sparse.save_npz(os.path.join(DATA_DIR, f"Xtest_{tag}.npz"), X_te)
    joblib.dump(vec, os.path.join(DATA_DIR, f"vec_{tag}.pkl"))
    return X_tr, X_te


print("===== 全量向量化 =====")  #把训练集和测试集里的所有评论，都用选定的文本表示方法转换成数字向量。

#创建一个词袋模型工具
cv = CountVectorizer(tokenizer=str.split,
                     preprocessor=None,
                     token_pattern=None,
                     lowercase=False,
                     max_features=50000)        #最多保留 50000 个词作为特征。
build(cv, "词袋(50000)", "bow")        #把刚创建好的这个词袋工具 cv 交给 build()，去处理整个训练集和测试集。

#TF-IDF 1-gram : 只把一个词作为特征。
tf1 = TfidfVectorizer(tokenizer=str.split, preprocessor=None, token_pattern=None,
                      lowercase=False, max_features=50000,
                      min_df=3, sublinear_tf=True)
build(tf1, "TF-IDF(1gram)", "tf1")

#TF-IDF 1-2gram：既看单个词，又看连续两个词。
tf2 = TfidfVectorizer(tokenizer=str.split, preprocessor=None, token_pattern=None,
                      lowercase=False, ngram_range=(1, 2), max_features=50000,
                      min_df=3, sublinear_tf=True)
Xtr2, Xte2 = build(tf2, "TF-IDF(1-2gram)", "tf2")

# ---------- 稀疏度: 这个数字矩阵里面，有多少比例的位置是 0。 ----------
#选用Xtr2 ,其表示TF-IDF 1-2gram 的训练数据数字矩阵
# 行 = 评论
# 列 = 单词/词组
sp = 100 * (1 - Xtr2.nnz / (Xtr2.shape[0] * Xtr2.shape[1]))
print(f"\nTF-IDF 矩阵稀疏度：{sp:.4f}%（{Xtr2.shape[0]:,} × {Xtr2.shape[1]:,} 的矩阵，"
      f"只有 {Xtr2.nnz:,} 个非零值）")

# In[16]:

# ============================================================
# 模块3-C：用朴素贝叶斯快速验证，确定最终方案(根据词语出现情况，判断这条评论更像积极还是消极的分类方法。)
#朴素贝叶斯的好处: 实现简单、训练和预测速度快、对高维稀疏的文本数据比较适合，而且作为基准模型容易和后面的其他分类器进行比较
# ============================================================
rows = []
for name, (X_tr, X_te) in results.items():         #把三种文本表示方法一个一个拿出来测试。
    m = MultinomialNB()                            #创建朴素贝叶斯模型,根据评论的数字向量，判断评论是 0（消极）还是 1（积极）
    m.fit(X_tr, y_train)                           #X_tr → 训练集的数字向量 , y_train → 训练集的正确答案; 用训练集向量和训练标签训练朴素贝叶斯。
    pred = m.predict(X_te)                         #拿测试集去考试,只有X_te,无y_test
    rows.append({
        "文本表示": name,
        "准确率": accuracy_score(y_test, pred),      #一共预测多少条，有多少条预测正确。
        "精确率": precision_score(y_test, pred),     #模型预测为“积极”的评论里面，到底有多少真的积极？
        "召回率": recall_score(y_test, pred),        #真正的积极评论里面，模型找出来了多少？
        "F1值": f1_score(y_test, pred),             #综合考虑精确率和召回率的一个指标。F1 是精确率（Precision）和召回率（Recall）的调和平均数：F1 = 2 × 精确率 × 召回率 ÷（精确率 + 召回率）
    })

vec_cmp = pd.DataFrame(rows).set_index("文本表示")      #把前面记录的实验结果变成 Pandas 表格，并把“文本表示”作为每一行的名称。
print(vec_cmp.round(4))     #保留四位小数

vec_cmp.round(4).to_csv(os.path.join(DATA_DIR, "vectorizer_comparison.csv"),
                        encoding="utf-8-sig")       #把结果保存为csv文件

#主要目的: 检查 TF-IDF 1-2gram 有没有捕捉到 not good、not work 之类的否定短语，并找出最常见的15个。
# ---------- 看看 bigram (即2-gram)捕获了哪些否定短语 ----------
names = np.array(tf2.get_feature_names_out())           #把 tf2 生成的所有词和词组名称取出来，放到 names 里。
not_idx = np.array([i for i, n in enumerate(names) if n.startswith("not ")])    #从这些特征里，找出以 "not " 开头的那些。然后把它们的位置编号保存到 not_idx。
df_cnt = np.asarray((Xtr2[:, not_idx] > 0).sum(axis=0)).ravel()     #统计每个 not + 词 的 bigram 在训练集中分别出现了多少条评论。
order = np.argsort(-df_cnt)[:15]                #找出出现次数最多的15个否定短语的位置。
not_df = pd.DataFrame({
    "否定短语": names[not_idx[order]],          #找到前15名到底是哪些短语。
    "出现的评论数": df_cnt[order],                #找到它们各自出现了多少次。
})
print(f"\nbigram 共捕获 {len(not_idx)} 个含 'not' 的短语")
print("出现最多的 Top15：")
print(not_df)

# In[17]:


# ============================================================
# 模块3-D：向量化方案对比可视化
# ============================================================
plt.close("all")
sns.set_theme(style="whitegrid")
if font_path:
    plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

# ---------- 图1：三种方案指标对比 ----------
ax = axes[0]
metrics = ["准确率", "精确率", "召回率", "F1值"]
x = np.arange(len(metrics))         #编号: 准确率 → 0 精确率 → 1 召回率 → 2; F1 → 3
w = 0.26
for i, (name, row) in enumerate(vec_cmp.iterrows()):        #遍历文本表示方法
    vals = [row[m] for m in metrics]                        #取出方法的四个指标
    bars = ax.bar(x + (i - 1) * w, vals, w, label=name,
                  color=["#B7C9E2", "#74A0C4", "#E76F51"][i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                ha="center", fontsize=7.5, color="#243B53")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0.78, 0.92)
ax.set_ylabel("得分")
ax.set_title("三种文本表示方案的对比", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=9)

# ---------- 图2：否定短语 ----------
ax = axes[1]
o = np.argsort(df_cnt[order])                   #按照数值大小进行排序，并返回排序后的位置。
ax.barh(not_df["否定短语"].values[o], not_df["出现的评论数"].values[o],
        color="#2A9D8F", height=0.65)
ax.set_xlabel("出现的评论数")
ax.set_title("bigram 捕获的否定短语 Top15", loc="left", fontweight="bold")

# ---------- 图3：TF-IDF 权重分布 ----------
ax = axes[2]
ax.hist(Xtr2[:5000].data, bins=60, color="#9BBBD4")
ax.set_xlabel("TF-IDF 权重")
ax.set_ylabel("特征数量")
ax.set_yscale("log")
ax.set_title("TF-IDF 权重分布（前5000条评论）", loc="left", fontweight="bold")

for ax in axes:                                     #遍历所有子图，把每个子图的上边框和右边框隐藏掉。
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

# In[18]:


# ============================================================
# 模块4-A：多模型建模准备
# 超参数搜索在 6 万条子集上做（全量20万上做太慢）
# 最终模型在全量 20 万上重训
# ============================================================
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

DATA_DIR = r"D:\训练\data2"

# 加载模块3存下的 TF-IDF 矩阵（不用重新向量化）
Xtr = sparse.load_npz(os.path.join(DATA_DIR, "Xtrain_tf2.npz"))
Xte = sparse.load_npz(os.path.join(DATA_DIR, "Xtest_tf2.npz"))
y_tr = pd.read_csv(os.path.join(DATA_DIR, "amazon_train_clean.csv"),
                   usecols=["sentiment"])["sentiment"].values           #y_tr: 训练集的正确答案

y_te = pd.read_csv(os.path.join(DATA_DIR, "amazon_test_clean.csv"),     #y-te: 测试集的正确答案
                   usecols=["sentiment"])["sentiment"].values

print("训练矩阵：", Xtr.shape, " 非零：", f"{Xtr.nnz:,}")
print("测试矩阵：", Xte.shape)
print("训练集标签：", np.bincount(y_tr), "（0=消极, 1=积极）")              #看正负样本数量

# 调参子集：从训练集里抽 6 万条          给模型试不同的设置，看看哪种设置效果更好
#六万条数据用于调参,20万所有数据太多了,故六万调好参数之后再预测20万的数据
N_TUNE = 60_000
X_tune, _, y_tune, _ = train_test_split(                #从 Xtr(训练集的评论数字向量) 和 y_tr(这 20 万条评论对应的正确答案) 中抽出6万条，保存为 X_tune 和 y_tune，剩下的部分不要。
    Xtr, y_tr, train_size=N_TUNE, random_state=42, stratify=y_tr)
print(f"\n调参子集：{X_tune.shape}")

# In[19]:


# ============================================================
# 模块4-B：三个模型的超参数网格搜索
# 评分标准用 F1（准确率在文本任务上同样会掩盖问题） F1 是精确率和召回率的综合指标
# ============================================================
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) #把6万条调参数据分成3份，轮流拿其中一份验证，另外两份训练。

searches = {                                                            #把三个模型以及它们需要尝试的参数范围列出来。
    "朴素贝叶斯": (MultinomialNB(), {"alpha": [0.01, 0.1, 0.5, 1.0]}),   #alpha 可以简单理解成：控制平滑程度的参数，用来避免某些词没出现导致概率变成 0。
    "逻辑回归": (LogisticRegression(max_iter=1000, solver="lbfgs"),
                 {"C": [0.5, 1, 2, 5]}),                                #C 小 → 正则化更强 → 模型更保守  C 大 → 正则化更弱 → 更容易拟合训练数据
    "线性SVM": (LinearSVC(max_iter=2000), {"C": [0.1, 0.5, 1, 2]}),       #控制模型对训练数据拟合程度和正则化程度。
}

best_params = {}                #用来保存每个模型最好的参数。
grid_results = {}               #用来保存每个模型所有参数组合的测试结果。

for name, (est, grid) in searches.items():              #name = 模型名字  est = 模型本身  grid = 要尝试的参数
    t0 = time.time()
    gs = GridSearchCV(est, grid, scoring="f1", cv=cv, n_jobs=-1)        #把参数一个一个试过去，并用交叉验证比较 F1，找出最好的参数。
    gs.fit(X_tune, y_tune)          #真正开始调参

    best_params[name] = gs.best_params_     #保存最佳参数
    dt = time.time() - t0

    print(f"\n===== {name} =====")
    print(f"最优参数：{gs.best_params_}    验证集F1：{gs.best_score_:.4f}    用时 {dt:.0f} 秒")

    res = pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "std_test_score"]]   #把所有参数组合的结果取出来,参数分别是: 参数名,平均F1,F1标准差
    res = res.sort_values("mean_test_score", ascending=False).round(4)   #把所有参数组合按照 F1 从高到低排列，并保留4位小数。
    grid_results[name] = res        #保存
    print(res)

# In[20]:


# ============================================================
# 模块4-C：用最优参数在全量训练集上重训，并在测试集评估
# ============================================================

def evaluate(model, name, X_train=Xtr, y_train=y_tr, note=""):
    """
    model：要测试的模型，比如朴素贝叶斯、逻辑回归、SVM
    name：模型名字
    X_train：训练数据，默认就是 Xtr
    y_train：训练标签，默认就是 y_tr
    note：备注，可写一些说明"""

    t0 = time.time()
    model.fit(X_train, y_train)     #让模型学习
    fit_time = time.time() - t0

    t1 = time.time()
    pred = model.predict(Xte)       #让训练好的模型去预测测试集
    pred_time = time.time() - t1

    row = {
        "模型": name,
        "训练耗时(s)": round(fit_time, 1),
        "预测耗时(s)": round(pred_time, 2),
        "训练样本数": X_train.shape[0],
        "准确率": accuracy_score(y_te, pred),
        "精确率": precision_score(y_te, pred),
        "召回率": recall_score(y_te, pred),
        "F1值": f1_score(y_te, pred),
        "说明": note,
    }
    print(f"  {name:22s} 训练 {fit_time:6.1f}s  |  准确率 {row['准确率']:.4f}  "
          f"F1 {row['F1值']:.4f}", flush=True)
    return row, model, pred
    """
    row：这个模型的成绩
    model：训练好的模型
    pred：模型对测试集的预测结果
    """

print("===== 全量 20 万条训练 =====")
rows = []           #存三个模型的成绩       row
fitted = {}         #存训练好的三个模型      m
preds = {}          #存三个模型对测试集的预测结果     p  (均为接受返回值的变量)
                                    # **表示把字典里的 键:值 自动拆开，作为函数的关键字参数传进去。
row, m, p = evaluate(MultinomialNB(**best_params["朴素贝叶斯"]), "朴素贝叶斯")    #创建一个使用最佳参数的朴素贝叶斯模型。
rows.append(row); fitted["朴素贝叶斯"] = m; preds["朴素贝叶斯"] = p

row, m, p = evaluate(LogisticRegression(max_iter=1000, solver="lbfgs",
                                        **best_params["逻辑回归"]), "逻辑回归")
rows.append(row); fitted["逻辑回归"] = m; preds["逻辑回归"] = p

row, m, p = evaluate(LinearSVC(max_iter=2000, **best_params["线性SVM"]), "线性SVM")
rows.append(row); fitted["线性SVM"] = m; preds["线性SVM"] = p

# ---------- 多层感知器：计算量大，用 6 万条子集 ----------
# 额外加入一个神经网络模型——多层感知器（MLP）——来比较。
print("\n===== 多层感知器（受算力限制，用 6 万条子集）=====")
print("训练中，约需 3 分钟...")

row, m, p = evaluate(
    MLPClassifier(hidden_layer_sizes=(50,), max_iter=15, batch_size=512,        #创建一个多层感知器模型。
                  early_stopping=True, n_iter_no_change=3, random_state=42),
    "多层感知器",
    X_train=X_tune, y_train=y_tune,             #这次不要用默认的20万条训练数据，而是用之前抽出来的6万条。
    note="受算力限制，仅用6万条子集训练")
rows.append(row); fitted["多层感知器"] = m; preds["多层感知器"] = p

# ---------- 公平对照：逻辑回归也在 6 万条上训一版 ----------
#由于 MLP 受计算量限制只使用 6 万条训练数据，因此额外训练一个同样使用 6 万条数据的逻辑回归作为对照，以减少训练样本数量差异带来的影响。
row, m, p = evaluate(
    LogisticRegression(max_iter=1000, solver="lbfgs", **best_params["逻辑回归"]),
    "逻辑回归(6万条对照)",
    X_train=X_tune, y_train=y_tune,
    note="6万条子集，用于与多层感知器公平对比")
rows.append(row); fitted["逻辑回归(6万条对照)"] = m; preds["逻辑回归(6万条对照)"] = p

# ---------- 对比表 ----------
result_df = pd.DataFrame(rows).set_index("模型").sort_values("F1值", ascending=False)
print("\n===== 模型效果对比 =====")
display(result_df.round(4))

result_df.round(4).to_csv(os.path.join(DATA_DIR, "model_comparison_amazon.csv"),
                          encoding="utf-8-sig")
print("\n已保存：model_comparison_amazon.csv")

# In[21]:


# ============================================================
# 模块4-D：最优模型详细评估 + 可视化
# ============================================================
best_name = result_df.index[0]                  #找出最好的模型
pred = preds[best_name]                         #把最佳模型之前保存好的预测结果取出来。
print(f"表现最好的模型：{best_name}\n")
print(classification_report(y_te, pred, target_names=["消极", "积极"], digits=4))   #一次性把最佳模型的分类评价指标详细打印出来

cm = confusion_matrix(y_te, pred)
print("混淆矩阵：")
print("                预测消极   预测积极")
print(f"实际消极        {cm[0][0]:7,}  {cm[0][1]:8,}")              #真正是消极评论的，有多少被预测成消极，有多少被预测成积极。
print(f"实际积极        {cm[1][0]:7,}  {cm[1][1]:8,}")              #真正是积极评论的，有多少被预测成消极，有多少被预测成积极。

# ---------- 可视化 ----------
plt.close("all")
sns.set_theme(style="whitegrid")
if font_path:
    plt.rcParams["font.family"] = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))

# 图1：指标对比
ax = axes[0]
metrics = ["准确率", "精确率", "召回率", "F1值"]          #指定4个评价指标
x = np.arange(len(metrics))                            #给4个指标设置横轴位置
w = 0.16
colors = ["#B7C9E2", "#74A0C4", "#2364AA", "#E76F51", "#F4A261"]
for i, (name, r) in enumerate(result_df.iterrows()):        #把 result_df 里的五个模型一个一个取出来。
    vals = [r[m] for m in metrics]                          #取当前模型的4个指标
    bars = ax.bar(x + (i - 2) * w, vals, w, label=name, color=colors[i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                ha="center", fontsize=6.5, color="#243B53")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0.85, 0.94)
ax.set_ylabel("得分")
ax.set_title("五个模型/配置的指标对比", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=8)

# 图2：耗时对比（对数坐标）
ax = axes[1]
names = list(result_df.index)                           #取出5个模型名称
times = result_df["训练耗时(s)"].values                  #从结果表中取出每个模型的训练耗时。
bars = ax.barh(names, times, color="#9BBBD4", height=0.6)
bars[-1].set_color("#E76F51")
for b, v in zip(bars, times):
    ax.text(v * 1.15, b.get_y() + b.get_height() / 2, f"{v:.1f}s",
            va="center", fontsize=9)
ax.set_xscale("log")            #横轴使用对数坐标，而不是普通坐标。(原因:训练时间差距很大，所以用对数坐标让图更好看、更容易比较。)
ax.set_xlabel("训练耗时（秒，对数坐标）")
ax.set_title("训练耗时对比", loc="left", fontweight="bold")

# 图3：混淆矩阵
ax = axes[2]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            annot_kws={"size": 13},
            xticklabels=["预测消极", "预测积极"],
            yticklabels=["实际消极", "实际积极"])
ax.set_title(f"{best_name} 混淆矩阵", loc="left", fontweight="bold")
for lab in ax.get_yticklabels():
    lab.set_rotation(0)

for ax in axes[:2]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

# In[22]:


# ============================================================
# 统一绘图样式设置（放在 notebook 靠前位置，全篇生效）
# ============================================================
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = {
    "primary": "#2E5AAC",        # 主色：深蓝
    "primary_light": "#A8C0E8",  # 主色浅版
    "accent": "#E4572E",         # 强调色：橙红
    "accent_light": "#F6B39B",
    "positive": "#2A9D8F",       # 积极/正面
    "negative": "#E76F51",       # 消极/负面
    "muted": "#9AA5B1",
    "grid": "#E8ECF1",
    "text": "#1F2933",
    "subtext": "#616E7C",
}


def set_style():
    """统一全局绘图样式"""
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": fm.FontProperties(fname=font_path).get_name(),
        "axes.unicode_minus": False,
        "figure.dpi": 150, "savefig.dpi": 150,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#D6DCE5", "axes.linewidth": 0.9,
        "axes.labelcolor": PALETTE["text"], "axes.labelsize": 11,
        "axes.titlesize": 13.5, "axes.titleweight": "bold",
        "xtick.color": PALETTE["subtext"], "ytick.color": PALETTE["subtext"],
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "grid.color": PALETTE["grid"], "grid.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": 10,
    })


def clean_axes(ax, grid_axis="y"):
    """去掉多余边框，网格线淡化"""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#D6DCE5")
    if grid_axis:
        ax.grid(axis=grid_axis, color=PALETTE["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def titles(ax, title, subtitle=None):
    """标题 + 灰色副标题（副标题写结论，不是描述）"""
    ax.set_title("")
    ax.text(0, 1.115 if subtitle else 1.04, title, transform=ax.transAxes,
            fontsize=13.5, fontweight="bold", color=PALETTE["text"], va="bottom")
    if subtitle:
        ax.text(0, 1.035, subtitle, transform=ax.transAxes,
                fontsize=9.5, color=PALETTE["subtext"], va="bottom")


set_style()
print("绘图样式已设置")

# In[23]:


# ============================================================
# 模块5-A：复杂语言现象——情感词典为什么会失效(演示复杂语言现象为什么容易让基于词汇的情感判断失效)
# ============================================================
import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

print("===== VADER 情感词典的表现 =====")
cases = [
    ("正面", "This product is amazing! I love it."),
    ("负面", "Terrible. Broke after two days. Do not buy."),
    ("反讽", "Oh great, another product that doesn't work. Just what I needed."),
    ("反讽", "Well done, it broke on day one. Brilliant design!"),
    ("让步", "Works fine, but a little wobbly and it does not fit all legs."),
    ("否定", "not good"),
]
for tag, s in cases:
    sc = sia.polarity_scores(s)
    verdict = "判为积极" if sc["compound"] > 0 else "判为消极"          #一个综合情感分数。
    print(f"[{tag}] compound={sc['compound']:+.3f}  {verdict:6s} | {s}")

# In[24]:

# ============================================================
# 模块5-B：构建情感词典特征 + 句子结构特征
# ============================================================
import os
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler

DATA_DIR = r"D:\训练\data2"
FEAT_TR = os.path.join(DATA_DIR, "text_features_train.csv")
FEAT_TE = os.path.join(DATA_DIR, "text_features_test.csv")


def structure_features(texts):
    """句子结构特征：长度、标点、大写比例"""
    out = []
    for t in texts:
        n_char = len(t)
        words = t.split()
        n_word = len(words)
        out.append({
            "n_char": n_char,                           #字符数
            "n_word": n_word,                           #单词数
            "avg_word_len": n_char / max(n_word, 1),    #平均单词个数
            "n_exclaim": t.count("!"),
            "n_question": t.count("?"),
            "upper_ratio": sum(1 for c in t if c.isupper()) / max(n_char, 1),       #大写比例
            "n_digit": sum(1 for c in t if c.isdigit()),            #数字数量
        })
    return pd.DataFrame(out)


def build_features(df, path, name):
    """ df   → 要处理的数据表
        path → 特征保存到哪里
        name → 这个任务叫什么名字 """
    # 有缓存就直接读，避免重复计算
    if os.path.exists(path):
        f = pd.read_csv(path)
        print(f"[{name}] 从缓存读取：{f.shape}")
        return f

    t0 = time.time()
    print(f"[{name}] 计算 VADER 情感分（较慢）...")
    v = pd.DataFrame(df["review"].apply(lambda t: sia.polarity_scores(t)).tolist())     #把每条评论交给 VADER，算出情感分数，然后整理成表格。
    v.columns = [f"vader_{c}" for c in v.columns]           #给 VADER 的列改名字,前面加上vader_

    s = structure_features(df["review"])
    f = pd.concat([v, s], axis=1)
    f.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[{name}] 完成 {f.shape}，用时 {time.time()-t0:.0f}s")
    return f


train = pd.read_csv(os.path.join(DATA_DIR, "amazon_train_clean.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "amazon_test_clean.csv"))

feat_tr = build_features(train, FEAT_TR, "训练集")
feat_te = build_features(test, FEAT_TE, "测试集")

print("\n附加特征列表：", list(feat_tr.columns))
display(feat_tr.describe().round(3))

# In[25]:


# ============================================================
# 模块5-C：附加特征的效果对比
# 关键：附加特征必须先标准化，否则尺度差异会干扰模型
# ============================================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

Xtr2 = sparse.load_npz(os.path.join(DATA_DIR, "Xtrain_tf2.npz"))
Xte2 = sparse.load_npz(os.path.join(DATA_DIR, "Xtest_tf2.npz"))
y_tr = train["sentiment"].values
y_te = test["sentiment"].values

# 标准化：VADER 分数是 0~1，字符数是 100~1015，不统一尺度会互相干扰
scaler = StandardScaler()
extra_tr = scaler.fit_transform(feat_tr.values)         #先根据训练集计算平均值、标准差，再完成标准化。
extra_te = scaler.transform(feat_te.values)             #直接使用训练集算出来的平均值和标准差进行标准化。不能重新根据测试集计算自己的平均值和标准差，否则就相当于让测试集参与“学习”。

#把附加特征分成两组：VADER 特征 和 句子结构特征。
#找出 feat_tr 里面所有以 vader_ 开头的列，并记录它们的列编号。
vader_cols = [i for i, c in enumerate(feat_tr.columns) if c.startswith("vader_")]

#不是 VADER 的列，就归到结构特征。
struct_cols = [i for i, c in enumerate(feat_tr.columns) if not c.startswith("vader_")]

#combine() 的作用就是把 TF-IDF 1-2gram 特征和指定的附加特征横向拼接起来，形成新的模型输入。
def combine(idx=None):
    a = extra_tr if idx is None else extra_tr[:, idx]       #这次准备加入训练集的附加特征
    b = extra_te if idx is None else extra_te[:, idx]       #测试集附加特征。
    return (sparse.hstack([Xtr2, sparse.csr_matrix(a)], format="csr"),      #也就是把两张表按列拼起来,同时看到词语信息和额外信息。
            sparse.hstack([Xte2, sparse.csr_matrix(b)], format="csr"))


print("===== 改进前后对比（逻辑回归 C=1，参数保持一致）=====")
#给 TF-IDF 增加 VADER 和结构特征，能不能让分类效果变好？
rows = []
for name, idx, use_base in [
    ("仅有TF-IDF", None, True),                   #① 只有 TF-IDF
    ("+ VADER情感词典(4维)", vader_cols, False),   #② TF-IDF + VADER
    ("+ 结构特征(7维)", struct_cols, False),       #③ TF-IDF + 结构特征
    ("+ 全部附加特征(11维)", None, False),          #④ TF-IDF + VADER + 结构特征
]:

    A, B = (Xtr2, Xte2) if use_base else combine(idx)

    t0 = time.time()
    m = LogisticRegression(C=1, max_iter=1000, solver="lbfgs")
    m.fit(A, y_tr)
    dt = time.time() - t0

    pred = m.predict(B)
    rows.append({
        "特征方案": name,
        "特征数": A.shape[1],
        "训练耗时(s)": round(dt, 1),
        "准确率": accuracy_score(y_te, pred),
        "精确率": precision_score(y_te, pred),
        "召回率": recall_score(y_te, pred),
        "F1值": f1_score(y_te, pred),
    })
    print(f"  {name:24s} F1 {rows[-1]['F1值']:.4f}   用时 {dt:6.1f}s", flush=True)

imp_df = pd.DataFrame(rows).set_index("特征方案")
display(imp_df.round(4))

base_f1 = imp_df.loc["仅有TF-IDF", "F1值"]
print("\n===== 相对基线的变化 =====")
for name, r in imp_df.iterrows():
    print(f"  {name:24s} {r['F1值'] - base_f1:+.4f}")

imp_df.round(4).to_csv(os.path.join(DATA_DIR, "feature_improvement.csv"),
                       encoding="utf-8-sig")

# In[26]:


# ============================================================
# 模块5-D：复杂语言现象的深入分析
# ============================================================
m = LogisticRegression(C=1, max_iter=1000, solver="lbfgs")
m.fit(Xtr2, y_tr)
pred = m.predict(Xte2)
proba = m.predict_proba(Xte2)[:, 1]

# ---------- 1. VADER 失效样本上模型的表现 ----------
vader_pred = (feat_te["vader_compound"].values > 0).astype(int)
disagree = vader_pred != y_te

print("===== 1. 情感词典失效的影响 =====")
print(f"VADER 单独使用准确率：{(vader_pred == y_te).mean()*100:.2f}%")
print(f"逻辑回归准确率：{(pred == y_te).mean()*100:.2f}%")
print(f"\nVADER 判断错误的样本：{disagree.sum():,} 条（{disagree.mean()*100:.2f}%）")
print(f"  → 模型在这些样本上准确率：{(pred[disagree] == y_te[disagree]).mean()*100:.2f}%")
print(f"  → 模型在其余样本上准确率：{(pred[~disagree] == y_te[~disagree]).mean()*100:.2f}%")

# ---------- 2. 疑似标签噪声 ----------
conf = np.where(y_te == 0, proba, 1 - proba)
wrong = pred != y_te
print(f"\n===== 2. 疑似标签噪声 =====")
print(f"模型出错且置信度 > 95%：{((conf > 0.95) & wrong).sum():,} 条")
print(f"模型出错且置信度 > 99%：{((conf > 0.99) & wrong).sum():,} 条")

noise = test.loc[(conf > 0.95) & wrong, ["sentiment", "review"]].head(3)
for _, r in noise.iterrows():
    tag = "数据集标为消极" if r["sentiment"] == 0 else "数据集标为积极"
    print(f"\n[{tag}，但模型高度确信相反]")
    print(f"  {r['review'][:180]}")

# ---------- 3. 哪些表达方式最难 ----------
res = test[["sentiment", "review"]].copy()
res["correct"] = pred == y_te

pats = {
    "含让步转折(but/however)": r"\b(but|however|although|though)\b",
    "含引号包裹的评价": r"['\"][^'\"]{3,40}['\"]",
    "含全部大写词(≥3字母)": r"\b[A-Z]{3,}\b",
    "含否定+正面词": r"\bnot\s+\w*\s*(good|great|nice|work|recommend)",
    "含感叹号": r"!",
}
print("\n===== 3. 不同表达方式的准确率差异 =====")
overall = res["correct"].mean() * 100
pattern_rows = []
for name, p in pats.items():
    mask = res["review"].str.contains(p, case=True, regex=True)
    if mask.sum() > 100:
        a1 = res.loc[mask, "correct"].mean() * 100
        pattern_rows.append({"表达方式": name, "命中条数": int(mask.sum()),
                             "准确率": round(a1, 2), "相对整体偏差": round(a1 - overall, 2)})
        print(f"  {name:26s} 命中 {mask.sum():>6,} 条  准确率 {a1:5.2f}%  "
              f"（整体 {overall:.2f}%，偏差 {a1-overall:+.2f}）")

pattern_df = pd.DataFrame(pattern_rows).set_index("表达方式")
pattern_df.to_csv(os.path.join(DATA_DIR, "language_patterns.csv"), encoding="utf-8-sig")

# ---------- 4. 让步转折样本的典型错误 ----------
mask = res["review"].str.contains(r"\b(but|however|although|though)\b",
                                  case=False, regex=True)
print(f"\n===== 4. 让步转折样本 =====")
print(f"平均长度：{res.loc[mask, 'review'].str.len().mean():.0f} 字符 "
      f"（其他样本 {res.loc[~mask, 'review'].str.len().mean():.0f} 字符）")
print("\n典型判断错误：")
for _, r in res[mask & ~res["correct"]].head(3).iterrows():
    tag = "实际消极" if r["sentiment"] == 0 else "实际积极"
    print(f"\n[{tag}] {r['review'][:200]}")

# In[ ]:


# ============================================================
# 模块5-E：改进效果与难点可视化
# ============================================================
set_style()
plt.close("all")

fig = plt.figure(figsize=(19, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1.25], wspace=0.32)

# ---------- 图1：附加特征效果 ----------
ax = fig.add_subplot(gs[0])
df = imp_df.sort_values("F1值")
base = imp_df.loc["仅有TF-IDF", "F1值"]
colors = [PALETTE["primary"] if v >= base else PALETTE["accent_light"] for v in df["F1值"]]
bars = ax.barh(df.index, df["F1值"], color=colors, height=0.6)
ax.axvline(base, color=PALETTE["muted"], linestyle="--", linewidth=1.4, zorder=0)
ax.text(base, len(df) - 0.35, f" 基线 {base:.4f}", color=PALETTE["subtext"],
        fontsize=9.5, va="center")
for b, (name, r) in zip(bars, df.iterrows()):
    ax.text(r["F1值"] + 0.0003, b.get_y() + b.get_height() / 2,
            f"{r['F1值']:.4f}  ({r['F1值'] - base:+.4f})", va="center",
            fontsize=9.5, fontweight="bold", color=PALETTE["text"])
ax.set_xlim(0.905, 0.9215)
ax.set_xlabel("F1 值")
clean_axes(ax, grid_axis="x")
titles(ax, "附加特征的效果对比", "只有结构特征带来提升，情感词典反而拖累")

# ---------- 图2：词典失效样本上的表现 ----------
ax = fig.add_subplot(gs[1])
cats = [f"VADER判断正确\n({(1-disagree.mean())*100:.1f}%)",
        f"VADER判断错误\n({disagree.mean()*100:.1f}%)"]
vals = [(pred[~disagree] == y_te[~disagree]).mean() * 100,
        (pred[disagree] == y_te[disagree]).mean() * 100]
bars = ax.bar(cats, vals, color=[PALETTE["positive"], PALETTE["negative"]], width=0.52)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.2f}%",
            ha="center", fontsize=11.5, fontweight="bold", color=PALETTE["text"])
ax.set_ylim(75, 98)
ax.set_ylabel("模型准确率")
clean_axes(ax)
titles(ax, "复杂表达是主要失分点", "模型在词典失效样本上掉了12个百分点")

# ---------- 图3：不同表达方式的难度 ----------
ax = fig.add_subplot(gs[2])
d = pattern_df.sort_values("相对整体偏差")
colors = [PALETTE["negative"] if v < 0 else PALETTE["positive"]
          for v in d["相对整体偏差"]]
bars = ax.barh(d.index, d["相对整体偏差"], color=colors, height=0.6)
ax.axvline(0, color="#B0B8C4", linewidth=1.2)
for b, v in zip(bars, d["相对整体偏差"]):
    off = 0.14 if v > 0 else -0.14
    ax.text(v + off, b.get_y() + b.get_height() / 2, f"{v:+.2f}%",
            va="center", ha="left" if v > 0 else "right",
            fontsize=9.5, fontweight="bold", color=PALETTE["text"])
ax.set_xlabel("相对整体准确率的偏差（百分点）")
clean_axes(ax, grid_axis="x")
titles(ax, "哪些表达方式最难判断", "让步转折影响最大：拉低3.72个百分点")

plt.tight_layout()
plt.show()
