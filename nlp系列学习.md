https://blog.csdn.net/jclian91/article/details/83617898?spm=1001.2014.3001.5502

# 一、词袋模型
## 1、思路
- 词袋模型的目的是将文本句子转为向量表示
- 1）将所有数据进行分词
- 2）将所有句子的所有词收集起来，去重，得到一个大字典V
- 3）输入一句话，分词后，进行数据统计，方法为：把大字典V看成一个大向量，如果输入的句子，在字典的某个位置有词，统计个数，如果在字典的某个位置没有词，个数就是0，也就得到了句子的向量表示，向量的长度就是大词典的长度
## 2、实践
- 1）先用分词工具将句子分词
```python
# 【1】、中文分词用jieba、英文分词用nltk的word_tokenize
sent = "I like running, I love reading."
en_texts = word_tokenize(sent) #-->['I', 'like', 'running', ',', 'I', 'love', 'reading', '.']
sent = '你好，这是nlp入门学习系统'
zh_text = jieba.lcut(sent)  # --> ['你好', '，', '这是', 'nlp', '入门', '学习', '系统']

sent1 = "I love sky, I love sea."
sent2 = "I like running, I love reading."
from nltk import word_tokenize
sents = [sent1, sent2]
texts = [[word for word in word_tokenize(sent)] for sent in sents] # [['I', 'love', 'sky', ',', 'I', 'love', 'sea', '.'], ['I', 'like', 'running', ',', 'I', 'love', 'reading', '.']]

# 【2】、上面得到了所有句子的所有词，接下来进行去重，得到大词典V
all_list = []
for text in texts:
    all_list += text
corpus = set(all_list) # {'love', 'running', 'reading', 'sky', '.', 'I', 'like', 'sea', ','}

# 【3】、将大词典进行数字映射
corpus_dict = dict(zip(corpus, range(len(corpus)))) # {'running': 1, 'reading': 2, 'love': 0, 'sky': 3, '.': 4, 'I': 5, 'like': 6, 'sea': 7, ',': 8}

# 【4】、将句子转为向量表示：
# 建立句子的向量表示
def vector_rep(text, corpus_dict):
    vec = []
    for key in corpus_dict.keys(): # 遍历大词典，统计大词典中的每个位置中，句子存在这个词的个数（没有这个词就为0）
        if key in text:
            vec.append((corpus_dict[key], text.count(key)))
        else:
            vec.append((corpus_dict[key], 0))
    vec = sorted(vec, key= lambda x: x[0])
    return vec
vec1 = vector_rep(texts[0], corpus_dict)
# [(0, 2), (1, 0), (2, 0), (3, 1), (4, 1), (5, 2), (6, 0), (7, 1), (8, 1)]
# 得到句子的向量表示：[2, 0, 0, 1, 1, 2, 0, 1, 1]，这个输入的所有句子都有统一长度（大词典的大小）的向量表示
```
## 3、现成工具
```python
# texts: [['I', 'love', 'sky', ',', 'I', 'love', 'sea', '.'], ['I', 'like', 'running', ',', 'I', 'love', 'reading', '.']]
dictionary = corpora.Dictionary(texts)
# 利用doc2bow作为词袋模型
corpus = [dictionary.doc2bow(text) for text in texts]
print("corpus:",corpus) # corpus: [[(0, 1), (1, 1), (2, 2), (3, 2), (4, 1), (5, 1)], [(0, 1), (1, 1), (2, 2), (3, 1), (6, 1), (7, 1), (8, 1)]]
```

# 二、句子相似度
## 1、思路
- 有了句子的向量表示，再将数据输入到余弦相似度，计算内积，即得到了句子之间的相似度值

## 2、实践
```python
from math import sqrt
def similarity_with_2_sents(vec1, vec2):
    inner_product = 0
    square_length_vec1 = 0
    square_length_vec2 = 0
    for tup1, tup2 in zip(vec1, vec2):
        inner_product += tup1[1]*tup2[1]
        square_length_vec1 += tup1[1]**2
        square_length_vec2 += tup2[1]**2
    return (inner_product/sqrt(square_length_vec1*square_length_vec2))
cosine_sim = similarity_with_2_sents(vec1, vec2)
print('两个句子的余弦相似度为： %.4f。'%cosine_sim)
```
## 3、现成工具包
```python
similarity = Similarity('-Similarity-index', corpus, num_features=len(dictionary))
new_sensence = sent1
corpus1 = dictionary.doc2bow(texts[0])  # texts中有两句话，这里取了其中的一句
print("sim:",similarity[corpus1]) # sim: [0.99999994 0.73029673]，这里因为similarity里面有两个句子，所有有两个相似度，而里面的第一个句子跟现在输入的句子是同句话，所有相似度很高：0.9999
```

# 三、TF-IDF
## 1、思路
- tf-idf是统计方法，评估一个字词对于文件集或者语料库的重要程度，用于提取文本的特征。字词的重要性，随着它在文档中出现的次数成正比增加，随着它在语料库中出现的频率（在很多资料库中都出现）而反比下降。
- tf为词频，即字词在一篇文档中出现的次数，假设文档总共有N个词，词“I”出现了n词，则tf = n / N
- idf为逆文档频率，假设有M篇文档，词'I'在m篇文档中都出现，则idf=log2（M/m),有时在分母加1进行平滑，防止分母为0。
- 通过计算出每个字词的tf-idf值，进行排序过后，就可以得到这个文档的关键词了

# 四、词形还原
## 1、基本概念
- 所谓词形还原就是去掉单词的词缀，还原出单词的原始形态。比如cars-->car,ate-->eat.

## 2、实践
```python
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
#这里传入单词以及单词的词性
print(wnl.lemmatize('cars','n'))
```
## 3、将句子进行词形还原
```python
from nltk import word_tokenize,pos_tag
from nltk.corpus import wordnet
def get_wordnet_pos(tag):
    if tag.startswith('J') : return wordnet.ADJ
    if tag.startswith('V') : return wordnet.VERB
    if tag.startswith('N') : return wordnet.NOUN
    if tag.startswith('R') : return wordnet.ADV
    return None

# 将整个句子的进行词形还原
sentence = 'football is a family of team sports that involve, to varying degrees, kicking a ball to score a goal.'
tokens = word_tokenize(sentence)
tagged_sent = pos_tag(tokens)  # 利用pos_tag可以进行词性还原 
# [(‘The’, ‘DT’), (‘brown’, ‘JJ’), (‘fox’, ‘NN’), (‘is’, ‘VBZ’), (‘quick’, ‘JJ’), (‘and’, ‘CC’), (‘he’, ‘PRP’), (‘is’, ‘VBZ’), (‘jumping’, ‘VBG’), (‘over’, ‘IN’), (‘the’, ‘DT’), (‘lazy’, ‘JJ’), (‘dog’, ‘NN’)]

wnl = WordNetLemmatizer()
lemmas_sent = []
for tag in tagged_sent:
    wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
    lemmas_sent.append(wnl.lemmatize(tag[0],wordnet_pos))
print(lemmas_sent)
```


# 四、命名实体识别
## 1、基础概念
- 识别出具体的识别，包括三大类（实体类、实践类、数字类）和七小类（人名、机构名、地名、实践、日期、货币、百分比）
- 举个例子：小明早上8点去学校上课-->人名：小明；时间：早上8点；地点：学校。
