


''' 31、短语的语序问题======================================================='''
from itertools import permutations  # 排列函数
from operator import itemgetter  # 获取可迭代对象某个位置的值
from tqdm import tqdm
def short_sentence_re_order(text:str='德景陶镇瓷'):
    """
    通过2-gram，进行短语的重新排列，得到概率最大的排列
    原理：
    1）加载语料库（可更换），将语料库当成一个巨大的文本content
    2）对输入的词进行去重，得到每一个字符，然后将字符进行全排序组合，得到所有可能的组合结果，后续计算每个组合的概率
    3）计算每个组合的概率：p = p1*p2*p3...pn，其中，
        p1 = count(W1) / len(content),即第一个字在语料库中的频率
        p2 = count(W1W2) / count(W1),即第1,2个字同时出现的次数除以第一个字出现的次数，也就是第一个字出现的情况下，第二个字也出现的概率
        p3 = count(W2W3) / count(W2)同理
        pn = count(W(n-1)Wn) / count(W(n-1))同理
    4）得到每个组合下的p,接下来排序比较大小即可，得到最大概率出现的短语组合了
    :param text:str: 输入的短语
    :return：str: 返回最大概率的短语结果
    """
    data_path = r'G:\知识萃取\NLP\系统学习\NLP_BASE\data\edu.zh'

    with open(data_path,'r',encoding='utf-8') as f:
        content = [data.strip() for data in f.readlines()]

    word_list = list(text)
    print("模拟输入：",word_list)

    candidata_list = set(list(permutations(word_list, r = len(word_list))))
    # print("全排列：",candidata_list)

    print("判断数据在不在预料中")
    for word in word_list:
        if word not in ''.join(content):
            print(f"{word}不在语料库中")
            return
            # raise Exception(f"{word}不在语料库中")

    word_prob_dict = {}

    all_content = ''.join(content)
    len_content = len(all_content)

    word_in_content_prob_dict = {word:all_content.count(word) / len_content for word in word_list}

    for candidata in tqdm(candidata_list):
        candidata = list(candidata)
        prob = word_in_content_prob_dict.get(candidata[0])  # 得到每个首字的频率
        # 2-gram
        # p = count(w_(i-1)w_(i)) / count(w_(i-1))
        # p1*p2*p3....pn
        for i in range(1,len(candidata)):
            char_cnt = word_in_content_prob_dict.get(candidata[i-1])
            word_cnt = all_content.count(''.join(candidata[i-1:i+1]))
            prob *= (word_cnt / char_cnt)
            if prob == 0: break

        word_prob_dict[''.join(candidata)] = prob  # 得到每个短语的概率

    res = sorted(word_prob_dict.items(),key=itemgetter(1),reverse=True)

    print('最终的结果:',res[0][0])
    return res[0][0]



''' 48、文本纠错之获取形近字==================================================='''
import pygame
import json
import cv2
import os
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
from operator import itemgetter
pygame.init()


def get_words_pic():
    '''step1：将汉字转为图片保存'''
    word_path = r"G:\知识萃取\NLP\系统学习\NLP_BASE\data\static\汉字-拼音.json"
    with open(word_path,'r',encoding='utf-8') as file:
        word_list = json.load(file)
    word_list = set([word.strip() for word in list(word_list.keys()) if len(word.strip())==1])
    # 通过pygame将汉字转化为黑白图片
    for char in tqdm(word_list):
        try:
            font = pygame.font.Font("C:\Windows\Fonts\simkai.ttf", 100)
            rtext = font.render(char, True, (0, 0, 0), (255, 255, 255))
            save_path = f"../data/word_pic/{char}.png"
            pygame.image.save(rtext,save_path)
        except Exception as e:
            print('error:',e)
            continue

def pic2vector(pic_path):
    """step2.1:将一张图片转为向量表示"""
    # 读取图片
    img = cv2.imdecode(np.fromfile(pic_path,dtype=np.uint8), -1)
    # 将图片转为灰度模式
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).reshape(-1, 1)
    return [_[0] for _ in img.tolist()]


def get_all_word_vectors():
    '''step2.2:获取所有字的向量表示'''
    dir_path = '../data/word_pic/'
    pic_path_list = [os.path.join(dir_path,pic) for pic in os.listdir(dir_path) if pic.endswith('png')]
    # pic_path_list = pic_path_list[:10]
    pic_vector_dict = {} # 每个汉字对应的向量表示
    for pic_path in tqdm(pic_path_list):
        pic_vector_dict[pic_path.split('.png')[0][-1]] = pic2vector(pic_path)

    with open('../data/word_vectors.json','w',encoding='utf-8') as file:
        json.dump(pic_vector_dict,file,ensure_ascii=False)


# def cal_cos_similary(vec1:list,vec2:list):
#     '''step3:计算两个向量的余弦相似度,范围[-1,1]'''
#     return 1 - cosine(vec2, vec2)

# 计算两个向量之间的余弦相似度
def cal_cos_similary(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return dot_product / ((normA ** 0.5) * (normB ** 0.5))


def get_shape_similary_word(input_word='国',top_n=10):
    """
    原理：计算两个汉字的形近字，通过获取汉字的图片的向量表示，再利用余弦相似度计算两个汉字的相似度
    step1：将汉字转为图片保存
    step2: 获取每个汉字的向量表示
    step3：余弦相似度计算汉字的相似度
    :return:
    """
    with open('../data/word_vectors.json','r',encoding='utf-8') as file:
        pic_vector_dict = json.load(file)
    print(f"获取前{top_n}个形近字")
    input_vec = pic_vector_dict.get(input_word,None)
    if not input_vec:
        print(f"数据中不存在{input_word}字")
        return []
    similary_dict = {word:cal_cos_similary(input_vec,word_vec) for word,word_vec in tqdm(pic_vector_dict.items())}
    similary_dict = sorted(similary_dict.items(),key=itemgetter(1),reverse=True)[:top_n+1]
    similary_list = [word[0] for word in similary_dict]
    if input_word in similary_list: similary_list.remove(input_word)
    print(similary_list)
    return similary_list

if __name__ == '__main__':
    get_shape_similary_word()