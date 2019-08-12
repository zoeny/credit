# -*- coding: utf-8 -*-
'''
@project: PY_project
@Time : 2019/8/4 15:08
@month : 八月
@Author : mhm
@FileName: word_similer_values.py
@Software: PyCharm
'''
import pickle
from gensim.models import KeyedVectors
import numpy as np
from aip import AipNlp
import jieba,os,sys

class SIM(object):

    def __init__(self):
        self.cn_model=self.load_dict()

    def load_dict(self):
        # 导入预训练语料模型,其实就是维基百科训练好的关系矩阵,这些矩阵是通过word2vec(一般是cbow或skigram模型)来训练,计算量非常大
        yl_pickle_path = 'cn_model.pickle'

        try:
            fr = open(yl_pickle_path, 'rb')
            cn_model = pickle.load(fr)

            fr.close()
        except FileNotFoundError:
            fw = open(yl_pickle_path, 'wb')
            cn_model = KeyedVectors.load_word2vec_format('sgns.wiki.bigram', binary=False)
            pickle.dump(cn_model, fw)
            fw.close()
        return cn_model

    def wiki_simi(self,word1,word2):

        print(word1,word2)
        try:

            w_1 = self.cn_model[word1]
            w_2 = self.cn_model[word2]
            # print(w_1,w_2)
            w_simi = np.dot(w_1/np.linalg.norm(w_1),w_2/np.linalg.norm(w_2))
            # print('wiki词典结果:',w_simi)
            return w_simi

        except KeyError:

            max_len=max([len(list(jieba.cut(word1))),len(list(jieba.cut(word2)))])

            matrix_1 = np.zeros([max_len,300])
            matrix_2 = np.zeros([max_len,300])

            for i,w1 in enumerate(jieba.cut(word1)):
                try:
                    w_1 = self.cn_model[w1]
                except:
                    w_1 = np.zeros(300)
                matrix_1[i:]=w_1

            # print(matrix_1.shape)

            for i,w2 in enumerate(jieba.cut(word2)):
                try:
                    w_2 = self.cn_model[w2]
                except:
                    w_2 = np.zeros(300)
                matrix_2[i:]=w_2
            # print(matrix_2.shape)
            # print(matrix_1, matrix_2)
            from sklearn.metrics.pairwise import cosine_similarity

            # w_simi = np.round(cosine_similarity(matrix_1, matrix_2))
            w_simi = np.round(float(np.sum(matrix_1 * matrix_2)) / (np.linalg.norm(matrix_1) * np.linalg.norm(matrix_2)),2)

            # print('wiki词典结果:',w_simi)
            return w_simi

    def Baidu_simi(self,word1,word2):
        ########## 百度开发者账号 #############
        APP_ID = '10244165'
        API_KEY = 'bghjqNNbey2CkCGG6etSLtDD'
        SECRET_KEY = 'GIQpKsiuOqbCrYXH1P0DY3XR2CY5gBKR '

        client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

        """ 调用短文本相似度 """
        client.simnet(word1,word2)

        """ 如果有可选参数 """
        options = {}
        options["model"] = "CNN"

        """ 带参数调用短文本相似度 """
        w_simi=client.simnet(word1,word2, options)['score']
        # print('Baidu接口结果:',w_simi)
        return w_simi

    def cosine_simi(self,word1,word2):

        list_word1 = [w for w in word1]
        list_word2 = [w for w in word2]

        key_word = list(set(list_word1 + list_word2))  # 取并集

        word_vector1 = np.zeros(len(key_word))  # 给定形状和类型的用0填充的矩阵存储向量
        word_vector2 = np.zeros(len(key_word))
        # 利用词频构造词向量
        for i in range(len(key_word)):  # 依次确定向量的每个位置的值
            for j in range(len(list_word1)):  # 遍历key_word中每个词在句子中的出现次数
                if key_word[i] == list_word1[j]:
                    word_vector1[i] += 1

            for k in range(len(list_word2)):
                if key_word[i] == list_word2[k]:
                    word_vector2[i] += 1
        v1=word_vector1
        v2=word_vector2
        # print(word_vector1)  # 输出向量
        # print(word_vector2)
        w_simi=np.round(float(np.sum(v1 * v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)),2)
        # print('纯余弦公式结果:',w_simi)
        return w_simi

# 测试
if __name__=='__main__':

    word1 = '拍拍贷'
    word2 = '爱奇艺'
    word3 = '信用钱包'
    word4 = '玖富叮当'
    word5 = '相机'
    word6 = '赶集网'

    SIM().wiki_simi(word1,word2)
    SIM().wiki_simi(word1,word3)
    SIM().wiki_simi(word1,word4)
    SIM().wiki_simi(word1,word5)
    SIM().wiki_simi(word1,word6)
    SIM().Baidu_simi(word1, word2)
    SIM().Baidu_simi(word1, word3)
    SIM().Baidu_simi(word1, word4)
    SIM().Baidu_simi(word1, word5)
    SIM().Baidu_simi(word1, word6)
    SIM().cosine_simi(word1,word2)
    SIM().cosine_simi(word1,word3)
    SIM().cosine_simi(word1,word4)
    SIM().cosine_simi(word1,word5)
    SIM().cosine_simi(word1,word6)