#coding:utf-8
import os
import sys
import numpy as np
import jieba
from gensim.models.word2vec import Word2Vec
from keras.models import model_from_json
reload(sys)
sys.setdefaultencoding('utf-8')

#找到相近的词语,可以尝试用doc2vec找到相近的句子
def find_similar(words):
    words = unicode(words)
    tol_word = list(jieba.cut(words))
    word2vec = Word2Vec.load('./word2vec_model/word2vec_model')
    word2vec.train(tol_word)
    #print words+'的近义词以及相似度：---------------------------'
    for word,strong in word2vec.most_similar(tol_word):
    	    yield word,strong

#预测词语获得句子的情感
def predic_sens(sens):
    sens = unicode(sens)
    vec = []
    word2vec = Word2Vec.load('./word2vec_model/word2vec_model')
    tol_word = list(jieba.cut(sens))
    with open('./sentiment_model/architecture.json','r') as architecture:
    	sentiment = model_from_json(architecture.read())
    sentiment.load_weights('./sentiment_model/model_weights.h5')
    sentiment.compile(optimizer='adagrad',loss='mse')
    word2vec.train(tol_word)
    for word in tol_word:
    	vec.append(word2vec[word])#词向量
    sens_vec =  (np.array(vec)).sum(axis=0)#句向量
    return str(sentiment.predict_classes(np.array([sens_vec])))

def print_pre(sens):
    print sens+predic_sens(sens)

def predict_file(fpath,result1,result2):
    if not os.path.exists(fpath):
        return False
    fengci = open(result1,'w')
    sens = open(result2,'w')
    with open(fpath,'r') as prefile:
        for line in prefile:
            fengci.write(' '.join(jieba.cut(line))+'\n')
            sens.write(predic_sens(line)+'\n')
    fengci.close()
    sens.close()
    return True

'''
if predict_file('./doc.txt','./fcdoc.txt','./sens.txt'):
    print '分析完成'
else:
    print '文件不存在'
'''
