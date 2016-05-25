#coding:utf-8
import os
import time
import csv
import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
from gensim.models.word2vec import Word2Vec
start_time = time.clock()
global times
times = 0

class each_sentence(object):
    def __init__(self,dir):
        self.dir = dir
    
    def __iter__(self):
        for fname in os.listdir(self.dir):
            for line in open(os.path.join(self.dir,fname),'r'):
                global times
                times += 1
		line =  line.split(',')
                print 'word2vec model 正在训练  第%d次'%times
                yield list(jieba.cut(line[1])) 

#定义参数
word2vec_size = 256 #词向量维度
print 'word2vec model 开始训练：'
word2vec = Word2Vec(sentences=each_sentence('./data'),size=word2vec_size,min_count=1,workers=4)
word2vec.save('./word2vec_model/word2vec_model')

end_time = time.clock()
run_time = end_time - start_time
print 'word2vec model 训练完成！'
print '运行时间：%f s'%run_time
    













