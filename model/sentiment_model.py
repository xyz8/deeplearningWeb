#coding:utf-8
import time
import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
from gensim.models.word2vec import Word2Vec
from keras.models import Sequential, model_from_json
from keras.utils import np_utils
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
start_train = time.clock()

class vec(object):
    def __init__(self):
        self.word2vec = Word2Vec.load('./word2vec_model/word2vec_model')

    def make(self,sentence):
        vec = []
        for word in sentence:
            vec.append(self.word2vec[word])#词向量
        return (np.array(vec)).sum(axis=0)#句向量

class mark_sentence(object):
    def __init__(self,dir):
        self.dir = dir
        self.vec = vec()

    def lstm_iter(self):
        while 1:
            for line in open(self.dir,'r'):
                line = line.split(',')
                if (line[0] == '4') or (line[0] == '5'):
                    yield (np.array([ self.vec.make(jieba.cut(line[1]))]),np.array([1]))
                if (line[0] == '1') or (line[0] == '2') or (line[0] == '3'):
                    yield (np.array([ self.vec.make(jieba.cut(line[1]))]),np.array([0]))

    def normal_iter(self):
            for line in open(self.dir,'r'):
                line = line.split(',')
                if (line[0] == '4') or (line[0] == '5'):
                    yield (np.array([ self.vec.make(jieba.cut(line[1]))]),np.array([1]))
                if (line[0] == '1') or (line[0] == '2') or (line[0] == '3'):
                    yield (np.array([ self.vec.make(jieba.cut(line[1]))]),np.array([0]))


def train(train_path,wordvec_size,hidden_node1,hidden_node2,hidden_node3,batch,nb,loss_way,loss_object):
    #建立深度神经网络
    print '初始化网络:\n输入层%d节点，隐含层%d节点，隐含层%d节点，隐含层%d节点，输出层%d节点'%(wordvec_size,hidden_node1,hidden_node2,hidden_node3,1)
    print '样本数%d，迭代数%d'%(batch,nb)
    print 'way:'+loss_way+'  object:'+loss_object
    
    model = Sequential()
    model.add(Dense(hidden_node1, input_dim=wordvec_size, init='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_node2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(hidden_node3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss_object,optimizer=loss_way,metrics=['accuracy'])
    
    #训练模型
    print '开始训练网络'
    hist = model.fit_generator(mark_sentence(train_path).lstm_iter(),samples_per_epoch=batch,nb_epoch=nb)
    train_time = time.clock()
    #保存模型
    json_string = model.to_json()  
    with open('./sentiment_model/architecture.json','w') as architecture:
        architecture.write(json_string)  
    model.save_weights('./sentiment_model/model_weights.h5') 
    with open('./sentiment_model/README.txt','w') as readme:
        readme.write('神经网络模型各个参数:\n\
        输入层:'+str(wordvec_size)+'节点\n\
        隐含层:'+str(hidden_node1)+'节点\n\
        隐含层:'+str(hidden_node2)+'节点\n\
        隐含层:'+str(hidden_node3)+'节点\n\
        输出层:1节点\n\
        每次更新梯度的样本数:'+str(batch)+'\n\
        训练的迭代次数:'+str(nb)+'\n\
        代价函数:'+loss_object+'\n\
        优化模型:'+loss_way+'\n\
        训练时间:'+str(train_time-start_train)+'s\n')
    print '训练完成！'


def test(test_path):
    print '加载模型'
    start_test = time.clock()
    with open('./sentiment_model/architecture.json','r') as architecture:
        model = model_from_json(architecture.read())
    model.load_weights('./sentiment_model/model_weights.h5')
    model.compile(optimizer='rmsprop',loss='binary_crossentropy')
    #测试模型
    print '开始测试网络'
    test_class = []
    test_label = []
    for test_vec , mark in mark_sentence(test_path).normal_iter():
    	test_class.append(model.predict_classes(test_vec))
        test_label.append(mark)
    test_time = time.clock()
    score1 = np_utils.accuracy(test_class,test_label)
    print '模型准确率为:%f'%score1
    #保存参数以及模型
    with open('./sentiment_model/README.txt','a') as readme:
        readme.write('测试时间:'+str(test_time-start_test)+'s\n\
        模型准确度:'+str(score1)+'\n')
    with open('./data/pre_labels.txt','w') as labels:#保存标签
        for label in test_class:
            labels.write(str(label)+'\n')
    print '测试完成！'
    

     #定义参数
wordvec_size = 256
hidden_node1,hidden_node2,hidden_node3 = 256,128,64#两个隐含层的节点数
batch = 5000 #每次更新梯度的样本数
nb = 1000 #训练数据的迭代次数
loss_way = 'rmsprop'#优化梯度的模型
loss_object = 'binary_crossentropy'#用什么公式作为代价函数
#训练
#train_path = './data/train.csv'
#train(train_path,wordvec_size,hidden_node1,hidden_node2,hidden_node3,batch,nb,loss_way,loss_object)
#测试
test_path = './data/test.csv'
test(test_path)
