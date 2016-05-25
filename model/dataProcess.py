#coding:utf-8
import random
import os
import sys
from  itertools import islice
import re
import csv

#获得文件路径
def list_dir(dir):
    filepath = []
    list = os.listdir(dir)
    for i in list:
        path = os.path.join(dir,i)
        filepath.append(path)
    return filepath

#读取一个文件的star跟句子
def read_file(dir):
    seg1 = re.compile('\((.+?)\)',re.S)
    with open(dir,'r') as one_file:
        for line in islice(one_file,1,None):
            result = []
            match = re.match(seg1,line)
            if match:
                star = match.group(1)
                star = star.replace('星','',1)
                sentence = line.replace(re.match(seg1,line).group(0),' ',1).strip()
                result.append(star)
                result.append(sentence)
                yield result
        
#写入csv文件
def write_file(dir,file_list):
    with open(dir,'w') as csv_file:
        writer = csv.writer(csv_file)
        for one_file in file_list:
            for line in read_file(one_file):
                writer.writerow(line)

#随机挑选十分之一的文件作为测试文件
dir = './douban'
file_list = list_dir(dir)
file_test = random.sample(file_list,len(file_list)/10)
file_train = []
for train in file_list:
    if train not in file_test:
        file_train.append(train)
write_file('./data/test.csv',file_test)
write_file('./data/train.csv',file_train)

read_test = csv.reader(file('./data/test.csv'))
test_tol = 0
test_pos = 0
test_neg = 0
for line in read_test:
    test_tol += 1
    if line[0] == '4' or line[0] == '5':
        test_pos += 1
    elif line[0] == '1' or line[0] == '2' or line[0] == '3':
        test_neg += 1
    else: pass
read_train = csv.reader(file('./data/train.csv'))
train_tol = 0
train_pos = 0
train_neg = 0
for line in read_train:
    train_tol += 1
    if line[0] == '4' or line[0] == '5':
        train_pos += 1
    elif line[0] == '1' or line[0] == '2' or line[0] == '3':
        train_neg += 1
data = open('./data/read.txt','w')
data.write('test tol:'+str(test_tol)+'\n')
data.write('test pos:'+str(test_pos)+'\n')
data.write('test neg:'+str(test_neg)+'\n')
data.write('train tol:'+str(train_tol)+'\n')
data.write('train pos:'+str(train_pos)+'\n')
data.write('train neg:'+str(train_neg)+'\n')
