
import json
import sys
import time
from constants import Output_DATA_DIR, Origin_DATA_DIR

#PyTorch-packages
import torch
import torch.nn as nn
import torch.optim
#from torch.autograd import Variable

# packages of NLP
import re 
from collections import Counter #搜集器，可以让统计词频更简单
import pandas as pd
import jieba

# packages of drawing
import matplotlib.pyplot as plt
import numpy as np

"""
Filter for noise
"""
# remove numbers and '-'
def numbr(string):
    newstring = ''.join([i for i in string if not i.isdigit()])
    newstring = newstring.replace('-','')
    return newstring
# remove punctuation
def filter_punc(sentence):
    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）)：]+", " ", sentence)  
    return(sentence)

class ShowProcess():
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 0 # 当前的处理进度
    max_steps = 0 # 总共需要处理的次数
    max_arrow = 50 #进度条的长度
    infoDone = 'done'

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps, infoDone = 'Done'):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #计算显示多少个'>'
        num_line = self.max_arrow - num_arrow #计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps #计算完成进度，格式为xx.xx%
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r' #带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar) #这两句打印字符到终端
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        print(self.infoDone)
        self.i = 0

#扫描所有的文本，分词、建立词典，分出正向还是负向的评论，is_filter可以过滤是否筛选掉标点符号
def Prepare_data(correct_file, incorrect_file, is_filter = True):
    """
    vacab building
    """
    all_words = [] #存储所有的单词
    pos_sentences = [] #存储correct的 pair
    neg_sentences = [] #存储incorrect的pair
    with open(correct_file, 'r') as fr:
        for idx, line in enumerate(fr):
            if is_filter:
                #过滤标点符号
                line = filter_punc(line)
            #分词
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                pos_sentences.append(words)
    print('{0} 包含 {1} 行, {2} 个词.'.format(correct_file, idx+1, len(all_words)))

    count = len(all_words)
    with open(incorrect_file, 'r') as fr:
        for idx, line in enumerate(fr):
            if is_filter:
                line = filter_punc(line)
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                neg_sentences.append(words)
    print('{0} 包含 {1} 行, {2} 个词.'.format(incorrect_file, idx+1, len(all_words)-count))

    #建立词典，diction的每一项为{w:[id, 单词出现次数]}
    diction = {}
    cnt = Counter(all_words)
    for word, freq in cnt.items():
        diction[word] = [len(diction), freq]
    print('字典大小：{}'.format(len(diction)))
    return(pos_sentences, neg_sentences, diction)

#根据单词返还单词的编码
def word2index(word, diction):
    if word in diction:
        value = diction[word][0]
    else:
        value = -1
    return(value)

#根据编码获得单词
def index2word(index, diction):
    for w,v in diction.items():
        if v[0] == index:
            return(w)
    return(None)


def writetxt2csv(inputfile, outfilename):
    """
    input: outfilepath+name
    output: csvfile
    """
    bugreportcommitfile = open(inputfile)
    lines = bugreportcommitfile.readlines()
    print('Total Data: {} lines'.format(len(lines)))

    column_names = ['bug_id', 'bug report text', 'bug report description', 'generated patch id', 'patch text', 'label']
    # create a dataframe to store cleaned data
    df = pd.DataFrame(columns = column_names)

    max_steps = len(lines)
    process_bar = ShowProcess(max_steps, 'preprocess finished, Okay!')

    for line in lines:
        line2list = line.split('$$')
        line2list[-1] = line2list[-1].replace('\n','')
        #print(line2list)
        line2list[0] = numbr(line2list[0])
        
        #for bug report text
        line2list[1] = filter_punc(line2list[1])
        
        #for bug report description
        line2list[2] = filter_punc(line2list[2])
        
        #for generated patch id
        line2list[3] = numbr(line2list[2])    
        
        #for patch text
        line2list[4] = filter_punc(line2list[3])
        
        # add data into df file
        df_toadd = pd.DataFrame([line2list], columns = column_names)
        df = df.append(df_toadd)

        process_bar.show_process()
        time.sleep(0.01)

    df.to_csv(outfilename)

    list_type = df['label'].unique()
    df_0 = df[df['label'].isin([list_type[0]])]
    df_1 = df[df['label'].isin([list_type[1]])]
    # store correct.csv & incorrect.csv into path './data/'
    correctfile = '%s/correct.csv' % Origin_DATA_DIR
    incorrectfile = '%s/incorrect.csv' % Origin_DATA_DIR
    df_0.to_csv(correctfile)
    df_1.to_csv(incorrectfile)

def splitdata(correct_file, incorrect_file):
    # split data into correct & incorrect text // build dictionary
    pos_sentences, neg_sentences, diction = Prepare_data(correct_file, incorrect_file, True)
    st = sorted([(v[1], w) for w, v in diction.items()])
    return (pos_sentences, neg_sentences, diction, st)