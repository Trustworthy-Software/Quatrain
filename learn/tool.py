"""
sentence2vec : transfer sentences to vectors
loaddata: load train/test/valid datas
evaluation_metrics: return scores of (recall_p, recall_n, acc, prc, rc, f1, auc_) 
"""
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, average_precision_score

import sys
sys.path.append("../preprocess/")
from data_util import word2index, index2word

# 输入一个句子和相应的词典，得到这个句子的向量化表示
# 向量的尺寸为词典中词汇的个数，i位置上面的数值为第i个单词在sentence中出现的频率
def sentence2vec(sentence, dictionary):
    vector = np.zeros(len(dictionary))
    for l in sentence:
        vector[l] += 1
    return(1.0 * vector / len(sentence))

def loaddata(pos_sentences, neg_sentences, diction):
    """
    input: pos_sentences, neg_sentences, diction
    output: train/test/valid texts, train/test/valid labels
    """
    # 遍历所有句子，将每一个词映射成编码
    dataset = [] #数据集
    labels = [] #标签
    sentences = [] #原始句子，调试用
    # 处理正向评论
    for sentence in pos_sentences:
        new_sentence = []
        for l in sentence:
            if l in diction:
                new_sentence.append(word2index(l, diction))
        dataset.append(sentence2vec(new_sentence, diction))
        labels.append(0) #正标签为0
        sentences.append(sentence)

    # 处理负向评论
    for sentence in neg_sentences:
        new_sentence = []
        for l in sentence:
            if l in diction:
                new_sentence.append(word2index(l, diction))
        dataset.append(sentence2vec(new_sentence, diction))
        labels.append(1) #负标签为1
        sentences.append(sentence)

    #打乱所有的数据顺序，形成数据集
    # indices为所有数据下标的一个全排列
    indices = np.random.permutation(len(dataset))

    #重新根据打乱的下标生成数据集dataset，标签集labels，以及对应的原始句子sentences
    dataset = [dataset[i] for i in indices]
    labels = [labels[i] for i in indices]
    sentences = [sentences[i] for i in indices]

    #对整个数据集进行划分，分为：训练集、校准集和测试集，其中校准和测试集合的长度都是整个数据集的10分之一
    test_size = len(dataset) // 10
    train_data = dataset[2 * test_size :]
    train_label = labels[2 * test_size :]

    valid_data = dataset[: test_size]
    valid_label = labels[: test_size]

    test_data = dataset[test_size : 2 * test_size]
    test_label = labels[test_size : 2 * test_size]

    return (train_data, train_label, test_data, test_label, valid_data, valid_label)

def evaluation_metrics(y_trues, y_pred_probs):
    fpr, tpr, thresholds = roc_curve(y_true=y_trues, y_score=y_pred_probs, pos_label=1)
    auc_ = auc(fpr, tpr)

    y_preds = [1 if p >= 0.5 else 0 for p in y_pred_probs]

    acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
    prc = precision_score(y_true=y_trues, y_pred=y_preds)
    rc = recall_score(y_true=y_trues, y_pred=y_preds)
    f1 = 2 * prc * rc / (prc + rc)

    """
    print('\n***------------***')
    print('Evaluating AUC, F1, +Recall, -Recall')
    print('Test data size: {}, Incorrect: {}, Correct: {}'.format(len(y_trues), y_trues.count(0), y_trues.count(1)))
    print('Accuracy: %f -- Precision: %f -- +Recall: %f -- F1: %f ' % (acc, prc, rc, f1))
    """
    tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
    recall_p = tp / (tp + fn)
    recall_n = tn / (tn + fp)
    print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(auc_, recall_p, recall_n))
    # return , auc_

    # print('AP: {}'.format(average_precision_score(y_trues, y_pred_probs)))
    # return recall_p, recall_n, acc, prc, rc, f1, auc_
    return acc, prc, rc, f1, auc_, recall_p, recall_n

def rightness(predictions, labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵，labels是数据之中的正确答案"""
    pred = torch.max(predictions.data, 1)[1] # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    rights = pred.eq(labels.data.view_as(pred)).sum() #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return rights, len(labels) #返回正确的数量和这一次一共比较了多少元素