import torch
import torch.nn as nn
import torch.optim
import numpy as np

seed = 3
torch.manual_seed(seed)
torch.backends.cudnn.deterministric = True
torch.backends.cudnn.benchmark = False

import sys
sys.path.append('../preprocess/')

from tool import sentence2vec, loaddata, rightness, evaluation_metrics
from constants import Output_DATA_DIR, Origin_DATA_DIR
from data_util import ShowProcess, numbr, filter_punc, Prepare_data, writetxt2csv, splitdata

correct_file = '%s/correct.csv' % Output_DATA_DIR
incorrect_file  = '%s/incorrect.csv' % Output_DATA_DIR

print(incorrect_file, correct_file)
pos_sentences, neg_sentences, diction, st = splitdata(correct_file, incorrect_file)
train_data, train_label, test_data, test_label, valid_data, valid_label = loaddata(pos_sentences, neg_sentences, diction)

# Model 1: full linked layers
model = nn.Sequential(
    nn.Linear(len(diction), 10),
    nn.ReLU(),
    nn.Linear(10, 2),
    nn.LogSoftmax(),
)

# 损失函数为交叉熵
cost = torch.nn.NLLLoss()
# 优化算法为Adam，可以自动调节学习率
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
records = []


#循环10个Epoch
losses = []
for epoch in range(10):
    for i, data in enumerate(zip(train_data, train_label)):
        x, y = data
        
        # 需要将输入的数据进行适当的变形，主要是要多出一个batch_size的维度，也即第一个为1的维度
        x = torch.tensor(x, requires_grad = True, dtype = torch.float).view(1,-1)
        # x的尺寸：batch_size=1, len_dictionary
        # 标签也要加一层外衣以变成1*1的张量
        y = torch.tensor(np.array([y]), dtype = torch.long)
        # y的尺寸：batch_size=1, 1
        
        # 清空梯度
        optimizer.zero_grad()
        # 模型预测
        predict = model(x)
        # 计算损失函数
        loss = cost(predict, y)
        # 将损失函数数值加入到列表中
        losses.append(loss.data.numpy())
        # 开始进行梯度反传
        loss.backward()
        # 开始对参数进行一步优化
        optimizer.step()
        
        # 每隔3000步，跑一下校验数据集的数据，输出临时结果
        if i % 3000 == 0:
            val_losses = []
            rights = []
            preds = []
            y_trues = []
            # 在所有校验数据集上实验
            for j, val in enumerate(zip(valid_data, valid_label)):
                x, y = val
                x = torch.tensor(x, requires_grad = True, dtype = torch.float).view(1,-1)
                y = torch.tensor(np.array([y]), dtype = torch.long)
                predict = model(x)
                # 调用rightness函数计算准确度
                right = rightness(predict, y)

                pred = torch.max(predict.data, 1)[1]
                preds.append(pred)
                y_trues.append(y)
                

                #rights.append(right)
                loss = cost(predict, y)
                val_losses.append(loss.data.numpy())

            #print(len(preds), len(y_trues), preds.count(0), preds.count(1), y_trues.count(0), y_trues.count(1))
            acc, prc, rc, f1, auc_, recall_p, recall_n = evaluation_metrics(preds, y_trues)

            print('\n***------------***')
            print('Evaluating AUC, F1, +Recall, -Recall')
            print('Valid data size: {}, Incorrect: {}, Correct: {}'.format(len(y_trues), y_trues.count(0), y_trues.count(1)))
            print('The {}-th epoch ，train loss：{:.4f}, valid loss：{:.4f}, valid accuracy: {:.4f}, precision: {:.4f}, +Recall: {:.3f}, -Recall: {:.3f}, f1: {:.4f}, auc: {:.4f} '.format(epoch, np.mean(losses),
                                                                        np.mean(val_losses), acc, prc, recall_p, recall_n, f1, auc_))

            #recall_p, recall_n, acc, prc, rc, f1, auc_ = evaluation_metrics(y_trues, preds)
            #print(recall_p, recall_n, acc, prc, rc, f1, auc_)  
            # 将校验集合上面的平均准确度计算出来
            # right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
            # 
            #records.append([np.mean(losses), np.mean(val_losses), right_ratio])

#在测试集上分批运行，并计算总的正确率
vals = [] #记录准确率所用列表

#对测试数据集进行循环
test_preds = []
test_y_trues = []
for data, target in zip(test_data, test_label):
    data, target = torch.tensor(data, dtype = torch.float).view(1,-1), torch.tensor(np.array([target]), dtype = torch.long)
    output = model(data) #将特征数据喂入网络，得到分类的输出
    test_pred = torch.max(output.data, 1)[1]
    test_preds.append(test_pred)
    test_y_trues.append(target)


#计算准确率
test_acc, test_prc, test_rc, test_f1, test_auc_, test_ecall_p, test_recall_n = evaluation_metrics(preds, y_trues)
print('test accuracy: {:.4f}, precision: {:.4f}, f1: {:.4f}, auc: {:.4f} '.format(test_acc, test_prc, test_f1, test_auc_))

torch.save(model, '../saved_models/fulllinedmodel.mdl')