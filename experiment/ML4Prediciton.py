import imp
import json
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
import xgboost as xgb
from tensorflow import keras
from keras.layers import LSTM, Dense, Dropout, RNN
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense, Embedding, Dropout, Input, Concatenate
from experiment.deep_learning import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,3"


class Classifier:
    def __init__(self, dataset, labels, algorithm, kfold, train_features=None, train_labels=None, test_features=None, test_labels=None, test_info_for_patch=None):
        self.dataset = dataset
        self.labels = labels
        self.algorithm = algorithm
        self.kfold = kfold
        self.threshold = 0.4

        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.test_info_for_patch = test_info_for_patch

    def evaluation_metrics(self, y_true, y_pred_prob):
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred_prob, pos_label=1)
        auc_ = auc(fpr, tpr)

        y_pred = [1 if p >= self.threshold else 0 for p in y_pred_prob]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        prc = precision_score(y_true=y_true, y_pred=y_pred)
        rc = recall_score(y_true=y_true, y_pred=y_pred)
        f1 = 2 * prc * rc / (prc + rc)

        print('Accuracy: %f -- Precision: %f -- +Recall: %f -- F1: %f ' % (acc, prc, rc, f1))
        if y_true == y_pred:
            tn, fp, fn, tp = 1, 0, 0, 1
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall_p = tp / (tp + fn)
        recall_n = tn / (tn + fp)
        print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(auc_, recall_p, recall_n))


        # return , auc_
        return auc_, recall_p, recall_n, acc, prc, rc, f1

    def evaluate_message(self, y_true, y_pred_prob, test_info_for_patch=None):
        y_pred = [1 if p >= self.threshold else 0 for p in y_pred_prob]
        correct_message_length, incorrect_message_length = [], []
        correct_report_length, incorrect_report_length = [], []
        if test_info_for_patch:
            with open('./data/CommitMessage/Generated_commit_message_All.json', 'r+') as f:
                commit_message_dict = json.load(f)
            with open('./data/CommitMessage/Developer_commit_message.json', 'r+') as f:
                developer_commit_message_dict = json.load(f)
            with open('./data/BugReport/Bug_Report_All.json', 'r+') as f:
                bug_report_dict = json.load(f)
            for i in range(len(y_pred)):
                patch_id = test_info_for_patch[i][1]
                project_id = patch_id.split('_')[0].split('-')[1] + '-' + patch_id.split('_')[0].split('-')[2]
                if project_id == 'closure-63':
                    developer_commit_message = developer_commit_message_dict['closure-62']
                elif project_id == 'closure-93':
                    developer_commit_message = developer_commit_message_dict['closure-92']
                else:
                    developer_commit_message = developer_commit_message_dict[project_id]

                if y_pred[i] == y_true[i]:
                    # print('Patch id: {}'.format(patch_id))
                    # print('Commit message: {}'.format(commit_message_dict[patch_id]))
                    generated_commit_message = commit_message_dict[patch_id]
                    word_number = len(generated_commit_message.split(' '))
                    correct_message_length.append(word_number)

                    bug_report = bug_report_dict[project_id]
                    word_number2 = len(bug_report[0].split(' '))
                    correct_report_length.append(word_number2)

                    # developer_commit_message = developer_commit_message_dict[project_id]

                    # similarity of generated commit vs. developer commit

                    # print('Correct prediction: ')
                    # print('Bug report: {}'.format(bug_report_dict[project_id]))
                    # print('G Commit message: {}'.format(generated_commit_message))
                    # print('D Commit message: {}'.format(developer_commit_message))
                    # print('---------------------')


                else:
                    # print('Patch id: {}'.format(patch_id))
                    # print('Commit message: {}'.format(commit_message_dict[patch_id]))
                    generated_commit_message = commit_message_dict[patch_id]
                    word_number = len(generated_commit_message.split(' '))
                    incorrect_message_length.append(word_number)

                    bug_report = bug_report_dict[project_id]
                    word_number2 = len(bug_report[0].split(' '))
                    incorrect_report_length.append(word_number2)


                    # print('Incorrect prediction: ')
                    # print('Bug report: {}'.format(bug_report_dict[project_id]))
                    # print('G Commit message: {}'.format(generated_commit_message))
                    # print('D Commit message: {}'.format(developer_commit_message))
                    # print('---------------------')

        messageL_correctP = np.array(correct_message_length).mean()
        messageL_incorrectP = np.array(incorrect_message_length).mean()
        print('message length in correct prediction : {}'.format(messageL_correctP))
        print('message length in incorrect prediction : {}'.format(messageL_incorrectP))

        reportL_correctP = np.array(correct_report_length).mean()
        reportL_incorrectP = np.array(incorrect_report_length).mean()
        print('report length in correct prediction : {}'.format(reportL_correctP))
        print('report length in incorrect prediction : {}'.format(reportL_incorrectP))

        return messageL_correctP, messageL_incorrectP, reportL_correctP, reportL_incorrectP

    def confusion_matrix(self, y_pred, y_test):
        for i in range(1, 100):
            y_pred_tn = [1 if p >= i / 100.0 else 0 for p in y_pred]
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tn).ravel()
            print('i:{}'.format(i / 100), end=' ')
            # print('TP: %d -- TN: %d -- FP: %d -- FN: %d' % (tp, tn, fp, fn))
            recall_p = tp / (tp + fn)
            recall_n = tn / (tn + fp)
            print('+Recall: {:.3f}, -Recall: {:.3f}'.format(recall_p, recall_n))

    def cross_validation(self,):
        print('All data size: {}, Incorrect: {}, Correct: {}'.format(len(self.labels), list(self.labels).count(0), list(self.labels).count(1)))
        print('Algorithm: {}'.format(self.algorithm))
        print('#####')

        accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
        rcs_p, rcs_n = list(), list()
        skf = StratifiedKFold(n_splits=self.kfold, shuffle=True)

        for train_index, test_index in skf.split(self.dataset, self.labels):
            x_train, y_train = self.dataset[train_index], self.labels[train_index]
            x_test, y_test = self.dataset[test_index], self.labels[test_index]

            # standard data
            scaler = StandardScaler().fit(x_train)
            # scaler = MinMaxScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)

            clf = None
            if self.algorithm == 'lr':
                clf = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train, y=y_train)
            elif self.algorithm == 'dt':
                clf = DecisionTreeClassifier().fit(X=x_train, y=y_train, sample_weight=None)
            elif self.algorithm == 'rf':
                clf = RandomForestClassifier().fit(X=x_train, y=y_train,)
            elif self.algorithm == 'xgb':
                dtrain = xgb.DMatrix(x_train, label=y_train)
                clf = xgb.train(params={'objective': 'binary:logistic', 'verbosity': 0}, dtrain=dtrain, )
            elif self.algorithm == 'nb':
                clf = GaussianNB().fit(X=x_train, y=y_train)

            if self.algorithm == 'xgb':
                x_test_xgb = x_test
                x_test_xgb_dmatrix = xgb.DMatrix(x_test_xgb, label=y_test)
                y_pred = clf.predict(x_test_xgb_dmatrix)
            else:
                y_pred = clf.predict_proba(x_test)[:, 1]

            auc_, recall_p, recall_n, acc, prc, rc, f1 = self.evaluation_metrics(y_true=list(y_test), y_pred_prob=list(y_pred))

            accs.append(acc)
            prcs.append(prc)
            rcs.append(rc)
            f1s.append(f1)

            aucs.append(auc_)
            rcs_p.append(recall_p)
            rcs_n.append(recall_n)

        print('')
        print('{}-fold cross validation mean: '.format(self.kfold))
        print('Accuracy: {:.1f} -- Precision: {:.1f} -- +Recall: {:.1f} -- F1: {:.1f} -- AUC: {:.3f}'.format(np.array(accs).mean()*100, np.array(prcs).mean()*100, np.array(rcs).mean()*100, np.array(f1s).mean()*100, np.array(aucs).mean()))
        print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(np.array(aucs).mean(), np.array(rcs_p).mean(), np.array(rcs_n).mean()))
        print('---------------')

    def leave1out_validation(self):
        x_train, y_train = self.train_features, self.train_labels
        x_test, y_test = self.test_features, self.test_labels

        # standard data
        scaler = StandardScaler().fit(x_train)
        # scaler = MinMaxScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        clf = None
        if self.algorithm == 'lr':
            clf = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X=x_train, y=y_train)
        elif self.algorithm == 'dt':
            clf = DecisionTreeClassifier().fit(X=x_train, y=y_train, sample_weight=None)
        elif self.algorithm == 'rf':
            clf = RandomForestClassifier().fit(X=x_train, y=y_train, )
        elif self.algorithm == 'xgb':
            dtrain = xgb.DMatrix(x_train, label=y_train)
            clf = xgb.train(params={'objective': 'binary:logistic', 'verbosity': 0}, dtrain=dtrain, num_boost_round=500)
        elif self.algorithm == 'nb':
            clf = GaussianNB().fit(X=x_train, y=y_train)
        elif self.algorithm == 'lstm':
            # reshape input to be [samples, time steps, features]
            x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
            x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
            y_train = np.array(y_train)

            clf = Sequential()
            clf.add(LSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))
            clf.add(Dropout(0.4))
            clf.add(LSTM(128))
            clf.add(Dense(64, activation='relu'))
            clf.add(Dropout(0.2))
            clf.add(Dense(2, activation='softmax'))
            clf.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            clf.fit(x_train, y_train, epochs=50, batch_size=100, verbose=2)
        elif self.algorithm == 'widedeep':
            x_train_q = x_train[:, :1024]
            x_train_a = x_train[:, 1024:]

            y_train = np.array(y_train)

            combine_qa_model = get_wide_deep(x_train_q.shape[1], x_train_a.shape[1])
            callback = [keras.callbacks.EarlyStopping(monitor='val_auc', patience=2, mode="max", verbose=1), ]
            combine_qa_model.fit([x_train_q, x_train_a], y_train, validation_split=0.1, batch_size=64,epochs=10,)
        elif self.algorithm == 'rnn_qa':
            x_train_q = x_train[:, :1024]
            x_train_a = x_train[:, 1024:]
            x_train_q = np.reshape(x_train_q, (x_train_q.shape[0], x_train_q.shape[1], 1))
            x_train_a = np.reshape(x_train_a, (x_train_a.shape[0], x_train_a.shape[1], 1))

            y_train = np.array(y_train)

            combine_qa_model = get_rnn_qa(x_train_q.shape[1:], x_train_a.shape[1:])
            callback = [keras.callbacks.EarlyStopping(monitor='auc', patience=3, mode="max", verbose=1), ]
            combine_qa_model.fit([x_train_q, x_train_a], y_train, callbacks=callback, batch_size=64,epochs=10, )
        elif self.algorithm == 'qa_attetion':
            seq_maxlen = 64
            y_train = np.array(y_train).astype(float)
            x_train_q = x_train[:, :1024]
            x_train_a = x_train[:, 1024:]
            x_train_q = np.reshape(x_train_q, (x_train_q.shape[0], seq_maxlen, -1))
            x_train_a = np.reshape(x_train_a, (x_train_a.shape[0], seq_maxlen, -1))
            combine_qa_model = get_qa_attention(x_train_q.shape[1:], x_train_a.shape[1:])
            callback = [keras.callbacks.EarlyStopping(monitor='val_auc', patience=1, mode="max", verbose=0), ]
            combine_qa_model.fit([x_train_q, x_train_a], y_train, callbacks=callback, validation_split=0.2, batch_size=128,epochs=10,)

        # predict
        if self.algorithm == 'xgb':
            x_test_xgb = x_test
            x_test_xgb_dmatrix = xgb.DMatrix(x_test_xgb, label=y_test)
            y_pred = clf.predict(x_test_xgb_dmatrix)
        elif self.algorithm == 'widedeep':
            x_test_q = x_test[:, :1024]
            x_test_a = x_test[:, 1024:]
            y_pred = combine_qa_model.predict([x_test_q, x_test_a])[:, 0]
        elif self.algorithm == 'qa_attetion':
            seq_maxlen = 64
            x_test_q = x_test[:, :1024]
            x_test_a = x_test[:, 1024:]
            x_test_q = np.reshape(x_test_q, (x_test_q.shape[0], seq_maxlen, -1))
            x_test_a = np.reshape(x_test_a, (x_test_a.shape[0], seq_maxlen, -1))
            y_pred = combine_qa_model.predict([x_test_q, x_test_a])[:, 0]
        else:
            y_pred = clf.predict_proba(x_test)[:, 1]

        auc_, recall_p, recall_n, acc, prc, rc, f1 = self.evaluation_metrics(y_true=list(y_test), y_pred_prob=list(y_pred),)
        messageL_correctP, messageL_incorrectP, reportL_correctP, reportL_incorrectP = self.evaluate_message(y_true=list(y_test), y_pred_prob=list(y_pred), test_info_for_patch=self.test_info_for_patch)

        # self.confusion_matrix(y_pred, y_test)

        print('---------------')
        return auc_, recall_p, recall_n, acc, prc, rc, f1, messageL_correctP, messageL_incorrectP, reportL_correctP, reportL_incorrectP
