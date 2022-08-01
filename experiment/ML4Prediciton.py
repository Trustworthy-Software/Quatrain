import imp
import json
from nltk.tokenize import word_tokenize
from gensim.utils import tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import *

import pickle
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
from keras.models import Sequential, load_model
import tensorflow as tf
from keras.layers import Dense, Embedding, Dropout, Input, Concatenate
from experiment.deep_learning import *
from scipy.spatial import distance as dis
import distance._levenshtein
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,3"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


class Classifier:
    def __init__(self, dataset, labels, algorithm, kfold, train_features=None, train_labels=None, test_features=None, test_labels=None, random_test_features=None, test_info_for_patch=None):
        self.dataset = dataset
        self.labels = labels
        self.algorithm = algorithm
        self.kfold = kfold
        self.threshold = 0.5

        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.random_test_features = random_test_features
        self.test_info_for_patch = test_info_for_patch

        self.matrix = []
        self.correct = 0
        self.predict_correct = []
        self.random_correct = []

        self.messageL = []
        self.reportL = []
        self.similarity_message = []

        with open('./data/CommitMessage/Generated_commit_message_All.json', 'r+') as f:
            self.commit_message_dict = json.load(f)
        with open('./data/CommitMessage/Generated_commit_message_All_bert.pickle', 'rb') as f:
            self.commit_message_vector_dict = pickle.load(f)
        with open('./data/CommitMessage/Developer_commit_message.json', 'r+') as f:
            self.developer_commit_message_dict = json.load(f)
        with open('./data/CommitMessage/Developer_commit_message_bert.pickle', 'rb') as f:
            self.developer_commit_message_vector_dict = pickle.load(f)
        with open('./data/BugReport/Bug_Report_All.json', 'r+') as f:
            self.bug_report_dict = json.load(f)
        # normalize
        scaler = Normalizer()
        scaler = MinMaxScaler()
        assemble = []
        for k,v in self.commit_message_vector_dict.items():
            assemble.append(v.flatten())
        self.scaler_message1 = scaler.fit(np.array(assemble))

        scaler = MinMaxScaler()
        assemble = []
        for k,v in self.developer_commit_message_vector_dict.items():
            assemble.append(v.flatten())
        self.scaler_message2 = scaler.fit(np.array(assemble))

    def evaluation_metrics(self, y_true, y_pred_prob):
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred_prob, pos_label=1)
        auc_ = auc(fpr, tpr)

        y_pred = [1 if p >= self.threshold else 0 for p in y_pred_prob]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        prc = precision_score(y_true=y_true, y_pred=y_pred)
        rc = recall_score(y_true=y_true, y_pred=y_pred)
        f1 = 2 * prc * rc / (prc + rc)

        print('AUC: %f -- F1: %f -- Accuracy: %f -- Precision: %f ' % (auc_, f1, acc, prc,))
        if y_true == y_pred:
            tn, fp, fn, tp = 1, 0, 0, 1
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall_p = tp / (tp + fn)
        recall_n = tn / (tn + fp)
        # print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(auc_, recall_p, recall_n))

        # return , auc_
        return auc_, recall_p, recall_n, acc, prc, rc, f1

    def evaluate_message(self, y_true, y_pred_prob, test_info_for_patch=None):
        y_pred = [1 if p >= self.threshold else 0 for p in y_pred_prob]
        message_length = []
        report_length = []
        if test_info_for_patch:
            for i in range(len(y_pred)):
                patch_id = test_info_for_patch[i][1]
                project_id = patch_id.split('_')[0].split('-')[1] + '-' + patch_id.split('_')[0].split('-')[2]
                if project_id == 'closure-63':
                    developer_commit_message = self.developer_commit_message_dict['closure-62']
                    developer_commit_message_vector = self.developer_commit_message_vector_dict['closure-62']
                elif project_id == 'closure-93':
                    developer_commit_message = self.developer_commit_message_dict['closure-92']
                    developer_commit_message_vector = self.developer_commit_message_vector_dict['closure-92']
                else:
                    developer_commit_message = self.developer_commit_message_dict[project_id]
                    developer_commit_message_vector = self.developer_commit_message_vector_dict[project_id]
                generated_commit_message = self.commit_message_dict[patch_id]
                generated_commit_message_vector = self.commit_message_vector_dict[patch_id]

                tokenizer = RegexpTokenizer(r'\w+')
                # tokenizer = WhitespaceTokenizer()

                if '_Developer_' in patch_id:
                    # word_number_message = len(developer_commit_message.split(' '))
                    word_number_message = len(set(list(tokenizer.tokenize(developer_commit_message))))
                else:
                    # word_number_message = len(generated_commit_message.split(' '))
                    word_number_message = len(set(list(tokenizer.tokenize(generated_commit_message))))

                bug_report = self.bug_report_dict[project_id]
                # word_number_report = len(bug_report[0].split(' '))
                word_number_report = len(set(list(tokenizer.tokenize(bug_report[0]))))

                # generated_commit_message_vector = self.scaler_message1.transform(generated_commit_message_vector)
                # developer_commit_message_vector = self.scaler_message2.transform(developer_commit_message_vector)

                if generated_commit_message == '' or developer_commit_message == '':
                    # print('null message')
                    continue
                if y_pred[i] == y_true[i]:
                    # print('Patch id: {}'.format(patch_id))
                    # print('Commit message: {}'.format(commit_message_dict[patch_id]))
                    message_length.append(['Correct', word_number_message])
                    report_length.append(['Correct', word_number_report])

                    # similarity of generated commit vs. developer commit
                    correct_distance_lev = distance.levenshtein(generated_commit_message, developer_commit_message)
                    correct_distance_eu = dis.euclidean(generated_commit_message_vector, developer_commit_message_vector)
                    # correct_similarity_cos = dis.cosine(generated_commit_message_vector, developer_commit_message_vector)
                    self.similarity_message.append(['Correct', correct_distance_lev])

                    # QualityOfMessage
                    # print('Correct prediction: ')
                    # print('Bug report: {}'.format(bug_report))
                    # print('D. Commit message: {}'.format(developer_commit_message))
                    # print('Patch ID: {}'.format(patch_id))
                    # print('G. Commit message: {}'.format(generated_commit_message))
                    # print('---------------------')

                else:
                    # print('Patch id: {}'.format(patch_id))
                    # print('Commit message: {}'.format(commit_message_dict[patch_id]))
                    message_length.append(['Incorrect', word_number_message])
                    report_length.append(['Incorrect', word_number_report])

                    incorrect_distance_lev = distance.levenshtein(generated_commit_message, developer_commit_message)
                    incorrect_distance_eu = dis.euclidean(generated_commit_message_vector, developer_commit_message_vector)
                    # incorrect_similarity_cos = dis.cosine(generated_commit_message_vector, developer_commit_message_vector)
                    self.similarity_message.append(['Incorrect', incorrect_distance_lev])

                    # print('Incorrect prediction: ')
                    # print('Bug report: {}'.format(bug_report_dict[project_id]))
                    # print('G Commit message: {}'.format(generated_commit_message))
                    # print('D Commit message: {}'.format(developer_commit_message))
                    # print('---------------------')

        self.messageL = message_length
        self.reportL = report_length

    def evaluation_sanity(self, y_true, y_pred_prob, y_pred_random_prob):
        # y_pred = [1 if p >= self.threshold else 0 for p in y_pred_prob]
        # y_pred_random = [1 if p >= self.threshold else 0 for p in y_pred_random]
        cnt_model, random_model = 0.0, 0.0
        cnt_model, random_model = [], []
        for i in range(len(y_true)):
            if y_pred_prob[i] >= 0.5:
                cnt_model.append(y_pred_prob[i])
                # if y_true[i] == y_pred_random[i]:
                random_model.append(y_pred_random_prob[i])
        self.correct += y_true.count(1)
        self.predict_correct = cnt_model
        self.random_correct = random_model
        # print('fail/cnt: {}'.format((cnt_model-random_model)/cnt_model))
    def confusion_quality(self, y_pred, y_test):
        for i in range(1, 10):
            y_pred_tn = [1 if p >= i / 10.0 else 0 for p in y_pred]
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tn).ravel()
            print('i:{}'.format(i / 10), end=' ')
            print('TP: %d -- TN: %d -- FP: %d -- FN: %d --' % (tp, tn, fp, fn), end=' ')
            recall_p = tp / (tp + fn)
            recall_n = tn / (tn + fp)
            print('+Recall: {:.3f}, -Recall: {:.3f}'.format(recall_p, recall_n))

            self.matrix.append([tp, tn, fp, fn, recall_p, recall_n])

    def confusion_matrix(self, y_pred, y_test):
        for i in range(1, 10):
            y_pred_tn = [1 if p >= i / 10.0 else 0 for p in y_pred]
            if y_test == y_pred_tn:
                tn, fp, fn, tp = 1, 0, 0, 1
            else:
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tn).ravel()
            print('i:{}'.format(i / 10), end=' ')
            print('TP: %d -- TN: %d -- FP: %d -- FN: %d --' % (tp, tn, fp, fn), end=' ')
            recall_p = tp / (tp + fn)
            recall_n = tn / (tn + fp)
            print('+Recall: {:.3f}, -Recall: {:.3f}'.format(recall_p, recall_n))

            self.matrix.append([tp, tn, fp, fn, recall_p, recall_n])


        # for i in range(1, 100):
        #     y_pred_tn = [1 if p >= i / 100.0 else 0 for p in y_pred]
        #     tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tn).ravel()
        #     print('i:{}'.format(i / 100), end=' ')
        #     # print('TP: %d -- TN: %d -- FP: %d -- FN: %d' % (tp, tn, fp, fn))
        #     recall_p = tp / (tp + fn)
        #     recall_n = tn / (tn + fp)
        #     print('+Recall: {:.3f}, -Recall: {:.3f}'.format(recall_p, recall_n))

    def cross_validation(self,):
        print('All data size: {}, Incorrect: {}, Correct: {}'.format(len(self.labels), list(self.labels).count(0), list(self.labels).count(1)))
        print('Algorithm: {}'.format(self.algorithm))
        print('#####')

        accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
        rcs_p, rcs_n = list(), list()
        skf = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=0)

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
                clf = RandomForestClassifier(random_state=0).fit(X=x_train, y=y_train,)
            elif self.algorithm == 'xgb':
                dtrain = xgb.DMatrix(x_train, label=y_train)
                clf = xgb.train(params={'objective': 'binary:logistic', 'verbosity': 0}, dtrain=dtrain, )
            elif self.algorithm == 'nb':
                clf = GaussianNB().fit(X=x_train, y=y_train)
            elif self.algorithm == 'qa_attetion':
                seq_maxlen = 64
                y_train = np.array(y_train).astype(float)
                x_train_q = x_train[:, :1024]
                x_train_a = x_train[:, 1024:]
                x_train_q = np.reshape(x_train_q, (x_train_q.shape[0], seq_maxlen, -1))
                x_train_a = np.reshape(x_train_a, (x_train_a.shape[0], seq_maxlen, -1))
                combine_qa_model = get_qa_attention(x_train_q.shape[1:], x_train_a.shape[1:])
                callback = [keras.callbacks.EarlyStopping(monitor='val_auc', patience=1, mode="max", verbose=0), ]
                combine_qa_model.fit([x_train_q, x_train_a], y_train, callbacks=callback, validation_split=0.2, batch_size=128, epochs=10,)


            if self.algorithm == 'xgb':
                x_test_xgb = x_test
                x_test_xgb_dmatrix = xgb.DMatrix(x_test_xgb, label=y_test)
                y_pred = clf.predict(x_test_xgb_dmatrix)
            elif self.algorithm == 'qa_attetion':
                seq_maxlen = 64
                x_test_q = x_test[:, :1024]
                x_test_a = x_test[:, 1024:]
                x_test_q = np.reshape(x_test_q, (x_test_q.shape[0], seq_maxlen, -1))
                x_test_a = np.reshape(x_test_a, (x_test_a.shape[0], seq_maxlen, -1))
                y_pred = combine_qa_model.predict([x_test_q, x_test_a])[:, 0]
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
        print('############################################')
        print('Insights, {}-fold cross validation.'.format(self.kfold))
        print('AUC: {:.3f} -- F1: {:.1f} -- Accuracy: {:.1f} -- Precision: {:.1f} -- +Recall: {:.1f} '.format(np.array(aucs).mean(), np.array(f1s).mean()*100, np.array(accs).mean()*100, np.array(prcs).mean()*100, np.array(rcs).mean()*100, ))
        # print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(np.array(aucs).mean(), np.array(rcs_p).mean(), np.array(rcs_n).mean()))
        print('---------------')

    def leave1out_validation(self, i, Sanity=None, QualityOfMessage=None):
        x_train, y_train = self.train_features, self.train_labels
        x_test, y_test = self.test_features, self.test_labels
        x_test_random = self.random_test_features

        # standard data
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        if Sanity:
            x_test_random = scaler.transform(x_test_random)

        clf = None
        if self.algorithm == 'lr':
            clf = LogisticRegression(random_state=0).fit(X=x_train, y=y_train)
        elif self.algorithm == 'dt':
            clf = DecisionTreeClassifier().fit(X=x_train, y=y_train, sample_weight=None)
        elif self.algorithm == 'rf':
            clf = RandomForestClassifier(random_state=0).fit(X=x_train, y=y_train, )
        elif self.algorithm == 'xgb':
            dtrain = xgb.DMatrix(x_train, label=y_train)
            clf = xgb.train(params={'objective': 'binary:logistic', 'verbosity': 0}, dtrain=dtrain,)
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
        elif self.algorithm == 'qa_attetion':
            model_saved = 'models/quatrain_'+str(i+1)+'.h5'
            if os.path.exists(model_saved):
                combine_qa_model = load_model('models/quatrain_'+str(i+1)+'.h5')
            else:
                seq_maxlen = 64
                y_train = np.array(y_train).astype(float)
                x_train_q = x_train[:, :1024]
                x_train_a = x_train[:, 1024:]
                x_train_q = np.reshape(x_train_q, (x_train_q.shape[0], seq_maxlen, -1))
                x_train_a = np.reshape(x_train_a, (x_train_a.shape[0], seq_maxlen, -1))
                combine_qa_model = get_qa_attention(x_train_q.shape[1:], x_train_a.shape[1:])
                callback = [keras.callbacks.EarlyStopping(monitor='val_auc', patience=1, mode="max", verbose=0), ]
                combine_qa_model.fit([x_train_q, x_train_a], y_train, callbacks=callback, validation_split=0.2, batch_size=128, epochs=10,)
                combine_qa_model.save('models/quatrain_'+str(i+1)+'.h5')

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

            if Sanity:
                x_test_random_q = x_test_random[:, :1024]
                x_test_random_a = x_test_random[:, 1024:]
                x_test_random_q = np.reshape(x_test_random_q, (x_test_random_q.shape[0], seq_maxlen, -1))
                x_test_random_a = np.reshape(x_test_random_a, (x_test_random_a.shape[0], seq_maxlen, -1))
                y_pred_random = combine_qa_model.predict([x_test_random_q, x_test_random_a])[:, 0]
        else:
            y_pred = clf.predict_proba(x_test)[:, 1]
        self.y_pred = y_pred
        self.y_test = y_test

        auc_, recall_p, recall_n, acc, prc, rc, f1 = self.evaluation_metrics(y_true=list(y_test), y_pred_prob=list(y_pred),)
        self.evaluate_message(y_true=list(y_test), y_pred_prob=list(y_pred), test_info_for_patch=self.test_info_for_patch)

        if Sanity:
            self.evaluation_sanity(y_true=list(y_test), y_pred_prob=list(y_pred), y_pred_random_prob=list(y_pred_random))

        if not Sanity and not QualityOfMessage:
            self.confusion_matrix(y_pred, y_test)

        # print('---------------')
        return auc_, recall_p, recall_n, acc, prc, rc, f1
