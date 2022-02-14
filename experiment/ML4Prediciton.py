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

class Classifier:
    def __init__(self, dataset, labels, algorithm, kfold, train_features=None, train_labels=None, test_features=None, test_labels=None, test_ids=None):
        self.dataset = dataset
        self.labels = labels
        self.algorithm = algorithm
        self.kfold = kfold

        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.test_ids=test_ids

    def evaluation_metrics(self, y_true, y_pred_prob, test_ids=None):
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred_prob, pos_label=1)
        auc_ = auc(fpr, tpr)

        y_pred = [1 if p >= 0.5 else 0 for p in y_pred_prob]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        prc = precision_score(y_true=y_true, y_pred=y_pred)
        rc = recall_score(y_true=y_true, y_pred=y_pred)
        f1 = 2 * prc * rc / (prc + rc)

        print('Accuracy: %f -- Precision: %f -- +Recall: %f -- F1: %f ' % (acc, prc, rc, f1))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall_p = tp / (tp + fn)
        recall_n = tn / (tn + fp)
        print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(auc_, recall_p, recall_n))

        # 
        for i in range(len(y_pred)):
            if y_pred[i] != y_true:
                print('Incorrect prediction for {}'.format(test_ids[i]))
        # return , auc_
        return auc_, recall_p, recall_n, acc, prc, rc, f1

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

        if self.algorithm == 'xgb':
            x_test_xgb = x_test
            x_test_xgb_dmatrix = xgb.DMatrix(x_test_xgb, label=y_test)
            y_pred = clf.predict(x_test_xgb_dmatrix)
        else:
            y_pred = clf.predict_proba(x_test)[:, 1]

        auc_, recall_p, recall_n, acc, prc, rc, f1 = self.evaluation_metrics(y_true=list(y_test),
                                                                             y_pred_prob=list(y_pred), test_ids=self.test_ids)
        # self.confusion_matrix(y_pred, y_test)   

        print('---------------')
        return auc_, recall_p, recall_n, acc, prc, rc, f1
