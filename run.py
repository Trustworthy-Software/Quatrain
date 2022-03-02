import pandas

import experiment.config as config
import os
from representation.word2vec import Word2vector
import pickle
from scipy.spatial import distance
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, average_precision_score
import numpy as np
import experiment.ML4Prediciton as ML4Prediciton
import signal
import json
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
# os.getcwd('./Naturality')
dirname = os.path.dirname(__file__)



class Experiment:
    def __init__(self):
        self.cf = config.Config()
        self.path_patch = self.cf.path_patch
        if not 'Unique' in self.path_patch:
            raise ('please deduplicate it!')
        self.path_patch_sliced = self.cf.path_patch + '_sliced'
        # self.dict_b = {}
        self.bugReportText = None
    def evaluation_metrics(self, y_trues, y_pred_probs):
        fpr, tpr, thresholds = roc_curve(y_true=y_trues, y_score=y_pred_probs, pos_label=1)
        auc_ = auc(fpr, tpr)

        y_preds = [1 if p >= 0.5 else 0 for p in y_pred_probs]

        acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
        prc = precision_score(y_true=y_trues, y_pred=y_preds)
        rc = recall_score(y_true=y_trues, y_pred=y_preds)
        f1 = 2 * prc * rc / (prc + rc)

        print('\n***------------***')
        print('Evaluating AUC, F1, +Recall, -Recall')
        print('Test data size: {}, Incorrect: {}, Correct: {}'.format(len(y_trues), y_trues.count(0), y_trues.count(1)))
        print('Accuracy: %f -- Precision: %f -- +Recall: %f -- F1: %f ' % (acc, prc, rc, f1))
        tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
        recall_p = tp / (tp + fn)
        recall_n = tn / (tn + fp)
        print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(auc_, recall_p, recall_n))
        # return , auc_

        # print('AP: {}'.format(average_precision_score(y_trues, y_pred_probs)))
        return recall_p, recall_n, acc, prc, rc, f1, auc_

    def save_bugreport_deprecated(self, project, embedding_method):
        file_name = '../data/bugreport_dict_'+embedding_method+'.pickle'
        if os.path.exists(file_name):
            return
        w = Word2vector(embedding_method)
        dict_b = {}
        path = '../preprocess/' + project + '_bugreport.txt'
        with open(path, 'r+') as f:
            for line in f:
                project_id = line.split(',')[0]
                bugReport = line.split(',')[1]
                learned_vector = w.embedding(bugReport)
                dict_b[project_id] = learned_vector
        pickle.dump(dict_b, open(file_name, 'wb'))

    def save_bugreport(self,):
        file_name = '../data/bugreport_dict.pickle'
        if os.path.exists(file_name):
            return
        # w = Word2vector(embedding_method)
        dict_b = {}
        path = 'data/BugReport'
        files = os.listdir(path)
        for file in files:
            if not file.endswith('bugreport.txt'):
                continue
            path_bugreport = os.path.join(path, file)
            with open(path_bugreport, 'r+') as f:
                for line in f:
                    project_id = line.split('$$')[0]
                    bugReportSummary = line.split('$$')[1].strip('\n')
                    bugReportDescription = line.split('$$')[2].strip('\n')
                    dict_b[project_id] = [bugReportSummary, bugReportDescription]
        pickle.dump(dict_b, open(file_name, 'wb'))


    def predict_10fold(self, embedding_method, algorithm):
        dataset = pickle.load(open('../data/bugreport_patch_'+embedding_method+'.pickle', 'rb'))
        bugreport_vector = np.array(dataset[0]).reshape((len(dataset[0]),-1))
        commit_vector = np.array(dataset[1]).reshape((len(dataset[1]),-1))
        labels = np.array(dataset[2])

        # combine bug report and commit message of patch
        features = np.concatenate((bugreport_vector, commit_vector), axis=1)
        cl = ML4Prediciton.Classifier(features, labels, algorithm, 10)
        cl.cross_validation()

    def predict_leave1out(self, embedding_method, times, algorithm):
        dataset_json = pickle.load(open(os.path.join(dirname, 'data/bugreport_patch_json_' + embedding_method + '.pickle'), 'rb'))
        # leave one out
        project_ids = list(dataset_json.keys())
        number = len(project_ids)
        accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
        rcs_p, rcs_n = list(), list()
        for i in range(times):
            random.shuffle(project_ids)
            train_ids = project_ids[:int(0.9*number)]
            test_ids = project_ids[int(0.9*number):]

            train_features, train_labels = [], []
            for train_id in train_ids:
                value = dataset_json[train_id]
                bugreport_vector_summary = value[0][0]
                bugreport_vector_description = value[0][1]
                for m in range(1, len(value)):
                    commit_vector, label = value[m][0], value[m][1]
                    features = np.concatenate((bugreport_vector_summary, bugreport_vector_description, commit_vector), axis=1)
                    # features = commit_vector

                    train_features.append(features[0])
                    train_labels.append(label)
            train_features = np.array(train_features)

            test_features, test_labels = [], []
            for test_id in test_ids:
                value = dataset_json[test_id]
                bugreport_vector_summary = value[0][0]
                bugreport_vector_description = value[0][1]
                for n in range(1, len(value)):
                    commit_vector, label = value[n][0], value[n][1]
                    features = np.concatenate((bugreport_vector_summary, bugreport_vector_description, commit_vector), axis=1)
                    # features = commit_vector

                    test_features.append(features[0])
                    test_labels.append(label)
            test_features = np.array(test_features)
            # dataset = np.concatenate((train_features, test_features), axis=0)
            labels = train_labels+test_labels

            if i == 0:
                print('All data size: {}, Incorrect: {}, Correct: {}'.format(len(labels), labels.count(0),
                                                                             labels.count(1)))
                print('Algorithm: {}'.format(algorithm))
                print('#####')

            cl = ML4Prediciton.Classifier(None, None, algorithm, None, train_features, train_labels, test_features, test_labels)
            auc_, recall_p, recall_n, acc, prc, rc, f1 = cl.leave1out_validation()
            accs.append(acc)
            prcs.append(prc)
            rcs.append(rc)
            f1s.append(f1)

            aucs.append(auc_)
            rcs_p.append(recall_p)
            rcs_n.append(recall_n)

        print('')
        print('{} leave one out mean: '.format('10-90'))
        print('Accuracy: {:.1f} -- Precision: {:.1f} -- +Recall: {:.1f} -- F1: {:.1f} -- AUC: {:.3f}'.format(
            np.array(accs).mean() * 100, np.array(prcs).mean() * 100, np.array(rcs).mean() * 100,
            np.array(f1s).mean() * 100, np.array(aucs).mean()))
        print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(np.array(aucs).mean(), np.array(rcs_p).mean(),
                                                                     np.array(rcs_n).mean()))
        print('---------------')

    def predict_leave1out_10fold(self, embedding_method, times, algorithm, ASE):
        dataset_json = pickle.load(open(os.path.join(dirname, 'data/bugreport_patch_json_' + embedding_method + '.pickle'), 'rb'))
        ASE_features = None
        if ASE:
            # ASE_features = pickle.load(open('../data/ASE_features_'+embedding_method+'.pickle', 'rb'))
            ASE_features = pickle.load(open(os.path.join(dirname, 'data/ASE_features_bert.pickle'), 'rb'))
        # leave one out
        project_ids = list(dataset_json.keys())
        n = len(project_ids)
        accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
        a_accs, a_prcs, a_rcs, a_f1s, a_aucs = list(), list(), list(), list(), list()
        rcs_p, rcs_n = list(), list()
        a_rcs_p, a_rcs_n = list(), list()
        meesage_length_distribution = []
        random.seed(1)
        random.shuffle(project_ids,)
        n = int(math.ceil(len(project_ids) / float(times)))
        groups = [project_ids[i:i+n] for i in range(0, len(project_ids), n)]

        for i in range(times):
            test_group = groups[i]
            train_group = groups[:i] + groups[i+1:]

            test_ids = test_group
            train_ids = []
            for j in train_group:
                train_ids += j

            enhance = False
            onlyCorrect = True
            train_features, train_labels, ASE_train_features, ASE_train_labels = self.get_train_data(train_ids, dataset_json, ASE_features, ASE, enhance=enhance)
            test_features, test_labels, ASE_test_features, ASE_test_labels, test_info_for_patch = self.get_test_data(test_ids, dataset_json, ASE_features, ASE, enhance=enhance, onlyCorrect=onlyCorrect)

            if i == 0:
                labels = train_labels + test_labels
                print('All data size: {}, Incorrect: {}, Correct: {}'.format(len(labels), labels.count(0), labels.count(1)))
                print('Algorithm: {}'.format(algorithm))
                print('#####')

            # 1. machine learning classifier
            cl = ML4Prediciton.Classifier(None, None, algorithm, None, train_features, train_labels, test_features, test_labels, test_info_for_patch=test_info_for_patch)
            auc_, recall_p, recall_n, acc, prc, rc, f1, messageL_correctP, messageL_incorrectP = cl.leave1out_validation()

            meesage_length_distribution.append([messageL_correctP, messageL_incorrectP])

            accs.append(acc)
            prcs.append(prc)
            rcs.append(rc)
            f1s.append(f1)

            aucs.append(auc_)
            rcs_p.append(recall_p)
            rcs_n.append(recall_n)

            if ASE:
                cl = ML4Prediciton.Classifier(None, None, algorithm, None, ASE_train_features, ASE_train_labels, ASE_test_features,
                                              ASE_test_labels)
                auc_, recall_p, recall_n, acc, prc, rc, f1 = cl.leave1out_validation()
                a_accs.append(acc)
                a_prcs.append(prc)
                a_rcs.append(rc)
                a_f1s.append(f1)

                a_aucs.append(auc_)
                a_rcs_p.append(recall_p)
                a_rcs_n.append(recall_n)

        print('')
        print('{} leave one out mean: '.format('10-90'))
        print('Accuracy: {:.1f} -- Precision: {:.1f} -- +Recall: {:.1f} -- F1: {:.1f} -- AUC: {:.3f}'.format(
            np.array(accs).mean() * 100, np.array(prcs).mean() * 100, np.array(rcs).mean() * 100,
            np.array(f1s).mean() * 100, np.array(aucs).mean()))
        print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(np.array(aucs).mean(), np.array(rcs_p).mean(),
                                                                     np.array(rcs_n).mean()))
        print('---------------')

        self.message_length_distribution(meesage_length_distribution, aucs)

        if ASE:
            print('')
            print('{} ASE leave one out mean: '.format('10-90'))
            print('Accuracy: {:.1f} -- Precision: {:.1f} -- +Recall: {:.1f} -- F1: {:.1f} -- AUC: {:.3f}'.format(
                np.array(a_accs).mean() * 100, np.array(a_prcs).mean() * 100, np.array(a_rcs).mean() * 100,
                np.array(a_f1s).mean() * 100, np.array(a_aucs).mean()))
            print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(np.array(a_aucs).mean(), np.array(a_rcs_p).mean(),
                                                                        np.array(a_rcs_n).mean()))
            print('---------------')

    def get_train_data(self, train_ids, dataset_json, ASE_features=None, ASE=False, enhance=False):
        train_features, train_labels = [], []
        ASE_train_features, ASE_train_labels = [], []
        for train_id in train_ids:
            value = dataset_json[train_id]
            bugreport_vector = value[0]
            for p in range(1, len(value)):
                train_patch_id = value[p][0]
                commit_vector, label = value[p][1], value[p][2]
                features = np.concatenate((bugreport_vector, commit_vector), axis=1)
                # features = commit_vector

                if enhance:
                    if label == 1:
                        # 6x increase in the number of correct patches
                        for _ in range(6):
                            train_features.append(features[0])
                            train_labels.append(label)
                    else:
                        train_features.append(features[0])
                        train_labels.append(label)
                else:
                    train_features.append(features[0])
                    train_labels.append(label)

            if ASE:
                try:
                    ASE_value = ASE_features[train_id]
                    for p in range(len(ASE_value)):
                        ASE_vector, ASE_label = ASE_value[p][0], ASE_value[p][1]

                        ASE_train_features.append(np.array(ASE_vector))
                        ASE_train_labels.append(ASE_label)
                except Exception as e:
                    print(e)

        train_features = np.array(train_features)
        ASE_train_features = np.array(ASE_train_features)

        return train_features, train_labels, ASE_train_features, ASE_train_labels

    def get_test_data(self, test_ids, dataset_json, ASE_features=None, ASE=False, enhance=False, onlyCorrect=False):
        test_features, test_labels = [], []
        ASE_test_features, ASE_test_labels = [], []
        test_info_for_patch = []
        bug_report_vector_list = [v[0] for k,v in dataset_json.items() if k not in test_ids]
        for test_id in test_ids:
            value = dataset_json[test_id]
            bugreport_vector = value[0]
            for v in range(1, len(value)):
                test_patch_id = value[v][0]
                commit_vector, label = value[v][1], value[v][2]
                features = np.concatenate((bugreport_vector, commit_vector), axis=1)
                # features = commit_vector

                if enhance:
                    if label == 1:
                        for _ in range(6):
                            test_features.append(features[0])
                            test_labels.append(label)
                            test_info_for_patch.append([test_id, test_patch_id])
                    else:
                        test_features.append(features[0])
                        test_labels.append(label)
                        test_info_for_patch.append([test_id, test_patch_id])
                elif onlyCorrect:
                    if label == 1:
                        # random bug report or not:
                        # ranindex = random.randint(0, len(bug_report_vector_list)-1)
                        # random_bug_report = bug_report_vector_list[ranindex]
                        random_bug_report = random.choice(bug_report_vector_list)
                        features = np.concatenate((random_bug_report, commit_vector), axis=1)

                        test_features.append(features[0])
                        test_labels.append(label)
                        test_info_for_patch.append([test_id, test_patch_id])
                    else:
                        pass
                else:
                    test_features.append(features[0])
                    test_labels.append(label)
                    test_info_for_patch.append([test_id, test_patch_id])
            if ASE:
                try:
                    ASE_value = ASE_features[test_id]
                    for p in range(len(ASE_value)):
                        ASE_vector, ASE_label = ASE_value[p][0], ASE_value[p][1]

                        ASE_test_features.append(np.array(ASE_vector))
                        ASE_test_labels.append(ASE_label)
                except Exception as e:
                    print(e)
        test_features = np.array(test_features)
        ASE_test_features = np.array(ASE_test_features)

        return test_features, test_labels, ASE_test_features, ASE_test_labels, test_info_for_patch


    def message_length_distribution(self, meesage_length_distribution, aucs):
        df_length = pd.DataFrame(np.array(meesage_length_distribution), index=['group-'+str(i+1) for i in range(10)], columns=['Correct prediction', 'Incorrect prediction'])
        from sklearn.preprocessing import MinMaxScaler
        df_AUC = pd.DataFrame(MinMaxScaler().fit_transform(np.array(aucs).reshape(-1,1)), index=['group-'+str(i+1) for i in range(10)], columns=['AUC'])

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(111)
        # df_length['Correct prediction'].plot(ax=ax1, kind='bar', color='blue', label='Correct prediction')
        # df_length['Incorrect prediction'].plot(ax=ax1, kind='bar', color='orange', label='Incorrect prediction')
        df_length.plot(ax=ax1, kind='bar', )
        plt.xlabel('Group', fontsize=22)
        ax1.set_ylabel('The number of words in generated commit messages', fontsize=22)
        plt.xticks(fontsize=20, rotation=0)
        plt.yticks(fontsize=20, )
        plt.legend(fontsize=18, loc=2)


        ax2 = ax1.twinx()
        df_AUC['AUC'].plot(ax=ax2, grid=False, label='AUC', style='go-.',)
        ax2.set_ylabel('AUC', fontsize=22)

        plt.xticks(fontsize=20, )
        plt.yticks(fontsize=20, )
        plt.legend(loc=1, fontsize=18,)

        plt.show()

        # img_df.plot(kind='bar', rot=0)
        # plt.xticks(fontsize=20, )
        # plt.yticks(fontsize=20, )
        # plt.legend(fontsize=15, )
        # plt.xlabel('Group', fontsize=22)
        # plt.ylabel('The number of words in generated commit messages', fontsize=22)
        # plt.show()

    def predictASE(self, path_patch_sliced):
        cnt = 0
        available_patchids = []
        deduplicated_patchids = pickle.load(open('utils/deduplicated_name.pickle', 'rb'))
        with open('data/bugreport_patch.txt', 'r+') as f:
            for line in f:
                    # project_id = line.split('$$')[0].strip()
                    bugreport_summary = line.split('$$')[1].strip()
                    # bugreport_description = line.split('$$')[2].strip()
                    patch_id = line.split('$$')[3].strip()
                    commit_content = line.split('$$')[4].strip()
                    # label = int(float(line.split('$$')[5].strip()))
                    # skip none and duplicated cases
                    if bugreport_summary != 'None' and commit_content != 'None' and patch_id in deduplicated_patchids:
                        available_patchids.append(patch_id)
        print('available patch number: {}'.format(len(available_patchids)))
        features_ASE, labels = [], []
        cnt = 0
        for root, dirs, files in os.walk(path_patch_sliced):
            for file in files:
                if file.endswith('$cross.json'):
                    cnt += 1
                    # # file: patch1-Closure-9-Developer-1.patch
                    # name_part = file.split('.')[0]
                    # name = '-'.join(name_part.split('-')[:-1])
                    # label = 1 if root.split('/')[-4] == 'Correct' else 0
                    # project = name.split('-')[1]
                    # id = name.split('-')[2]
                    # project_id = project + '-' + id
                    # feature_json = root + '_cross.json'

                    # file: patch1$.json
                    name = file.split('$cross')[0]
                    id = root.split('/')[-1]
                    project = root.split('/')[-2]
                    label = 1 if root.split('/')[-3] == 'Correct' else 0
                    tool = root.split('/')[-4]
                    patch_id_test = '-'.join([name, project, id, tool])
                    patch_id_tmp = '-'.join([name+'_1', project, id, tool])
                    # only consider the patches that have associated bug report
                    if patch_id_test not in available_patchids and patch_id_tmp not in available_patchids:
                        # print('name: {}'.format(patch_id_test))
                        continue
                    feature_json = os.path.join(root, file)
                    try:
                        with open(feature_json, 'r+') as f:
                            vector_str = json.load(f)
                            vector_ML = np.array(list(map(float, vector_str)))
                    except Exception as e:
                        print(e)
                        continue
                    features_ASE.append(vector_ML)
                    labels.append(label)
                    # print('collecting {}'.format(project_id))
        # print('cnt js: {}'.format(cnt))
        features_ASE = np.array(features_ASE)
        labels = np.array(labels)
        cl = ML4Prediciton.Classifier(features_ASE, labels, 'lr', 10)
        cl.cross_validation()

    def statistics(self, embedding_method,):
        dataset_json = pickle.load(open('../data/bugreport_patch_json_' + embedding_method + '.pickle', 'rb'))
        project_ids = list(dataset_json.keys())

        plt_data = []
        index = []
        for project_id in project_ids:
            value = dataset_json[project_id]
            correct, incorrect = 0, 0
            for p in range(1, len(value)):
                _, label = value[p][0], value[p][1]
                if label == 1:
                    correct +=1
                elif label == 0:
                    incorrect += 1
            index.append(project_id)
            plt_data.append([correct, incorrect])

        print(len(index))
        img_df = pd.DataFrame(np.array(plt_data), index, columns=['correct', 'incorrect'])
        img_df.plot(kind='bar', rot=0)
        plt.show()

if __name__ == '__main__':
    embedding = 'bert'
    e = Experiment()

    # e.statistics(embedding+'(description)')

    # e.predict_10fold(embedding+'(description)', algorithm='lr')
    # e.predict_leave1out(embedding, times=30, algorithm='lr')
    e.predict_leave1out_10fold(embedding, times=10, algorithm='qa_attetion', ASE=False)
    # e.predict_leave1out_10fold(embedding, times=10, algorithm='lr', ASE=False)
