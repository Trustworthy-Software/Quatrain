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
import seaborn as sns
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import pearsonr
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
        matrix_average = np.zeros((9,6))
        ASE_matrix_average = np.zeros((9,6))
        all_correct, all_predict_correct, all_random_correct = 0.0, 0.0, 0.0
        meesage_length_distribution, report_length_distribution, similarity_message_distribution = [], [], []
        dataset_distribution = []
        random.seed(1)
        random.shuffle(project_ids,)
        n = int(math.ceil(len(project_ids) / float(times)))
        groups = [project_ids[i:i+n] for i in range(0, len(project_ids), n)]

        all_labels = []
        for train_id in project_ids:
            value = dataset_json[train_id]
            for p in range(1, len(value)):
                label = value[p][2]
                all_labels.append(label)
        print('Dataset size: {}, Incorrect: {}, Correct: {}'.format(len(all_labels), all_labels.count(0), all_labels.count(1)))
        print('Algorithm: {}'.format(algorithm))
        print('#####')


        enhance = False
        Sanity = False
        QualityOfMessage = True
        for i in range(times):
            test_group = groups[i]
            train_group = groups[:i] + groups[i+1:]

            test_ids = test_group
            train_ids = []
            for j in train_group:
                train_ids += j

            # with open('./data/CommitMessage/Developer_commit_message_bert.pickle', 'rb') as f:
            #     Developer_commit_message_dict = pickle.load(f)
            # for data_id in train_ids+test_ids:
            #     value = dataset_json[data_id]
            #     bugreport_vector = value[0]
            #     patch_id = value[1][0]
            #     project, id = patch_id.split('_')[0].split('-')[1], \
            #                   patch_id.split('_')[0].split('-')[2]
            #     project_id = project + '-' + id
            #     developer_commit_message_vector = Developer_commit_message_dict[project_id]


            # train_features, train_labels, ASE_train_features, ASE_train_labels = self.get_train_data(train_ids, dataset_json, ASE_features, ASE, enhance=enhance)
            train_features, train_labels, ASE_train_features, ASE_train_labels = self.get_train_data(train_ids, dataset_json, ASE_features, ASE, enhance=enhance)
            test_features, test_labels, ASE_test_features, ASE_test_labels, random_test_features, test_info_for_patch = self.get_test_data(test_ids, dataset_json, ASE_features, ASE, enhance=enhance, Sanity=Sanity, QualityOfMessage=QualityOfMessage)

            # labels = train_labels + test_labels
            print('Train data size: {}, Incorrect: {}, Correct: {}'.format(len(train_labels), train_labels.count(0), train_labels.count(1)))
            print('Test data size: {}, Incorrect: {}, Correct: {}'.format(len(test_labels), test_labels.count(0), test_labels.count(1)))
            print('#####')
            dataset_distribution.append([len(train_labels), len(test_labels)])
            # classifier
            NLP_model = ML4Prediciton.Classifier(None, None, algorithm, None, train_features, train_labels, test_features, test_labels, random_test_features=random_test_features, test_info_for_patch=test_info_for_patch)
            auc_, recall_p, recall_n, acc, prc, rc, f1= NLP_model.leave1out_validation(Sanity=Sanity, QualityOfMessage=QualityOfMessage)

            if not Sanity and not QualityOfMessage:
                matrix_average += np.array(NLP_model.matrix)
                # length of bug report and commit message
                for m in NLP_model.messageL:
                    m.insert(0, str(i+1))
                for r in NLP_model.reportL:
                    r.insert(0, str(i+1))
                meesage_length_distribution += NLP_model.messageL
                report_length_distribution += NLP_model.reportL

            if Sanity:
                all_correct += NLP_model.correct
                all_predict_correct += NLP_model.predict_correct
                all_random_correct += NLP_model.random_correct
            if QualityOfMessage:
                for s in NLP_model.similarity_message:
                    s.insert(0, str(i+1))
                similarity_message_distribution += NLP_model.similarity_message

            accs.append(acc)
            prcs.append(prc)
            rcs.append(rc)
            f1s.append(f1)

            aucs.append(auc_)
            rcs_p.append(recall_p)
            rcs_n.append(recall_n)

            if ASE:
                ASE_model = ML4Prediciton.Classifier(None, None, 'lr', None, ASE_train_features, ASE_train_labels, ASE_test_features,
                                              ASE_test_labels)
                auc_, recall_p, recall_n, acc, prc, rc, f1 = ASE_model.leave1out_validation()

                ASE_matrix_average += np.array(ASE_model.matrix)
                a_accs.append(acc)
                a_prcs.append(prc)
                a_rcs.append(rc)
                a_f1s.append(f1)

                a_aucs.append(auc_)
                a_rcs_p.append(recall_p)
                a_rcs_n.append(recall_n)

        # self.bar_distribution(dataset_distribution)
        dataset_distribution = np.array(dataset_distribution)
        print('train:test, {}'.format(dataset_distribution[:,0].mean()/dataset_distribution[:,1].mean()))

        if not Sanity and not QualityOfMessage:
            print('RQ-1:')
            print('{} leave one out mean: '.format('10-90'))
            print('Accuracy: {:.1f} -- Precision: {:.1f} -- +Recall: {:.1f} -- F1: {:.1f} -- AUC: {:.3f}'.format(
                np.array(accs).mean() * 100, np.array(prcs).mean() * 100, np.array(rcs).mean() * 100,
                np.array(f1s).mean() * 100, np.array(aucs).mean()))
            print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(np.array(aucs).mean(), np.array(rcs_p).mean(),
                                                                         np.array(rcs_n).mean()))
            print('---------------')

            # confusion matrix
            print('TP _ TN _ FP _ FN _ +Recall _ -Recall')
            np.set_printoptions(suppress=True)
            # calculate average +Recall and -Recall based on all data
            recall_list = []
            for i in range(matrix_average[:,:4].shape[0]):
                tp, tn, fp, fn = matrix_average[i][0], matrix_average[i][1], matrix_average[i][2], matrix_average[i][3]
                recall_p = tp / (tp + fn)
                recall_n = tn / (tn + fp)
                recall_list.append([np.round(recall_p, decimals=3), np.round(recall_n, decimals=3)])

            new_matrix_average = np.concatenate((matrix_average[:,:4], np.array(recall_list)), axis=1)
            print(new_matrix_average)

            print('RQ-2.1:')
            print('[Figure]')

            self.boxplot_distribution(meesage_length_distribution, 'Length of code change description')
            self.boxplot_distribution(report_length_distribution, 'Length of bug report')

        if Sanity:
            # sanity check result
            print('RQ-2.2:')
            print('All correct: {}, Predict correct: {}, Random correct :{}'.format(all_correct, all_predict_correct, all_random_correct))
            print('Fail rate with random: {}'.format((all_predict_correct-all_random_correct)/all_predict_correct))

            pass

        if QualityOfMessage:
            print('RQ-2.3:')
            print('{} leave one out mean: '.format('10-90'))
            print('Accuracy: {:.1f} -- Precision: {:.1f} -- +Recall: {:.1f} -- F1: {:.1f} -- AUC: {:.3f}'.format(
                np.array(accs).mean() * 100, np.array(prcs).mean() * 100, np.array(rcs).mean() * 100,
                np.array(f1s).mean() * 100, np.array(aucs).mean()))
            print('+Recall: {:.3f}'.format(np.array(rcs_p).mean()))
            print('---------------')

            # for the comparison of generated message v.s developer message
            self.boxplot_distribution(similarity_message_distribution, 'Distance between descriptions')

        if ASE:
            print('RQ-3, ASE: ')
            print('{} ASE leave one out mean: '.format('10-90'))
            print('Accuracy: {:.1f} -- Precision: {:.1f} -- +Recall: {:.1f} -- F1: {:.1f} -- AUC: {:.3f}'.format(
                np.array(a_accs).mean() * 100, np.array(a_prcs).mean() * 100, np.array(a_rcs).mean() * 100,
                np.array(a_f1s).mean() * 100, np.array(a_aucs).mean()))
            print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(np.array(a_aucs).mean(), np.array(a_rcs_p).mean(),
                                                                        np.array(a_rcs_n).mean()))
            print('---------------')

            # confusion matrix
            print('TP _ TN _ FP _ FN _ +Recall _ -Recall')
            np.set_printoptions(suppress=True)
            # calculate average +Recall and -Recall based on all data
            recall_list = []
            for i in range(ASE_matrix_average[:,:4].shape[0]):
                tp, tn, fp, fn = ASE_matrix_average[i][0], ASE_matrix_average[i][1], ASE_matrix_average[i][2], ASE_matrix_average[i][3]
                recall_p = tp / (tp + fn)
                recall_n = tn / (tn + fp)
                recall_list.append([np.round(recall_p, decimals=3), np.round(recall_n, decimals=3)])

            new_ASE_matrix_average = np.concatenate((ASE_matrix_average[:,:4], np.array(recall_list)), axis=1)
            print(new_ASE_matrix_average)

    def get_train_data_deprecated(self, train_ids, dataset_json, ASE_features=None, ASE=False, enhance=False):
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

    def get_train_data(self, train_ids, dataset_json, ASE_features=None, ASE=False, enhance=False):
        with open('./data/CommitMessage/Developer_commit_message_bert.pickle', 'rb') as f:
            Developer_commit_message_dict = pickle.load(f)
        train_features, train_labels = [], []
        ASE_train_features, ASE_train_labels = [], []
        random_bug_report_vector_list = [v[0] for k, v in dataset_json.items() if (k in train_ids)]

        for train_id in train_ids:
            value = dataset_json[train_id]
            bugreport_vector = value[0]
            for p in range(1, len(value)):
                train_patch_id = value[p][0]
                project, id = train_patch_id.split('_')[0].split('-')[1], train_patch_id.split('_')[0].split('-')[2]
                project_id = project + '-' + id
                commit_vector, label = value[p][1], value[p][2]
                if project_id == 'closure-63':
                    developer_commit_message_vector = Developer_commit_message_dict['closure-62']
                elif project_id == 'closure-93':
                    developer_commit_message_vector = Developer_commit_message_dict['closure-92']
                else:
                    developer_commit_message_vector = Developer_commit_message_dict[project_id]

                # random_bug_report_vector_list = [v[0] for k, v in dataset_json.items() if (k in train_ids and k != train_id)]

                if '_Developer_' in train_patch_id:
                    # nprandom = np.random.RandomState(1)
                    random.seed(10)
                    # for _ in range(6):
                    features = np.concatenate((bugreport_vector, developer_commit_message_vector), axis=1)
                    train_features.append(features[0])
                    train_labels.append(label)

                    features_random = np.concatenate((random.choice(random_bug_report_vector_list), developer_commit_message_vector), axis=1)
                    train_features.append(features_random[0])
                    train_labels.append(0)
                else:
                    features = np.concatenate((bugreport_vector, commit_vector), axis=1)
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

    def get_test_data(self, test_ids, dataset_json, ASE_features=None, ASE=False, enhance=False, Sanity=False, QualityOfMessage=False):
        with open('./data/CommitMessage/Developer_commit_message_bert.pickle', 'rb') as f:
            Developer_commit_message_dict = pickle.load(f)
        test_features, test_labels = [], []
        random_test_features = []
        ASE_test_features, ASE_test_labels = [], []
        test_info_for_patch = []

        for test_id in test_ids:
            value = dataset_json[test_id]
            bugreport_vector = value[0]
            for v in range(1, len(value)):
                test_patch_id = value[v][0]
                project, id = test_patch_id.split('_')[0].split('-')[1], test_patch_id.split('_')[0].split('-')[2]
                project_id = project + '-' + id
                commit_vector, label = value[v][1], value[v][2]
                features = np.concatenate((bugreport_vector, commit_vector), axis=1)
                if project_id == 'closure-63':
                    developer_commit_message_vector = Developer_commit_message_dict['closure-62']
                elif project_id == 'closure-93':
                    developer_commit_message_vector = Developer_commit_message_dict['closure-92']
                else:
                    developer_commit_message_vector = Developer_commit_message_dict[project_id]

                if Sanity:
                    # only consider developer patches and corresponding dev.written commit messages
                    # bug report vs random bug report
                    if '_Developer_' in test_patch_id:
                    # if label == 1:
                        # 1.1 the best
                        features = np.concatenate((bugreport_vector, developer_commit_message_vector), axis=1)
                        test_features.append(features[0])
                        test_labels.append(label)
                        test_info_for_patch.append([test_id, test_patch_id])
                        # 1.2 random bug report
                        bug_report_vector_list = [v[0] for k, v in dataset_json.items() if (k in test_ids and k != test_id)]
                        random.seed(100)
                        random_bug_report = random.choice(bug_report_vector_list)
                        random_features = np.concatenate((random_bug_report, developer_commit_message_vector), axis=1)
                        random_test_features.append(random_features[0])
                    else:
                        pass
                elif QualityOfMessage:
                    # only consider developer patches
                    # dev.written commit message vs generated commit message
                    if '_Developer_' in test_patch_id:
                        # 2. generated commit message
                        features = np.concatenate((bugreport_vector, commit_vector), axis=1)

                        test_features.append(features[0])
                        test_labels.append(label)
                        test_info_for_patch.append([test_id, test_patch_id])
                    else:
                        pass
                else:
                    if label == 0:
                        test_features.append(features[0])
                        test_labels.append(label)
                    else:
                        if '_Developer_' in test_patch_id:
                            features = np.concatenate((bugreport_vector, developer_commit_message_vector), axis=1)
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

        return test_features, test_labels, ASE_test_features, ASE_test_labels, random_test_features, test_info_for_patch


    def plot_distribution(self, distribution, aucs, y_title):
        df_length = pd.DataFrame(np.array(distribution), index=['group-'+str(i+1) for i in range(10)], columns=['Correct prediction', 'Incorrect prediction'])
        from sklearn.preprocessing import MinMaxScaler
        df_AUC = pd.DataFrame(MinMaxScaler().fit_transform(np.array(aucs).reshape(-1,1)), index=['group-'+str(i+1) for i in range(10)], columns=['AUC'])

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(111)
        # df_length['Correct prediction'].plot(ax=ax1, kind='bar', color='blue', label='Correct prediction')
        # df_length['Incorrect prediction'].plot(ax=ax1, kind='bar', color='orange', label='Incorrect prediction')
        df_length.plot(ax=ax1, kind='bar', )
        plt.xlabel('Group', fontsize=22)
        ax1.set_ylabel(y_title, fontsize=22)
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

    def bar_distribution(self, dataset_distribution):
        df_length = pd.DataFrame(np.array(dataset_distribution), index=[str(i+1) for i in range(10)], columns=['Train', 'Test'])
        # df_AUC = pd.DataFrame(MinMaxScaler().fit_transform(np.array(aucs).reshape(-1,1)), index=['group-'+str(i+1) for i in range(10)], columns=['AUC'])

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(111)
        # df_length['Correct prediction'].plot(ax=ax1, kind='bar', color='blue', label='Correct prediction')
        # df_length['Incorrect prediction'].plot(ax=ax1, kind='bar', color='orange', label='Incorrect prediction')
        df_length.plot(ax=ax1, kind='bar', color={"Train": "White", "Test": "Grey"}, edgecolor='black')
        plt.xlabel('Round', fontsize=28)
        ax1.set_ylabel('The number of patches', fontsize=28)
        plt.xticks(fontsize=22, rotation=0)
        plt.yticks(fontsize=22, )
        plt.ylim((0, 11000))
        plt.legend(fontsize=20, loc=2)
        plt.show()


        # ax2 = ax1.twinx()
        # df_AUC['AUC'].plot(ax=ax2, grid=False, label='AUC', style='go-.',)
        # ax2.set_ylabel('AUC', fontsize=22)
        # plt.xticks(fontsize=20, )
        # plt.yticks(fontsize=20, )
        # plt.legend(loc=1, fontsize=18,)


        # img_df.plot(kind='bar', rot=0)
        # plt.xticks(fontsize=20, )
        # plt.yticks(fontsize=20, )
        # plt.legend(fontsize=15, )
        # plt.xlabel('Group', fontsize=22)
        # plt.ylabel('The number of words in generated commit messages', fontsize=22)
        # plt.show()


    def boxplot_distribution(self, distribution, y_title):
        dfl = pd.DataFrame(distribution)
        dfl.columns = ['Group', 'Prediction', y_title]
        # put H on left side in plot
        if dfl.iloc[0]['Prediction'] != 'Correct':
            b, c = dfl.iloc[0].copy(), dfl[dfl['Prediction']=='Correct'].iloc[0].copy()
            dfl.iloc[0], dfl[dfl['Prediction']=='Correct'].iloc[0] = c, b
        colors = {'Correct': 'white', 'Incorrect': 'grey'}
        fig = plt.figure(figsize=(15, 8))
        plt.xticks(fontsize=28, )
        plt.yticks(fontsize=28, )
        bp = sns.boxplot(x='Group', y=y_title, data=dfl, showfliers=False, palette=colors, hue='Prediction', width=0.7, )
        # bp.set_xticklabels(bp.get_xticklabels(), rotation=320)
        bp.set_xticklabels(bp.get_xticklabels())
        # bp.set_xticklabels(bp.get_xticklabels(), fontsize=28)
        # bp.set_yticklabels(bp.get_yticklabels(), fontsize=28)
        plt.xlabel('Group', size=31)
        plt.ylabel(y_title, size=30)
        plt.legend(fontsize=22, loc=1)
        # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=3, fontsize=30, )
        self.adjust_box_widths(fig, 0.8)
        plt.tight_layout()
        plt.show()

        # distribution = np.array(distribution)
        # correct_list = []
        # distribution_c = distribution[np.array(distribution)[:, 1] == 'Correct']
        # for i in range(1, 11):
        #     mean_number = distribution_c[distribution_c[:, 0] == (str(i))][:,2].astype(int).mean()
        #     correct_list.append(mean_number)
        #
        # incorrect_list = []
        # distribution_inc = distribution[np.array(distribution)[:, 1] == 'Incorrect']
        # for i in range(1, 11):
        #     mean_number = distribution_inc[distribution_inc[:, 0] == (str(i))][:,2].astype(int).mean()
        #     incorrect_list.append(mean_number)

        # MWW test
        length_correct_list = dfl[dfl.iloc[:]['Prediction'] == 'Correct'][y_title].tolist()
        length_incorrect_list = dfl[dfl.iloc[:]['Prediction'] == 'Incorrect'][y_title].tolist()
        try:
            hypo = stats.mannwhitneyu(length_correct_list, length_incorrect_list, alternative='two-sided')
            # hypo = stats.mannwhitneyu(correct_list, incorrect_list, alternative='two-sided')
            p_value = hypo[1]
        except Exception as e:
            if 'identical' in e:
                p_value = 1
        print('p-value: {}'.format(p_value))
        if p_value < 0.05:
            print('Significant!')
        else:
            print('NOT Significant!')

    def adjust_box_widths(self, g, fac):
        """
        Adjust the widths of a seaborn-generated boxplot.
        """
        # iterating through Axes instances
        for ax in g.axes:
            # iterating through axes artists:
            for c in ax.get_children():

                # searching for PathPatches
                if isinstance(c, PathPatch):
                    # getting current width of box:
                    p = c.get_path()
                    verts = p.vertices
                    verts_sub = verts[:-1]
                    xmin = np.min(verts_sub[:, 0])
                    xmax = np.max(verts_sub[:, 0])
                    xmid = 0.5 * (xmin + xmax)
                    xhalf = 0.5 * (xmax - xmin)

                    # setting new width of box
                    xmin_new = xmid - fac * xhalf
                    xmax_new = xmid + fac * xhalf
                    verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                    verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                    # setting new width of median line
                    for l in ax.lines:
                        if np.all(l.get_xdata() == [xmin, xmax]):
                            l.set_xdata([xmin_new, xmax_new])


    def plot_distribution2(self, distribution, y_title):
        df_length = pd.DataFrame(np.array(distribution), index=['group-'+str(i+1) for i in range(10)], columns=['Correct prediction', 'Incorrect prediction'])
        from sklearn.preprocessing import MinMaxScaler

        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(111)
        # df_length['Correct prediction'].plot(ax=ax1, kind='bar', color='blue', label='Correct prediction')
        # df_length['Incorrect prediction'].plot(ax=ax1, kind='bar', color='orange', label='Incorrect prediction')
        df_length.plot(ax=ax1, kind='bar', )
        plt.xlabel('Group', fontsize=22)
        ax1.set_ylabel(y_title, fontsize=22)
        plt.xticks(fontsize=20, )
        plt.yticks(fontsize=20, )
        plt.ylim((0.9, 0.94))
        plt.legend(loc=1, fontsize=18,)

        plt.show()

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
    e.predict_leave1out_10fold(embedding, times=10, algorithm='qa_attetion', ASE=True)
    # e.predict_leave1out_10fold(embedding, times=10, algorithm='lr', ASE=False)
