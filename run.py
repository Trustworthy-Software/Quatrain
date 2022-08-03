import pandas

import experiment.config as config
import os
from representation.word2vec import Word2vector
import pickle
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, AffinityPropagation
from pyclustering.cluster.xmeans import xmeans, splitting_type
from sklearn.metrics import silhouette_score ,calinski_harabasz_score, davies_bouldin_score
import distance._levenshtein
from scipy.spatial import distance as dis
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, average_precision_score
import numpy as np
import experiment.ML4Prediciton as ML4Prediciton
import experiment.API as API
import signal
import json
import random
import math
import pandas as pd
import seaborn as sns
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
# os.getcwd('./Quatrain')
dirname = os.path.dirname(__file__)

import sys


class Experiment:
    def __init__(self):
        self.cf = config.Config()
        self.embedding = self.cf.embedding
        self.dataset_json = self.cf.dataset_json
        self.path_patch = self.cf.path_patch
        self.path_ASE2020_feature = self.cf.path_ASE2020_feature
        if not 'Unique' in self.path_patch:
            raise ('please deduplicate it!')
        self.path_patch_sliced = self.cf.path_patch + '_sliced'
        # self.dict_b = {}
        self.bugReportText = None
    def evaluation_metrics(self, y_trues, y_pred_probs, RQ=None):
        fpr, tpr, thresholds = roc_curve(y_true=y_trues, y_score=y_pred_probs, pos_label=1)
        auc_ = auc(fpr, tpr)

        y_preds = [1 if p >= 0.5 else 0 for p in y_pred_probs]

        acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
        prc = precision_score(y_true=y_trues, y_pred=y_preds)
        rc = recall_score(y_true=y_trues, y_pred=y_preds)
        f1 = 2 * prc * rc / (prc + rc)

        print('***------------***')
        # print('Evaluating AUC, F1, +Recall, -Recall')
        print('Test data size: {}, Incorrect: {}, Correct: {}'.format(len(y_trues), y_trues.count(0), y_trues.count(1)))
        # print('AUC: %f -- F1: %f  -- Accuracy: %f -- Precision: %f ' % (auc_, f1, acc, prc,))
        print('AUC: %f -- F1: %f ' % (auc_, f1,))

        if y_trues == y_preds:
            tn, fp, fn, tp = 1, 0, 0, 1
        else:
            tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
        recall_p = tp / (tp + fn)
        recall_n = tn / (tn + fp)
        # print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(auc_, recall_p, recall_n))
        # return , auc_

        # print('AP: {}'.format(average_precision_score(y_trues, y_pred_probs)))
        return recall_p, recall_n, acc, prc, rc, f1, auc_

    def evaluation_metrics_noprint(self, y_trues, y_pred_probs):
        fpr, tpr, thresholds = roc_curve(y_true=y_trues, y_score=y_pred_probs, pos_label=1)
        auc_ = auc(fpr, tpr)

        y_preds = [1 if p >= 0.5 else 0 for p in y_pred_probs]

        acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
        prc = precision_score(y_true=y_trues, y_pred=y_preds)
        rc = recall_score(y_true=y_trues, y_pred=y_preds)
        f1 = 2 * prc * rc / (prc + rc)

        # print('***------------***')
        # # print('Evaluating AUC, F1, +Recall, -Recall')
        # print('Test data size: {}, Incorrect: {}, Correct: {}'.format(len(y_trues), y_trues.count(0), y_trues.count(1)))
        # print('Accuracy: %f -- Precision: %f -- +Recall: %f -- F1: %f ' % (acc, prc, rc, f1))
        if y_trues == y_preds:
            tn, fp, fn, tp = 1, 0, 0, 1
        else:
            tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
        recall_p = tp / (tp + fn)
        recall_n = tn / (tn + fp)
        # print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(auc_, recall_p, recall_n))
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

    def validate_hypothesis(self, embedding_method):
        self.validateByVector(embedding_method)
        # self.validateBysString()

    def validateBysString(self,):
        cnt = 0
        distribution_distance = []
        bug_report_txts = []
        developer_commits = []
        with open('./data/bugreport_patch.txt', 'r+') as f:
            for line in f:
                project_id = line.split('$$')[0].strip()
                bugreport_summary = line.split('$$')[1].strip()
                bugreport_description = line.split('$$')[2].strip()

                patch_id = line.split('$$')[3].strip()
                commit_content = line.split('$$')[4].strip()
                label = int(float(line.split('$$')[5].strip()))
                if 'Developer' not in patch_id or bugreport_summary == 'None' or commit_content == 'None':
                    continue
                bug_report_txt = bugreport_summary + '. ' + bugreport_description
                bug_report_txts.append(bug_report_txt)
                developer_commits.append(commit_content)
                cnt +=1
        for i in range(len(bug_report_txts)):
            br = bug_report_txts[i]
            dc = developer_commits[i]
            random_ind = random.randint(0, len(bug_report_txts)-1)
            dc_r = developer_commits[random_ind]

            dis_ground = distance.levenshtein(br, dc)
            dis_random = distance.levenshtein(br, dc_r)

            distribution_distance.append(['Original pairs' , dis_ground])
            distribution_distance.append(['Random pairs' , dis_random])

        y_title = 'distribution of distance'
        self.boxplot_distribution_correlation(distribution_distance, y_title, 'Original pairs', 'Random pairs')


    def validateByVector(self, embedding_method):
        dataset_json = pickle.load(open(os.path.join(dirname, 'data/bugreport_patch_json_' + embedding_method + '.pickle'), 'rb'))
        # dataset_json = pickle.load(open(os.path.join(dirname, 'data/bugreport_patch_json_onlysummary_bert.nocode'), 'rb'))
        with open('./data/CommitMessage/Developer_commit_message_bert.pickle', 'rb') as f:
            Developer_commit_message_dict = pickle.load(f)
        all_bug_report, all_developer_commit = [], []
        project_ids = []

        for k,v in dataset_json.items():
            value = v
            bugreport_vector = value[0]
            project_id = k
            if project_id == 'closure-63':
                developer_commit_message_vector = Developer_commit_message_dict['closure-62']
            elif project_id == 'closure-93':
                developer_commit_message_vector = Developer_commit_message_dict['closure-92']
            else:
                developer_commit_message_vector = Developer_commit_message_dict[project_id]
            bugreport_vector = bugreport_vector.flatten()
            developer_commit_message_vector = developer_commit_message_vector.flatten()

            project_ids.append(project_id)
            all_bug_report.append(bugreport_vector)
            all_developer_commit.append(developer_commit_message_vector)

        # 1.
        # clusters, number = self.cluster_bug_report(all_bug_report, 'xmeans', 0)
        # self.patch_commit(all_developer_commit, clusters, number)
        # # random cluster
        # self.cluster_patch_commit(all_developer_commit, number)

        scaler = StandardScaler()
        all_bug_report = scaler.fit_transform(np.array(all_bug_report))
        scaler = StandardScaler()
        all_developer_commit = scaler.fit_transform(np.array(all_developer_commit))
        # 2.
        distribution_distance = []
        for i in range(all_bug_report.shape[0]):
            # print(project_ids[i])
            br = all_bug_report[i]
            dc = all_developer_commit[i]
            random_ind = random.randint(0, all_bug_report.shape[0]-1)
            dc_r = all_developer_commit[random_ind]

            dis_ground = dis.euclidean(br, dc)
            dis_random = dis.euclidean(br, dc_r)
            # if dis_ground < 0.1:
            #     continue
            distribution_distance.append(['Original pairs' , dis_ground])
            distribution_distance.append(['Random pairs' , dis_random])

        y_title = 'Distance of pairs'
        self.boxplot_distribution_correlation(distribution_distance, y_title, 'Original pairs', 'Random pairs', 'figure3_hypothesis_validation.jpg')

    def boxplot_distribution_correlation(self, distribution_distance, y_title, group1, group2, figureName):

        dfl = pd.DataFrame(distribution_distance)
        dfl.columns = ['Category', y_title]
        # put H on left side in plot
        # if dfl.iloc[0]['Category'] != 'Original pairs':
        #     b, c = dfl.iloc[0].copy(), dfl[dfl['Category']=='Original pairs'].iloc[0].copy()
        #     dfl.iloc[0], dfl[dfl['Category']=='Original pairs'].iloc[0] = c, b
        colors = {group1: 'white', group2: 'grey'}
        fig = plt.figure(figsize=(14, 5))
        plt.xticks(fontsize=28, )
        plt.yticks(fontsize=28, )

        bp = sns.boxplot(x=y_title, y='Category', data=dfl, showfliers=False, palette=colors, width=0.5, orient='h', notch=False)
        # bp = sns.stripplot(x=y_title, y='Category', data=dfl, alpha = 0.2, color = 'blue')
        # bp.set_xticklabels(bp.get_xticklabels(), rotation=320)
        # bp.set_xticklabels(bp.get_xticklabels())
        # bp.set_xticklabels(bp.get_xticklabels(), fontsize=28)
        # bp.set_yticklabels(bp.get_yticklabels(), fontsize=28)
        plt.xlabel(y_title, size=31)
        plt.ylabel('', size=30)
        # plt.legend(fontsize=22, loc=1)
        # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=3, fontsize=30, )
        # self.adjust_box_widths(fig, 0.8)
        # plt.tight_layout()
        plt.subplots_adjust(bottom=0.3, left=0.2)

        plt.savefig('./figure/' + figureName)
        print('The figure is saved to ./figure/' + figureName)
        plt.show()

        # MWW test
        length_correct_list = dfl[dfl.iloc[:]['Category'] == group1][y_title].tolist()
        length_incorrect_list = dfl[dfl.iloc[:]['Category'] == group2][y_title].tolist()
        try:
            hypo = stats.mannwhitneyu(length_correct_list, length_incorrect_list, alternative='two-sided')
            # hypo = stats.mannwhitneyu(correct_list, incorrect_list, alternative='two-sided')
            p_value = hypo[1]
        except Exception as e:
            if 'identical' in e:
                p_value = 1
        print('p-value: {}'.format(p_value))
        if p_value <= 0.05:
            print('Reject Null Hypothesis: Significantly different!')
        else:
            print('Support Null Hypothesis!')

    def cluster_bug_report(self, vectors, method, number):
        scaler = Normalizer()
        vectors = scaler.fit_transform(vectors)
        X = pd.DataFrame(vectors)

        # original distance as one cluster
        center_one = np.mean(X, axis=0)
        dists_one = [distance.euclidean(vec, np.array(center_one)) for vec in np.array(X)]

        if method == 'kmeans':
            kmeans = KMeans(n_clusters=number, random_state=1)
            # kmeans.fit(np.array(test_vector))
            clusters = kmeans.fit_predict(X)
        elif method == 'dbscan':
            db = DBSCAN(eps=0.5, min_samples=5)
            clusters = db.fit_predict(X)
            number = max(clusters)+2
        elif method == 'hier':
            hu = AgglomerativeClustering(n_clusters=number)
            clusters = hu.fit_predict(X)
        elif method == 'xmeans':
            xmeans_instance = xmeans(X, kmax=200, splitting_type=splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH)
            clusters = xmeans_instance.process().predict(X)
            # clusters = xmeans_instance.process().get_clusters()
            number = max(clusters)+1
        elif method == 'biKmeans':
            bk = biKmeans()
            clusters = bk.biKmeans(dataSet=np.array(X), k=number)
        elif method == 'ap':
            # ap = AffinityPropagation(random_state=5)
            # clusters = ap.fit_predict(X)
            APC = AffinityPropagation(verbose=True, max_iter=200, convergence_iter=25).fit(X)
            APC_res = APC.predict(X)
            clusters = APC.cluster_centers_indices_
        else:
            raise
        # X["Cluster"] = clusters
        print('Cluster bug report------')
        SC_list = self.score_inside_outside(X, clusters, number)

        # s1 = silhouette_score(X, clusters)
        # s2 = calinski_harabasz_score(X, clusters)
        # s3 = davies_bouldin_score(X, clusters)
        # print('number: {}'.format(number))
        # print('Silhouette: {}'.format(s1))
        # print('CH: {}'.format(s2))
        # print('DBI: {}'.format(s3))

        return clusters, number

    def patch_commit(self, all_developer_commit, clusters, number):
        scaler = Normalizer()
        P = pd.DataFrame(scaler.fit_transform(all_developer_commit))

        print('Associated patch commit------')
        SC_list = self.score_inside_outside(P, clusters, number)

        # s1 = silhouette_score(P, clusters)
        # s2 = calinski_harabasz_score(P, clusters)
        # s3 = davies_bouldin_score(P, clusters)
        # print('Silhouette: {}'.format(s1))
        # print('CH: {}'.format(s2))
        # print('DBI: {}'.format(s3))

    def cluster_patch_commit(self, all_developer_commit, number):
        scaler = Normalizer()
        P = pd.DataFrame(scaler.fit_transform(all_developer_commit))
        clusters = [random.randint(0, number-1) for i in range(len(all_developer_commit))]
        clusters = np.array(clusters)
        print('Random cluster patch Commit------')
        SC_list = self.score_inside_outside(P, clusters, number)

        # s1 = silhouette_score(P, clusters)
        # s2 = calinski_harabasz_score(P, clusters)
        # s3 = davies_bouldin_score(P, clusters)
        # print('number: {}'.format(number))
        # print('Silhouette: {}'.format(s1))
        # print('CH: {}'.format(s2))
        # print('DBI: {}'.format(s3))


    def score_inside_outside(self, vectors, clusters, number):
        cnt = 0
        SC_list = []
        print('Calculating...')
        for n in range(number):
            # print('cluster index: {}'.format(n))
            index_inside = np.where(clusters == n)
            score_inside_mean = []
            score_outside_mean = []
            vector_inside = vectors.iloc[index_inside]
            for i in range(vector_inside.shape[0]):
                cur = vector_inside.iloc[i]

                # compared to vectors inside this cluster
                for j in range(i+1, vector_inside.shape[0]):
                    cur2 = vector_inside.iloc[j]
                    dist = distance.euclidean(cur, cur2) / (1 + distance.euclidean(cur, cur2))
                    score = 1 - dist
                    score_inside_mean.append(score)

                # compared to vectors outside the cluster
                index_outside = np.where(clusters!=n)
                vector_outside = vectors.iloc[index_outside]
                for k in range(vector_outside.shape[0]):
                    cur3 = vector_outside.iloc[k]
                    dist = distance.euclidean(cur, cur3) / (1 + distance.euclidean(cur, cur3))
                    score = 1 - dist
                    score_outside_mean.append(score)

            inside_score = np.array(score_inside_mean).mean()
            outside_score = np.array(score_outside_mean).mean()
            # print('inside: {}'.format(inside_score), end='    ')
            # print('outside: {}'.format(outside_score))

            SC = (inside_score - outside_score) / max(inside_score, outside_score)
            SC_list.append(SC)


        CSC = np.array(SC_list).mean()
        print('Qualified: {}/{}'.format(len(np.where(np.array(SC_list)>0)[0]), len(SC_list)))
        print('CSC: {}'.format(CSC))

        return SC_list

    def predict_10fold(self, embedding_method, algorithm):
        dataset = pickle.load(open(os.path.join(dirname, 'data/bugreport_patch_array_'+embedding_method+'.pickle'), 'rb'))
        bugreport_vector = np.array(dataset[0]).reshape((len(dataset[0]),-1))
        commit_vector = np.array(dataset[1]).reshape((len(dataset[1]),-1))
        labels = np.array(dataset[2])

        # combine bug report and commit message of patch
        features = np.concatenate((bugreport_vector, commit_vector), axis=1)
        cl = ML4Prediciton.Classifier(features, labels, algorithm, 10)
        auc_10fold, f1_10fold = cl.cross_validation()
        return auc_10fold, f1_10fold

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

    def predict_leave1out_10group(self, dataset_json, embedding_method, times, algorithm, comparison, para=None, Sanity=False, QualityOfMessage=False, RQ=None):
        self.ASE_features = None
        if comparison == 'DL':
            # self.ASE_features = pickle.load(open(os.path.join(dirname, 'data/ASE_features2_bert.pickle'), 'rb'))
            self.ASE_features = pickle.load(open(self.path_ASE2020_feature, 'rb'))
        elif comparison == 'BATS':
            if para == '0.0':
                with open(os.path.join(dirname, 'data/BATS_RESULT_0.0.json'), 'r+') as f:
                    self.BATS_RESULTS_json = json.load(f)
            elif para == '0.8':
                with open(os.path.join(dirname, 'data/BATS_RESULT_0.8.json'), 'r+') as f:
                    self.BATS_RESULTS_json = json.load(f)
        elif comparison == 'PATCHSIM':
            with open(os.path.join(dirname, 'data/PATCHSIM_RESULT.json'), 'r+') as f:
                self.PATCHSIM_RESULTS_json = json.load(f)
        # leave one out
        project_ids = list(dataset_json.keys())
        n = len(project_ids)
        accs, prcs, rcs, f1s, aucs = list(), list(), list(), list(), list()
        a_accs, a_prcs, a_rcs, a_f1s, a_aucs, a_rcs_p, a_rcs_n = list(), list(), list(), list(), list(), list(), list()
        bp_aucs, bp_f1s, bp_rcs_p, bp_rcs_n = list(), list(), list(), list()
        rcs_p, rcs_n = list(), list()
        NLP_model_ytests, NLP_model_preds, ASE_model_ytests, ASE_model_preds, BatsPatchSim_model_preds = [], [], [], [], []
        new_identify_cnt, identify_cnt = 0, 0
        test_patches_info = []
        matrix_average = np.zeros((9,6))
        ASE_matrix_average = np.zeros((9,6))
        # all_correct, all_predict_correct, all_random_correct = 0.0, 0.0, 0.0
        all_correct, all_predict_correct, all_random_correct = 0.0, [], []
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
        # print('Algorithm: {}'.format(algorithm))
        print('#####')


        # Sanity = False
        # QualityOfMessage = False
        for i in range(times):
            print('***************')
            print('ROUND: {}'.format(str(i+1)))
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

            self.RQ = RQ
            train_features, train_labels, ASE_train_features, ASE_train_labels = self.get_train_data(train_ids, dataset_json, comparison, QualityOfMessage=QualityOfMessage)
            test_features, test_labels, ASE_test_features, ASE_test_labels, random_test_features, test_info_for_patch = self.get_test_data(test_ids, dataset_json, comparison, para, Sanity=Sanity, QualityOfMessage=QualityOfMessage)

            # labels = train_labels + test_labels
            print('Train data size: {}, Incorrect: {}, Correct: {}'.format(len(train_labels), train_labels.count(0), train_labels.count(1)))
            print('Test data size: {}, Incorrect: {}, Correct: {}'.format(len(test_labels), test_labels.count(0), test_labels.count(1)))
            print('#####')
            dataset_distribution.append([len(train_labels), len(test_labels)])
            # classifier
            NLP_model = ML4Prediciton.Classifier(None, None, algorithm, None, train_features, train_labels, test_features, test_labels, random_test_features=random_test_features, test_info_for_patch=test_info_for_patch)
            print('Quatrain.')
            auc_, recall_p, recall_n, acc, prc, rc, f1 = NLP_model.leave1out_validation(i, Sanity=Sanity, QualityOfMessage=QualityOfMessage)

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

            # for average results
            NLP_model_preds += list(NLP_model.y_pred)
            NLP_model_ytests += list(NLP_model.y_test)
            test_patches_info += test_info_for_patch

            if comparison == 'DL':
                ASE_model = ML4Prediciton.Classifier(None, None, para, None, ASE_train_features, ASE_train_labels, ASE_test_features,
                                              ASE_test_labels)
                print('a DL-based patch classifier.')
                auc_, recall_p, recall_n, acc, prc, rc, f1 = ASE_model.leave1out_validation(i)

                # a_accs.append(acc)
                # a_prcs.append(prc)
                # a_rcs.append(rc)
                # a_f1s.append(f1)
                #
                # a_aucs.append(auc_)
                # a_rcs_p.append(recall_p)
                # a_rcs_n.append(recall_n)

                ASE_matrix_average += np.array(ASE_model.matrix)
                ASE_model_preds += list(ASE_model.y_pred)
                ASE_model_ytests += list(ASE_model.y_test)
            elif comparison == 'BATS' or comparison == 'PATCHSIM':
                recall_p, recall_n, acc, prc, rc, f1, auc_ = self.evaluation_metrics(NLP_model.y_test, self.comparison_pred)
                bp_aucs.append(auc_)
                bp_f1s.append(f1)
                bp_rcs_p.append(recall_p)
                bp_rcs_n.append(recall_n)
                BatsPatchSim_model_preds += list(self.comparison_pred)

        if RQ == 'RQ1' and not Sanity and not QualityOfMessage:
            print('############################################')
            if para == '':
                print('RQ1, Effectiveness of Quatrain.')
                dataset_distribution = np.array(dataset_distribution)
                print('Figure 6: Distribution of Patches --- Train data:Test data, {}:{}'.format(round(dataset_distribution[:, 0].mean() / dataset_distribution[:, 1].mean()), 1))
                self.bar_distribution(dataset_distribution)
            elif para == 'balance':
                print('RQ1-balance, Effectiveness of Quatrain.')
                # balance test data for F1
                index_1 = [i for i in range(len(NLP_model_ytests)) if NLP_model_ytests[i]==1]
                index_0 = [j for j in range(len(NLP_model_ytests)) if NLP_model_ytests[j]==0]
                random.shuffle(index_0)
                index_0_small = index_0[:1591]
                index_final = index_0_small + index_1
                NLP_model_ytests_4f1 = [NLP_model_ytests[m] for m in index_final]
                NLP_model_preds_4f1 = [NLP_model_preds[n] for n in index_final]
                print('The improved F1')
                self.evaluation_metrics(NLP_model_ytests_4f1, NLP_model_preds_4f1)
                return

            if para == '':
                print('#####################')
                print('Table 2: Confusion matrix of Quatrain prediction.')
            _, _, _, _, _, f1_quatrain, auc_quatrain = self.evaluation_metrics(NLP_model_ytests, NLP_model_preds, RQ)
            print('---------------')

            # confusion matrix
            # print('TP _ TN _ FP _ FN _ +Recall _ -Recall')
            np.set_printoptions(suppress=True)
            # calculate average +Recall and -Recall based on all data
            recall_list = []
            for i in range(matrix_average[:,:4].shape[0]):
                tp, tn, fp, fn = matrix_average[i][0], matrix_average[i][1], matrix_average[i][2], matrix_average[i][3]
                recall_p = tp / (tp + fn)
                recall_n = tn / (tn + fp)
                recall_list.append([np.round(recall_p, decimals=3)*100, np.round(recall_n, decimals=3)*100])
            new_matrix_average = np.concatenate((matrix_average[:,:4], np.array(recall_list)), axis=1)
            new_matrix_average = new_matrix_average.T
            recall_p_quatrain, recall_n_quatrain = new_matrix_average[4][3], new_matrix_average[5][3]

            # print matrix
            print('——————————————————————————————————————————————————————————————————————')
            print('                              Threshold')
            print('——————————————————————————————————————————————————————————————————————')
            print('               0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9')
            print('——————————————————————————————————————————————————————————————————————')
            print('#TP:         {}'.format(new_matrix_average[0]))
            print('#TN:         {}'.format(new_matrix_average[1]))
            print('#FP:         {}'.format(new_matrix_average[2]))
            print('#FN:         {}'.format(new_matrix_average[3]))
            print('——————————————————————————————————————————————————————————————————————')
            print('+Recall(%):  {}'.format(new_matrix_average[4]))
            print('-Recall(%):  {}'.format(new_matrix_average[5]))
            print('——————————————————————————————————————————————————————————————————————')

        if RQ == 'RQ2.1':
            print('############################################')
            print('RQ2.1, Figure 7: Impact of length of patch description to prediction.')
            self.boxplot_distribution(meesage_length_distribution, 'Length of patch description', 'figure7_patch_description_length.jpg')
            # self.boxplot_distribution(report_length_distribution, 'Length of bug report')

        if RQ == 'RQ2.2' and Sanity:
            # sanity check result
            print('############################################')
            print('RQ2.2, Figure 8: The distribution of probability of patch correctness on original and random bug report.')
            cnt_all_predict_correct = len(all_predict_correct)
            cnt_all_random_correct =  sum(i>=0.5 for i in all_random_correct)
            print('All correct: {}, Correct by Quatrain: {}, Correct by Random:{}'.format(all_correct, cnt_all_predict_correct, cnt_all_random_correct))
            fail_number = cnt_all_predict_correct-cnt_all_random_correct
            print('The dropped +Recall: {}% ({}:{})'.format(int(round(fail_number/cnt_all_predict_correct,2)*100), fail_number, cnt_all_predict_correct))
            all_predict_correct_box = [['Original pairs', prob] for prob in all_predict_correct]
            all_random_correct_box = [['Random pairs', prob] for prob in all_random_correct]
            distribution_prob = all_predict_correct_box + all_random_correct_box

            self.boxplot_distribution_correlation(distribution_prob, 'Prediction probability by Quatrain', 'Original pairs', 'Random pairs', 'figure8_bug_report_quality.jpg')

        if RQ == 'RQ2.3':
            print('############################################')
            if para == '' and QualityOfMessage:
                print('RQ2.3, Figure 9: Impact of distance between generated patch description to ground truth on prediction performance.')
                # print('{} leave one out mean: '.format('10-90'))
                # print('Accuracy: {:.1f} -- Precision: {:.1f} -- +Recall: {:.1f} -- F1: {:.1f} -- AUC: {:.3f}'.format(np.array(accs).mean() * 100, np.array(prcs).mean() * 100, np.array(rcs).mean() * 100, np.array(f1s).mean() * 100, np.array(aucs).mean()))
                # for the comparison of generated message v.s developer message
                self.boxplot_distribution(similarity_message_distribution, 'Distance between descriptions', 'figure9_patch_description_quality.jpg')
                print('---------------')
                print('The dropped +Recall with CodeTrans-generated descriptions: {:.3f}'.format(np.array(rcs_p).mean()))

            elif para == 'CodeTrans':
                print('RQ2.3-CodeTrans,')
                print('The dropped AUC:')
                _, _, _, _, _, f1_quatrain, auc_quatrain = self.evaluation_metrics(NLP_model_ytests, NLP_model_preds)
                print('---------------')

        if RQ == 'RQ3':
            _, _, _, _, _, f1_quatrain, auc_quatrain = self.evaluation_metrics_noprint(NLP_model_ytests, NLP_model_preds)
            incorrect_number, correct_number = int(NLP_model_ytests.count(0)), int(NLP_model_ytests.count(1))
            print('---------------')

            # confusion matrix
            # print('TP _ TN _ FP _ FN _ +Recall _ -Recall')
            np.set_printoptions(suppress=True)
            # calculate average +Recall and -Recall based on all data
            recall_list = []
            for i in range(matrix_average[:, :4].shape[0]):
                tp, tn, fp, fn = matrix_average[i][0], matrix_average[i][1], matrix_average[i][2], matrix_average[i][3]
                recall_p = tp / (tp + fn)
                recall_n = tn / (tn + fp)
                recall_list.append([np.round(recall_p, decimals=3), np.round(recall_n, decimals=3)])
            new_matrix_average = np.concatenate((matrix_average[:, :4], np.array(recall_list)), axis=1)
            new_matrix_average = new_matrix_average.T
            recall_p_quatrain, recall_n_quatrain = new_matrix_average[4][3], new_matrix_average[5][3]

            # compare against ...
            NLP_model_preds = [1 if p >= 0.5 else 0 for p in NLP_model_preds]
            if comparison == 'DL':
                # print('############################################')
                # print('RQ3DL, a DL-based patch classifier (tian).')
                # print(para)
                # print('Test data size: {}'.format(len(NLP_model_ytests)))
                # print('Accuracy: {:.1f} -- Precision: {:.1f} -- +Recall: {:.1f} -- F1: {:.1f} -- AUC: {:.3f}'.format(
                #     np.array(a_accs).mean() * 100, np.array(a_prcs).mean() * 100, np.array(a_rcs).mean() * 100,
                #     np.array(a_f1s).mean() * 100, np.array(a_aucs).mean()))
                # print('AUC: {:.3f}'.format(np.array(a_aucs).mean()))
                if para == 'lr':
                    ASE_model_preds = [1 if p >= 0.1 else 0 for p in ASE_model_preds]
                elif para == 'rf':
                    ASE_model_preds = [1 if p >= 0.3 else 0 for p in ASE_model_preds]
                recall_positive, recall_negative, acc, prc, rc, f1, auc_ = self.evaluation_metrics_noprint(NLP_model_ytests, ASE_model_preds)

                # diff against NLP
                for i in range(len(NLP_model_ytests)):
                    if NLP_model_ytests[i] == NLP_model_preds[i]:
                        identify_cnt += 1
                        if NLP_model_ytests[i] != ASE_model_preds[i]:
                            new_identify_cnt += 1
                # print('new/all: {}/{}'.format(new_identify_cnt, identify_cnt))

                if para == 'lr':
                    lr_re = 'Tian et al. (LR):        {}:{} | {:.3f} {:.3f}  {:.3f}   {:.3f}'.format(incorrect_number, correct_number, auc_, f1, recall_positive, recall_negative)
                    qua_re = 'Quatrain:                {}:{} | {:.3f} {:.3f}  {:.3f}   {:.3f}'.format(incorrect_number, correct_number, auc_quatrain, f1_quatrain,  recall_p_quatrain, recall_n_quatrain)
                    return lr_re, qua_re
                elif para == 'rf':
                    rf_re = 'Tian et al. (RF):        {}:{} | {:.3f} {:.3f}  {:.3f}   {:.3f}'.format(incorrect_number, correct_number, auc_, f1, recall_positive, recall_negative,)
                    exclusively_re = '{} out of {} patches are exclusively identified by Quatrain.'.format(new_identify_cnt, identify_cnt,)
                    return rf_re, exclusively_re

            elif comparison == 'BATS':
                BatsPatchSim_model_preds = [1 if p >= 0.5 else 0 for p in BatsPatchSim_model_preds]
                # print('RQ3, {}: '.format(comparison))
                # print('Test data size: {}'.format(len(NLP_model_ytests)))
                # print('AUC: {:.3f} -- F1: {:.1f} -- +Recall: {:.1f} -- -Recall: {:.1f}'.format(
                #     np.array(bp_aucs).mean() * 100, np.array(bp_f1s).mean() * 100, np.array(bp_rcs_p).mean() * 100,
                #     np.array(bp_rcs_n).mean() * 100))
                recall_positive, recall_negative, acc, prc, rc, f1, auc_ = self.evaluation_metrics_noprint(NLP_model_ytests, BatsPatchSim_model_preds)

                print('---------------')
                for i in range(len(NLP_model_ytests)):
                    if NLP_model_ytests[i] == NLP_model_preds[i]:
                        identify_cnt += 1
                        if NLP_model_ytests[i] != BatsPatchSim_model_preds[i]:
                            new_identify_cnt += 1
                            # print('New identification: {}'.format(test_patches_info[i]))
                # print('new_identify_cnt: {}/{}'.format(new_identify_cnt, identify_cnt))
                if para == '0.0':
                    b00_re = 'BATS (cut-off: 0.0):      {}:{} | {:.3f} {:.3f}  {:.3f}   {:.3f}'.format(incorrect_number, correct_number, auc_, f1, recall_positive, recall_negative)
                    qua_re = 'Quatrain:                 {}:{} | {:.3f} {:.3f}  {:.3f}   {:.3f}'.format(incorrect_number, correct_number, auc_quatrain, f1_quatrain,  recall_p_quatrain, recall_n_quatrain)
                    return b00_re, qua_re
                elif para == '0.8':
                    b08_re = 'BATS (cut-off: 0.8):        {}:{} | {:.3f} {:.3f}  {:.3f}   {:.3f}'.format(incorrect_number, correct_number, auc_, f1, recall_positive, recall_negative)
                    qua_re = 'Quatrain:                   {}:{} | {:.3f} {:.3f}  {:.3f}   {:.3f}'.format(incorrect_number, correct_number, auc_quatrain, f1_quatrain,  recall_p_quatrain, recall_n_quatrain)
                    exclusively_re = '{} out of {} patches are exclusively identified by Quatrain'.format(new_identify_cnt, identify_cnt)
                    return b08_re, qua_re, exclusively_re
            elif comparison == 'PATCHSIM':
                BatsPatchSim_model_preds = [1 if p >= 0.5 else 0 for p in BatsPatchSim_model_preds]
                recall_positive, recall_negative, acc, prc, rc, f1, auc_ = self.evaluation_metrics_noprint(NLP_model_ytests, BatsPatchSim_model_preds)
                print('---------------')
                for i in range(len(NLP_model_ytests)):
                    if NLP_model_ytests[i] == NLP_model_preds[i]:
                        identify_cnt += 1
                        if NLP_model_ytests[i] != BatsPatchSim_model_preds[i]:
                            new_identify_cnt += 1
                            # print('New identification: {}'.format(test_patches_info[i]))
                # print('new_identify_cnt: {}/{}'.format(new_identify_cnt, identify_cnt))

                patchsim_re = 'PATCH-SIM:                 {}:{} | {:.3f} {:.3f}  {:.3f}   {:.3f}'.format(incorrect_number, correct_number, auc_, f1, recall_positive, recall_negative)
                qua_re = 'Quatrain:                  {}:{} | {:.3f} {:.3f}  {:.3f}   {:.3f}'.format(incorrect_number, correct_number, auc_quatrain, f1_quatrain, recall_p_quatrain, recall_n_quatrain)
                exclusively_re = '{} out of {} patches are exclusively identified by Quatrain'.format(new_identify_cnt, identify_cnt, )
                return patchsim_re, qua_re, exclusively_re

        if RQ == 'insights':
            _, _, _, _, _, f1_quatrain, auc_quatrain = self.evaluation_metrics_noprint(NLP_model_ytests, NLP_model_preds)
            return auc_quatrain, f1_quatrain

    def get_train_data_deprecated(self, train_ids, dataset_json, ASE=False, enhance=False):
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
                    ASE_value = self.ASE_features[train_id]
                    for p in range(len(ASE_value)):
                        ASE_vector, ASE_label = ASE_value[p][0], ASE_value[p][1]

                        ASE_train_features.append(np.array(ASE_vector))
                        ASE_train_labels.append(ASE_label)
                except Exception as e:
                    print(e)

        train_features = np.array(train_features)
        ASE_train_features = np.array(ASE_train_features)

        return train_features, train_labels, ASE_train_features, ASE_train_labels

    def get_train_data(self, train_ids, dataset_json,  comparison=False, QualityOfMessage=None):
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
                    features = np.concatenate((bugreport_vector, developer_commit_message_vector), axis=1)
                    train_features.append(features[0])
                    train_labels.append(label)
                    random.seed(1)
                    features_random = np.concatenate((random.choice(random_bug_report_vector_list), developer_commit_message_vector), axis=1)
                    train_features.append(features_random[0])
                    train_labels.append(0)
                else:
                    features = np.concatenate((bugreport_vector, commit_vector), axis=1)
                    train_features.append(features[0])
                    train_labels.append(label)

            if comparison == 'DL':
                try:
                    ASE_value = self.ASE_features[train_id]
                    for p in range(len(ASE_value)):
                        patch_id, ASE_vector, ASE_label = ASE_value[p][0], ASE_value[p][1], ASE_value[p][2]
                        ASE_train_features.append(np.array(ASE_vector))
                        ASE_train_labels.append(ASE_label)
                except Exception as e:
                    pass
                    # print(e)

        train_features = np.array(train_features)
        ASE_train_features = np.array(ASE_train_features)

        return train_features, train_labels, ASE_train_features, ASE_train_labels

    def get_test_data(self, test_ids, dataset_json, comparison=False, para='', Sanity=False, QualityOfMessage=False):
        with open('./data/CommitMessage/Developer_commit_message_bert.pickle', 'rb') as f:
            Developer_commit_message_dict = pickle.load(f)
        test_features, test_labels = [], []
        random_test_features = []
        ASE_test_features, ASE_test_labels = [], []
        test_info_for_patch = []
        self.comparison_pred = []

        for test_id in test_ids:
            value = dataset_json[test_id]
            bugreport_vector = value[0]
            if comparison == 'DL':
                try:
                    ASE_value = self.ASE_features[test_id]
                except Exception as e:
                    pass
                    # print('No this project in ASE feature!')
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
                        random.seed(1)
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
                    # count test data of compared approaches earlier than Quatrain, keep the test dataset size same.
                    if comparison == 'DL':
                        for p in range(len(ASE_value)):
                            patch_id, ASE_vector, ASE_label = ASE_value[p][0], ASE_value[p][1], ASE_value[p][2]
                            if patch_id.lower() == test_patch_id.lower():
                                ASE_test_features.append(np.array(ASE_vector))
                                ASE_test_labels.append(ASE_label)
                                break
                        else:
                            # print('patch_id: {}'.format(patch_id))
                            # continue
                            ASE_test_features.append(np.zeros(2050))
                            ASE_test_labels.append(1)
                    elif comparison == 'BATS':
                        if test_patch_id.lower() in self.BATS_RESULTS_json.keys():
                            BATS_pred = self.BATS_RESULTS_json[test_patch_id.lower()]
                            self.comparison_pred.append(BATS_pred)
                        else:
                            continue
                    elif comparison == 'PATCHSIM':
                        if test_patch_id.lower() in self.PATCHSIM_RESULTS_json.keys():
                            PATCHSIM_pred = self.PATCHSIM_RESULTS_json[test_patch_id.lower()]
                            self.comparison_pred.append(PATCHSIM_pred)
                        else:
                            continue
                    elif comparison == '':
                        pass

                    if label == 0:
                        test_features.append(features[0])
                    else:
                        if '_Developer_' in test_patch_id and (para != 'CodeTrans'):
                            # use developer commit message for RQ1 not for RQ1gen
                            features = np.concatenate((bugreport_vector, developer_commit_message_vector), axis=1)
                        test_features.append(features[0])
                    test_labels.append(label)
                    test_info_for_patch.append([test_id, test_patch_id])


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

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(111)
        # df_length['Correct prediction'].plot(ax=ax1, kind='bar', color='blue', label='Correct prediction')
        # df_length['Incorrect prediction'].plot(ax=ax1, kind='bar', color='orange', label='Incorrect prediction')
        df_length.plot(ax=ax1, kind='bar', color={"Train": "White", "Test": "Grey"}, edgecolor='black')
        plt.xlabel('Round', fontsize=32)
        ax1.set_ylabel('The number of patches', fontsize=30)
        plt.xticks(fontsize=28, rotation=0)
        plt.yticks(fontsize=28, )
        plt.ylim((0, 12000))
        plt.legend(fontsize=25, loc=2)
        plt.subplots_adjust(bottom=0.2, left=0.2)
        plt.savefig('./figure/figure6_patches_distribution.jpg')
        print('The figure is saved to ./figure/figure6_patches_distribution.jpg')
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


    def boxplot_distribution(self, distribution, y_title, figureName):
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
        bp = sns.boxplot(x='Group', y=y_title, data=dfl, showfliers=False, palette=colors, hue='Prediction', width=0.7, notch=False)
        # bp.set_xticklabels(bp.get_xticklabels(), rotation=320)
        bp.set_xticklabels(bp.get_xticklabels())
        # bp.set_xticklabels(bp.get_xticklabels(), fontsize=28)
        # bp.set_yticklabels(bp.get_yticklabels(), fontsize=28)
        plt.xlabel('Group', size=32)
        plt.ylabel(y_title, size=30)
        plt.legend(fontsize=28, loc=1)
        # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=3, fontsize=30, )
        self.adjust_box_widths(fig, 0.8)
        # plt.tight_layout()
        plt.subplots_adjust(bottom=0.2, left=0.1)
        plt.savefig('./figure/' + figureName)
        print('The figure is saved to ./figure/' + figureName)
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
        if p_value <= 0.05:
            print('Reject Null Hypothesis: Significantly different!')
        else:
            print('Support Null Hypothesis!')

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
        dataset_json = pickle.load(open('./data/bugreport_patch_json_' + embedding_method + '.pickle', 'rb'))
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

    if len(sys.argv) == 2:
        script_name = sys.argv[0]
        arg1 = sys.argv[1]
        arg2 = ''
        arg3 = ''
    elif len(sys.argv) == 3:
        script_name = sys.argv[0]
        arg1 = sys.argv[1]
        arg2 = sys.argv[2]
        arg3 = ''
    elif len(sys.argv) == 4:
        script_name = sys.argv[0]
        arg1 = sys.argv[1]
        arg2 = sys.argv[2]
        arg3 = sys.argv[3]
    else:
        arg1 = 'predict'
        arg2 = 'Missing type-checks for var_args notation'
        arg3 = 'check var_args properly'
    print('task: {}'.format(arg1))

    e = Experiment()
    dataset_json = e.dataset_json
    embedding = e.embedding

    if arg1 == 'hypothesis':
        e.validate_hypothesis(embedding)
    elif arg1 == 'RQ1':
        if arg2 == '':
            e.predict_leave1out_10group(dataset_json, embedding, times=10, algorithm='qa_attetion', comparison='', para=arg2, Sanity=False, QualityOfMessage=False, RQ=arg1)
        elif arg2 == 'balance':
            e.predict_leave1out_10group(dataset_json, embedding, times=10, algorithm='qa_attetion', comparison='', para=arg2, Sanity=False, QualityOfMessage=False, RQ=arg1)
    elif arg1 == 'RQ2.1':
        e.predict_leave1out_10group(dataset_json, embedding, times=10, algorithm='qa_attetion', comparison='', para='', Sanity=False, QualityOfMessage=False, RQ=arg1)
    elif arg1 == 'RQ2.2':
        e.predict_leave1out_10group(dataset_json, embedding, times=10, algorithm='qa_attetion', comparison='', para='', Sanity=True, QualityOfMessage=False, RQ=arg1)
    elif arg1 == 'RQ2.3':
        if arg2 == '':
            e.predict_leave1out_10group(dataset_json, embedding, times=10, algorithm='qa_attetion', comparison='', para='', Sanity=False, QualityOfMessage=True, RQ=arg1)
        elif arg2 == 'CodeTrans':
            e.predict_leave1out_10group(dataset_json, embedding, times=10, algorithm='qa_attetion', comparison='', para=arg2, Sanity=False, QualityOfMessage=False, RQ=arg1)
    elif arg1 == 'RQ3' and arg2 == 'DL':
        result_lr, result_quatrain = e.predict_leave1out_10group(dataset_json, embedding, times=10, algorithm='qa_attetion', comparison='DL', para='lr', Sanity=False, QualityOfMessage=False, RQ=arg1)
        result_rf, exclusively_re = e.predict_leave1out_10group(dataset_json, embedding, times=10, algorithm='qa_attetion', comparison='DL', para='rf', Sanity=False, QualityOfMessage=False, RQ=arg1)
        print('############################################')
        print('RQ3-DL, Table 3: Quatrain vs a DL-based patch classifier.')
        print('——————————————————————————————————————————————————————————————————————')
        print('Classifier       Incorrect:Correct |   AUC    F1 +Recall -Recall')
        print('——————————————————————————————————————————————————————————————————————')
        print(result_lr)
        print(result_rf)
        print('——————————————————————————————————————————————————————————————————————')
        print(result_quatrain)
        print('——————————————————————————————————————————————————————————————————————')
        print('New identification: {}'.format(exclusively_re))
    elif arg1 == 'RQ3' and arg2 == 'BATS':
        result_00, result_qua_00 = e.predict_leave1out_10group(dataset_json, embedding, times=10, algorithm='qa_attetion', comparison='BATS', para='0.0', Sanity=False, QualityOfMessage=False, RQ=arg1)
        result_08, result_qua_08, exclusively_re = e.predict_leave1out_10group(dataset_json, embedding, times=10, algorithm='qa_attetion', comparison='BATS', para='0.8', Sanity=False, QualityOfMessage=False, RQ=arg1)
        print('############################################')
        print('RQ3-BATS, Table 4: Quatrain vs BATS.')
        print('——————————————————————————————————————————————————————————————————————')
        print('Classifier       Incorrect:Correct |   AUC    F1 +Recall -Recall')
        print('——————————————————————————————————————————————————————————————————————')
        print(result_00)
        print(result_qua_00)
        print('——————————————————————————————————————————————————————————————————————')
        print(result_08)
        print(result_qua_08)
        print('——————————————————————————————————————————————————————————————————————')
        print('New identification: {}'.format(exclusively_re))
    elif arg1 == 'RQ3' and arg2 == 'PATCHSIM':
        result_patchsim, result_quatrain, exclusively_re = e.predict_leave1out_10group(dataset_json, embedding, times=10, algorithm='qa_attetion', comparison='PATCHSIM', para='v1', Sanity=False, QualityOfMessage=False, RQ=arg1)
        print('############################################')
        print('RQ3-PATCHSIM, Table 5: Quatrain vs (execution-based) PATCH-SIM.')
        print('——————————————————————————————————————————————————————————————————————')
        print('Classifier       Incorrect:Correct |   AUC    F1 +Recall -Recall')
        print('——————————————————————————————————————————————————————————————————————')
        print(result_patchsim)
        print(result_quatrain)
        print('——————————————————————————————————————————————————————————————————————')
        print('New identification: {}'.format(exclusively_re))
    elif arg1 == 'insights':
        auc_10fold, f1_10fold = e.predict_10fold(embedding, algorithm='rf')
        auc_10group, f1_10group = e.predict_leave1out_10group(dataset_json, embedding, times=10, algorithm='rf', comparison='', para=arg2, Sanity=False, QualityOfMessage=False, RQ=arg1)
        print('############################################')
        print('Experimental insights, 10-fold vs 10-group.')
        print('RF with 10-fold:  AUC: {:.3f} -- F1: {:.3f}'.format(auc_10fold, f1_10fold))
        print('RF with 10-group: AUC: {:.3f} -- F1: {:.3f}'.format(auc_10group, f1_10group))
    elif arg1 == 'predict':
        bug_report_txt = arg2
        patch_description = arg3
        q = API.Quatrain()
        bugreport_vector, commit_vector = q.learn_embedding('bert', bug_report_txt, patch_description)
        q.predict(bugreport_vector, commit_vector)

