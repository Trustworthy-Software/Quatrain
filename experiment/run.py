import config
import os
from representation.word2vec import Word2vector
import pickle
from scipy.spatial import distance
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, average_precision_score
import numpy as np
import ML4Prediciton
import signal

class Experiment:
    def __init__(self):
        self.cf = config.Config()
        self.path_patch = self.cf.path_patch
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
        path = '../data/BugReport'
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

    def save_bugreport_commit(self, path_patch, ):
        # dataset_text = ''
        dataset_text_with_description = ''
        file_name = '../data/bugreport_patch.txt'
        if os.path.exists(file_name):
            return
        with open('../data/bugreport_dict.pickle', 'rb') as f:
            self.bugReportText = pickle.load(f)
        for root, dirs, files in os.walk(path_patch):
            for file in files:
                if file.endswith('.patch'):
                    name = file.split('.')[0]
                    label = 1 if root.split('/')[-4] == 'Correct' else 0
                    project = name.split('-')[1]
                    # if project != 'Closure':
                    #     continue
                    id = name.split('-')[2]
                    project_id = project + '-' + id
                    print('collecting {}'.format(project_id))
                    commit_file = name + '.txt'
                    try:
                        with open(os.path.join(root, commit_file), 'r+') as f:
                            commit_content = f.readlines()[0].strip('\n')
                    except Exception as e:
                        print(e)
                        commit_content = ''
                    if project_id in self.bugReportText.keys():
                        bug_report_text = self.bugReportText[project_id]
                        bug_report_summary = bug_report_text[0].strip()
                        bug_report_description = bug_report_text[1].strip()
                    else:
                        bug_report_summary = 'None'
                        bug_report_description = 'None'
                    # TODO: use description info of bug report
                    # dataset_text += '$$'.join([project_id, bug_report_summary, name, commit_content, str(label)]) + '\n'
                    dataset_text_with_description += '$$'.join([project_id, bug_report_summary, bug_report_description, name, commit_content, str(label)]) + '\n'
        with open(file_name, 'w+') as f:
            f.write(dataset_text_with_description)


    def save_bugreport_patch_vector(self, embedding_method):
        labels, y_preds = [], []
        file_name = '../data/bugreport_patch_'+embedding_method+'.pickle'
        if os.path.exists(file_name):
            return
        bugreport_vectors, commit_vectors = [], []
        bugreport_v2_vectors = []
        cnt = 0
        signal.signal(signal.SIGALRM, self.handler)
        with open('../data/bugreport_patch.txt', 'r+') as f:
            for line in f:
                    project_id = line.split('$$')[0].strip()
                    bugreport_summary = line.split('$$')[1].strip()
                    bugreport_description = line.split('$$')[2].strip()
                    patch_id = line.split('$$')[3].strip()
                    commit_content = line.split('$$')[4].strip()
                    label = int(float(line.split('$$')[5].strip()))
                    if bugreport_summary == 'None' or commit_content == 'None':
                        continue
                    w = Word2vector(embedding_method)

                    signal.alarm(300)
                    try:
                        bugreport_vector = w.embedding(bugreport_summary)
                        bugreport_v2_vector = w.embedding(bugreport_summary+'.'+bugreport_description)
                        commit_vector = w.embedding(commit_content)
                    except Exception as e:
                        print(e)
                        continue
                    signal.alarm(0)

                    bugreport_vectors.append(bugreport_vector)
                    bugreport_v2_vectors.append(bugreport_v2_vector)
                    commit_vectors.append(commit_vector)
                    labels.append(label)

                    cnt += 1
                    print('{} {}'.format(cnt, project_id))
                    # dist = distance.euclidean(bug_report, commit_vector)/(1+distance.euclidean(bug_report, commit_vector))
                    # similarity = 1- dist
                    # if similarity >= 0.5:
                    #     y_pred = 1
                    # else:
                    #     y_pred = 0
                    #
                    # labels.append(label)
                    # y_preds.append(y_pred)


        pickle.dump([bugreport_vectors, commit_vectors, labels], open('../data/bugreport_patch_'+embedding_method+'.pickle', 'wb'))
        pickle.dump([bugreport_v2_vectors, commit_vectors, labels], open('../data/bugreport_patch_'+embedding_method+'(description).pickle', 'wb'))
        # self.evaluation_metrics(labels, y_preds)

    def predict(self, embedding_method):
        dataset = pickle.load(open('../data/bugreport_patch_'+embedding_method+'.pickle', 'rb'))
        bugreport_vector = np.array(dataset[0]).reshape((len(dataset[0]),-1))
        commit_vector = np.array(dataset[1]).reshape((len(dataset[1]),-1))
        labels = np.array(dataset[2])

        # combine bug report and commit message of patch
        features = np.concatenate((bugreport_vector, commit_vector), axis=1)
        cl = ML4Prediciton.Classifier(features, labels, 'rf', 10)
        cl.cross_validation()

    def handler(signum, frame):
        raise Exception("end of time")

if __name__ == '__main__':
    embedding = 'bert'
    e = Experiment()
    e.save_bugreport()
    e.save_bugreport_commit(e.cf.path_patch,)
    e.save_bugreport_patch_vector(embedding_method=embedding)
    e.predict(embedding)