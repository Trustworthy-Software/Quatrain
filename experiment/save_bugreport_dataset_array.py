import config
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from representation.word2vec import Word2vector
import pickle
from scipy.spatial import distance
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, average_precision_score
import numpy as np
import ML4Prediciton
import signal
import json
dirname = os.path.dirname(__file__)

path = os.path.join(dirname, '../data/bugreport_patch.txt')

def handler(signum, frame):
    raise Exception("end of time")

def save_bugreport_patch_vector(embedding_method):
    
    labels, y_preds = [], []
    file_name = '../data/bugreport_patch_array_' + embedding_method + '.pickle'
    if os.path.exists(file_name):
        return
    w = Word2vector(embedding_method)
    bugreport_vectors, commit_vectors = [], []
    bugreport_v2_vectors = []
    cnt = 0
    signal.signal(signal.SIGALRM, handler)
    with open(path, 'r+') as f:
        for line in f:
            project_id = line.split('$$')[0].strip()
            bugreport_summary = line.split('$$')[1].strip()
            bugreport_description = line.split('$$')[2].strip()

            patch_id = line.split('$$')[3].strip()
            commit_content = line.split('$$')[4].strip()
            label = int(float(line.split('$$')[5].strip()))
            # skip none and duplicated cases
            # if bugreport_summary == 'None' or commit_content == 'None' or (patch_id not in deduplicated_patchids):
            if bugreport_summary == 'None' or commit_content == 'None':
                continue

            signal.alarm(300)
            try:
                # bugreport_vector = w.embedding(bugreport_summary)
                bugreport_v2_vector = w.embedding(bugreport_summary + '.' + bugreport_description)
                commit_vector = w.embedding(commit_content)
            except Exception as e:
                print(e)
                continue
            signal.alarm(0)

            # bugreport_vectors.append(bugreport_vector)
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

    # pickle.dump([bugreport_vectors, commit_vectors, labels],
    #             open('../data/bugreport_patch_' + embedding_method + '.pickle', 'wb'))
    pickle.dump([bugreport_v2_vectors, commit_vectors, labels],
                open(os.path.join(dirname, file_name), 'wb'))
    # self.evaluation_metrics(labels, y_preds)

if __name__ == '__main__':
    path_patch = config.Config().path_patch
    embedding = 'bert'

    # save all bug report and patch as vector
    save_bugreport_patch_vector(embedding_method=embedding)