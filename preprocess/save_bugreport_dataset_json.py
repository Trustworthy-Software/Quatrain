import os
from representation.word2vec import Word2vector
import pickle
from scipy.spatial import distance
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, average_precision_score
import numpy as np
import signal
import json
import pickle

path = '../data/bugreport_patch.txt'
bugreport_dataset_json = '../data/bugreport_dataset.json'

def process(path, embedding_method):
    saved_dict = {}
    w = Word2vector(embedding_method)
    cnt = 0
    with(path, 'r+') as f:
        for line in f:
            project_id = line.split('$$')[0].strip()
            bugreport_summary = line.split('$$')[1].strip()
            bugreport_description = line.split('$$')[2].strip()

            patch_id = line.split('$$')[3].strip()
            commit_content = line.split('$$')[4].strip()
            label = int(float(line.split('$$')[5].strip()))

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

            # saved by project id as key
            if not project_id in saved_dict.keys():
                saved_dict[project_id] = [bugreport_v2_vector, [commit_vector, label]]
            else:
                saved_dict[project_id].append([commit_vector, label])

            print(cnt)

    with open(bugreport_dataset_json, 'w+') as f:
        pickle.dump(saved_dict, f)

if '__name__' == '__main__':
    process(path, 'bert')

