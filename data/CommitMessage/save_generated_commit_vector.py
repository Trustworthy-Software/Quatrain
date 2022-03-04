import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
from representation.word2vec import Word2vector
import pickle
from scipy.spatial import distance
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix, average_precision_score
import numpy as np
import signal
import json
import pickle
dirname = os.path.dirname(__file__)

path = os.path.join(dirname, './Generated_commit_message_All.json')
print(path)

def process(path, embedding_method):
    saved_dict = {}
    w = Word2vector(embedding_method)
    cnt = 0
    with open(path, 'r+') as f:
        generated_commit_message_dict = json.load(f)

    for k,v in generated_commit_message_dict.items():
        project_id = k
        generated_commit = v
        signal.alarm(300)
        try:
            generated_commit_vector = w.embedding(generated_commit)
        except Exception as e:
            print(e)
            continue
        signal.alarm(0)

        saved_dict[project_id] = generated_commit_vector

        cnt += 1
        print(cnt)

    with open(os.path.join(dirname, './Generated_commit_message_All_' + embedding_method + '.pickle'), 'wb') as f:
        pickle.dump(saved_dict, f)

if __name__ == '__main__':
    process(path, 'bert')

