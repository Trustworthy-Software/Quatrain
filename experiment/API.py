import os
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
from representation.word2vec import Word2vector

dirname = os.path.dirname(__file__)

class Quatrain:
    def __init__(self, ):
        self.embedding_method = 'bert'
        self.threshold = 0.5
        if os.path.exists('quatrain_model.h5'):
            self.predict_model = load_model('quatrain_model.h5')
        else:
            self.train_model()
            self.predict_model = load_model('quatrain_model.h5')
        print('QUATRAIN model loaded!')

    def train_model(self, ):
        print('training QUATRAIN model......')
        dataset = pickle.load(open(os.path.join(dirname, '../data/bugreport_patch_array_' + self.embedding_method + '.pickle'), 'rb'))
        bugreport_vector = np.array(dataset[0]).reshape((len(dataset[0]), -1))
        commit_vector = np.array(dataset[1]).reshape((len(dataset[1]), -1))
        labels = np.array(dataset[2])

        # combine bug report and commit message of patch
        train_features = np.concatenate((bugreport_vector, commit_vector), axis=1)
        # standard data
        scaler = StandardScaler().fit(train_features)
        x_train = scaler.transform(train_features)
        y_train = labels

        # qa_attention QUATRAIN
        seq_maxlen = 64
        y_train = np.array(y_train).astype(float)
        x_train_q = x_train[:, :1024]
        x_train_a = x_train[:, 1024:]
        x_train_q = np.reshape(x_train_q, (x_train_q.shape[0], seq_maxlen, -1))
        x_train_a = np.reshape(x_train_a, (x_train_a.shape[0], seq_maxlen, -1))
        quatrain_model = get_qa_attention(x_train_q.shape[1:], x_train_a.shape[1:])
        callback = [keras.callbacks.EarlyStopping(monitor='val_auc', patience=1, mode="max", verbose=0), ]
        quatrain_model.fit([x_train_q, x_train_a], y_train, callbacks=callback, validation_split=0.2, batch_size=128, epochs=10, )
        quatrain_model.save('quatrain_model.h5')

    def learn_embedding(self, embedding_method, bug_report_txt, patch_description_txt):
        w = Word2vector(embedding_method)
        try:
            bugreport_vector = w.embedding(bug_report_txt)
            commit_vector = w.embedding(patch_description_txt)
        except Exception as e:
            print(e)
            raise e
        return bugreport_vector, commit_vector

    def predict(self, b_vector, p_vector):
        seq_maxlen = 64
        x_test_q = b_vector
        x_test_a = p_vector
        x_test_q = np.reshape(x_test_q, (x_test_q.shape[0], seq_maxlen, -1))
        x_test_a = np.reshape(x_test_a, (x_test_a.shape[0], seq_maxlen, -1))
        y_pred_prob = self.predict_model.predict([x_test_q, x_test_a])[:, 0]
        y_pred = [1 if p >= self.threshold else 0 for p in list(y_pred_prob)]

        if y_pred[0] == 1:
            print('Congrats! Your patch is CORRECT.')
        elif y_pred[0] == 0:
            print('Sorry, your patch is INCORRECT.')

