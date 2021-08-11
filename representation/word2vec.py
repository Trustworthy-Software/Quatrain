import sys, os
os.path.abspath(os.path.join('..', './representation'))
import pickle
from representation.CC2Vec import lmg_cc2ftr_interface
import logging
from bert_serving.client import BertClient
from nltk.tokenize import word_tokenize
import numpy as np

MODEL_CC2Vec = '../representation/CC2Vec/'

class Word2vector:
    def __init__(self, method=None,):
        # self.w2v = word2vec
        self.w2v = method
        # self.path_patch_root = path_patch_root


        # init for patch vector
        if self.w2v == 'cc2vec':
            self.dictionary = pickle.load(open(MODEL_CC2Vec+'dict.pkl', 'rb'))
        elif self.w2v == 'bert':
            import nltk
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            nltk.download('punkt')

            logging.getLogger().info('Waiting for Bert server')
            self.m = BertClient(check_length=False, check_version=False)

    def embedding(self, text):
        # if self.w2v == 'cc2vec':
        #     learned_vector = lmg_cc2ftr_interface.learned_feature(path_patch_id, load_model=MODEL_CC2Vec + 'cc2ftr.pt',
        #                                                           dictionary=self.dictionary)
        #     learned_vector = list(learned_vector.flatten())
        if self.w2v == 'bert':
            # learned_vector = self.learned_feature(self.w2v)
            token = word_tokenize(text)
            vec = self.output_vec(self.w2v, token)
            learned_vector = vec.reshape((1, -1))

        elif self.w2v == 'string':
            learned_vector = text

        return learned_vector

    def output_vec(self, w2v, token):
        if w2v == 'bert':
            if token == []:
                vec = np.zeros((1, 1024))
            else:
                vec = self.m.encode([token], is_tokenized=True)

        elif w2v == 'doc':
            # m = Doc2Vec.load('../model/doc_file_64d.model')
            m = Doc2Vec.load('../models/Doc_frag_ASE.model')
            bug_vec = m.infer_vector(token, alpha=0.025, steps=300)
            patched_vec = m.infer_vector(token, alpha=0.025, steps=300)
        else:
            print('wrong model')
            raise

        return vec