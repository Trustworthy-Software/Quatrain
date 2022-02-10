import sys, os
os.path.abspath(os.path.join('..', './representation'))
import nltk
# nltk.download('wordnet')
import pickle
from representation.CC2Vec import lmg_cc2ftr_interface
import logging
from bert_serving.client import BertClient
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import re
from sklearn.metrics.pairwise import *
import string

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
            # nltk.download('punkt')

            logging.getLogger().info('Waiting for Bert server')
            self.m = BertClient(check_length=False, check_version=False)

    def embedding(self, text):
        # if self.w2v == 'cc2vec':
        #     learned_vector = lmg_cc2ftr_interface.learned_feature(path_patch_id, load_model=MODEL_CC2Vec + 'cc2ftr.pt',
        #                                                           dictionary=self.dictionary)
        #     learned_vector = list(learned_vector.flatten())
        if self.w2v == 'bert':
            # initial
            # lemmatizer = WordNetLemmatizer()

            # learned_vector = self.learned_feature(self.w2v)
            # 1.
            token = word_tokenize(text)

            # # 2. split function name
            # translator = re.compile('[%s]' % re.escape(string.punctuation))
            # text_without_punctuation = translator.sub(' ', text)
            # text_without_tense = lemmatizer.lemmatize(text_without_punctuation)
            # # text_split = re.split(pattern='(?=[A-Z.\s])', string=text_without_tense)
            # # token = [i for i in text_split if (i != '' and i != ' ')]
            # token = word_tokenize(text_without_tense)

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

    def get_only_change(self, path_patch, type='patched'):
        with open(path_patch, 'r+') as file:
            lines = ''
            p = r"([^\w_])"
            # try:
            for line in file:
                line = line.strip()
                if line != '':
                    if line.startswith('@@') or line.startswith('diff') or line.startswith('index'):
                        continue
                    elif type == 'buggy':
                        if line.startswith('--- ') or line.startswith('-- ') or line.startswith('PATCH_DIFF_ORIG=---'):
                            continue
                        elif line.startswith('-'):
                            if line[1:].strip() == '':
                                continue
                            if line[1:].strip().startswith('//'):
                                continue
                            line = re.split(pattern=p, string=line[1:].strip())

                            final = []
                            for s in line:
                                s = s.strip()
                                if s == '' or s == ' ':
                                    continue
                                # CamelCase to underscore_case
                                cases = re.split(pattern='(?=[A-Z0-9])', string=s)
                                for c in cases:
                                    if c == '' or c == ' ':
                                        continue
                                    final.append(c)

                            line = ' '.join(final)
                            lines += line.strip() + ' '
                        else:
                            # do nothing
                            pass
                    elif type == 'patched':
                        if line.startswith('+++ ') or line.startswith('++ '):
                            continue
                            # line = re.split(pattern=p, string=line.split(' ')[1].strip())
                            # lines += ' '.join(line) + ' '
                        elif line.startswith('+'):
                            if line[1:].strip() == '':
                                continue
                            if line[1:].strip().startswith('//'):
                                continue
                            line = re.split(pattern=p, string=line[1:].strip())
                            final = []
                            for s in line:
                                s = s.strip()
                                if s == '' or s == ' ':
                                    continue
                                # CamelCase to underscore_case
                                cases = re.split(pattern='(?=[A-Z0-9])', string=s)
                                for c in cases:
                                    if c == '' or c == ' ':
                                        continue
                                    final.append(c)
                            line = ' '.join(final)
                            lines += line.strip() + ' '
                        else:
                            # do nothing
                            pass
        return lines
    def learned_feature(self, path_patch, w2v):
        try:
            # bugy_all = self.get_diff_files_frag(path_patch, type='buggy')
            # patched_all = self.get_diff_files_frag(path_patch, type='patched')
            bugy_all = self.get_only_change(path_patch, type='buggy')
            patched_all = self.get_only_change(path_patch, type='patched')

            # tokenize word
            bugy_all_token = word_tokenize(bugy_all)
            patched_all_token = word_tokenize(patched_all)

            bug_vec, patched_vec = self.output_vec_bats(w2v, bugy_all_token, patched_all_token)
        except Exception as e:
            # print('patch: {}, exception: {}'.format(path_patch, e))
            raise e

        bug_vec = bug_vec.reshape((1, -1))
        patched_vec = patched_vec.reshape((1, -1))

        # embedding feature cross
        # subtract, multiple, cos, euc = self.multi_diff_features(bug_vec, patched_vec)
        # embedding = np.hstack((subtract, multiple, cos, euc,))

        embedding = self.subtraction(bug_vec, patched_vec)

        return list(embedding.flatten())

    def subtraction(self, buggy, patched):
        return buggy - patched

    def multiplication(self, buggy, patched):
        return buggy * patched

    def cosine_similarity(self, buggy, patched):
        return paired_cosine_distances(buggy, patched)

    def euclidean_similarity(self, buggy, patched):
        return paired_euclidean_distances(buggy, patched)

    def multi_diff_features(self, buggy, patched):
        subtract = self.subtraction(buggy, patched)
        multiple = self.multiplication(buggy, patched)
        cos = self.cosine_similarity(buggy, patched).reshape((1, 1))
        euc = self.euclidean_similarity(buggy, patched).reshape((1, 1))

        return subtract, multiple, cos, euc

    def convert_single_patch(self, path_patch):
        try:
            if self.w2v == 'cc2vec':
                multi_vector = [] # sum up vectors of different parts of patch
                patch = os.listdir(path_patch)
                for part in patch:
                    p = os.path.join(path_patch, part)
                    learned_vector = lmg_cc2ftr_interface.learned_feature(p, load_model=MODEL_CC2Vec + 'cc2ftr.pt', dictionary=self.dictionary)
                    multi_vector.append(list(learned_vector.flatten()))
                combined_vector = np.array(multi_vector).sum(axis=0)

            elif self.w2v == 'bert':
                multi_vector = []
                multi_vector_cross = []
                patch = os.listdir(path_patch)
                for part in patch:
                    p = os.path.join(path_patch, part)
                    learned_vector = self.learned_feature(p, self.w2v)
                    learned_vector_cross = self.learned_feature_cross(p, self.w2v)

                    multi_vector.append(learned_vector)
                    multi_vector_cross.append(learned_vector_cross)
                combined_vector = np.array(multi_vector).sum(axis=0)
                combined_vector_cross = np.array(multi_vector_cross).sum(axis=0)
                return combined_vector, combined_vector_cross
            elif self.w2v == 'string':
                multi_vector = []
                patch = os.listdir(path_patch)
                for part in patch:
                    p = os.path.join(path_patch, part)
                    learned_vector = self.extract_text(p, )
                    multi_vector.append(learned_vector)
                combined_vector = ''
                for s in multi_vector:
                    combined_vector += s
                combined_vector = [combined_vector]
            # combined_vector = np.array(multi_vector).mean(axis=0)
            return combined_vector, None
        except Exception as e:
            raise e

    def output_vec_bats(self, w2v, bugy_all_token, patched_all_token):

        if w2v == 'bert':
            if bugy_all_token == []:
                bug_vec = np.zeros((1, 1024))
            else:
                bug_vec = self.m.encode([bugy_all_token], is_tokenized=True)

            if patched_all_token == []:
                patched_vec = np.zeros((1, 1024))
            else:
                patched_vec = self.m.encode([patched_all_token], is_tokenized=True)
        elif w2v == 'doc':
            # m = Doc2Vec.load('../model/doc_file_64d.model')
            m = Doc2Vec.load('../model/Doc_frag_ASE.model')
            bug_vec = m.infer_vector(bugy_all_token, alpha=0.025, steps=300)
            patched_vec = m.infer_vector(patched_all_token, alpha=0.025, steps=300)
        else:
            print('wrong model')
            raise

        return bug_vec, patched_vec

    def learned_feature_cross(self, path_patch, w2v):
        try:
            bugy_all = self.get_diff_files_frag(path_patch, type='buggy')
            patched_all = self.get_diff_files_frag(path_patch, type='patched')

            # tokenize word
            bugy_all_token = word_tokenize(bugy_all)
            patched_all_token = word_tokenize(patched_all)

            bug_vec, patched_vec = self.output_vec_bats(w2v, bugy_all_token, patched_all_token)
        except Exception as e:
            # print('patch: {}, exception: {}'.format(path_patch, e))
            raise e

        bug_vec = bug_vec.reshape((1, -1))
        patched_vec = patched_vec.reshape((1, -1))

        # embedding feature cross
        subtract, multiple, cos, euc = self.multi_diff_features(bug_vec, patched_vec)
        embedding = np.hstack((subtract, multiple, cos, euc,))

        return list(embedding.flatten())

    def get_diff_files_frag(self, path_patch, type):
        with open(path_patch, 'r') as file:
            lines = ''
            p = r"([^\w_])"
            flag = True
            # try:
            for line in file:
                line = line.strip()
                if '*/' in line:
                    flag = True
                    continue
                if flag == False:
                    continue
                if line != '':
                    if line.startswith('@@') or line.startswith('diff') or line.startswith('index'):
                        continue
                    if line.startswith('Index') or line.startswith('==='):
                        continue
                    elif '/*' in line:
                        flag = False
                        continue
                    elif type == 'buggy':
                        if line.startswith('--- ') or line.startswith('-- ') or line.startswith('PATCH_DIFF_ORIG=---'):
                            continue
                            # line = re.split(pattern=p, string=line.split(' ')[1].strip())
                            # lines += ' '.join(line) + ' '
                        elif line.startswith('-'):
                            if line[1:].strip() == '':
                                continue
                            if line[1:].strip().startswith('//'):
                                continue
                            line = re.split(pattern=p, string=line[1:].strip())
                            final = []
                            for s in line:
                                s = s.strip()
                                if s == '' or s == ' ':
                                    continue
                                # CamelCase to underscore_case
                                cases = re.split(pattern='(?=[A-Z0-9])', string=s)
                                for c in cases:
                                    if c == '' or c == ' ':
                                        continue
                                    final.append(c)
                            line = ' '.join(final)
                            lines += line.strip() + ' '
                        elif line.startswith('+'):
                            # do nothing
                            pass
                        else:
                            line = re.split(pattern=p, string=line.strip())
                            final = []
                            for s in line:
                                s = s.strip()
                                if s == '' or s == ' ':
                                    continue
                                # CamelCase to underscore_case
                                cases = re.split(pattern='(?=[A-Z0-9])', string=s)
                                for c in cases:
                                    if c == '' or c == ' ':
                                        continue
                                    final.append(c)
                            line = ' '.join(final)
                            lines += line.strip() + ' '
                    elif type == 'patched':
                        if line.startswith('+++ ') or line.startswith('++ '):
                            continue
                            # line = re.split(pattern=p, string=line.split(' ')[1].strip())
                            # lines += ' '.join(line) + ' '
                        elif line.startswith('+'):
                            if line[1:].strip() == '':
                                continue
                            if line[1:].strip().startswith('//'):
                                continue
                            line = re.split(pattern=p, string=line[1:].strip())
                            final = []
                            for s in line:
                                s = s.strip()
                                if s == '' or s == ' ':
                                    continue
                                # CamelCase to underscore_case
                                cases = re.split(pattern='(?=[A-Z0-9])', string=s)
                                for c in cases:
                                    if c == '' or c == ' ':
                                        continue
                                    final.append(c)
                            line = ' '.join(final)
                            lines += line.strip() + ' '
                        elif line.startswith('-'):
                            # do nothing
                            pass
                        else:
                            line = re.split(pattern=p, string=line.strip())
                            final = []
                            for s in line:
                                s = s.strip()
                                if s == '' or s == ' ':
                                    continue
                                # CamelCase to underscore_case
                                cases = re.split(pattern='(?=[A-Z0-9])', string=s)
                                for c in cases:
                                    if c == '' or c == ' ':
                                        continue
                                    final.append(c)
                            line = ' '.join(final)
                            lines += line.strip() + ' '
            # except Exception:
            #     print(Exception)
            #     return 'Error'
            return lines