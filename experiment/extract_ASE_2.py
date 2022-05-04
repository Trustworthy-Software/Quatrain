import shutil
from subprocess import *
import os,sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from subprocess import *
from bert_serving.client import BertClient
from gensim.models import word2vec,Doc2Vec
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import *
import re
import json
import pickle
from CC2Vec import lmg_cc2ftr_interface
import numpy as np

import config
import json

def learned_feature(patch_diff, w2v):
    try:
        bugy_all = get_diff_files_frag(patch_diff, type='buggy')
        patched_all = get_diff_files_frag(patch_diff, type='patched')
    except Exception as e:
        print('name: {}, exception: {}'.format(patch_diff, e))
        return []

    # tokenize word
    bugy_all_token = word_tokenize(bugy_all)
    patched_all_token = word_tokenize(patched_all)

    try:
        bug_vec, patched_vec = output_vec(w2v, bugy_all_token, patched_all_token)
    except Exception as e:
        print('name: {}, exception: {}'.format(patch_diff, e))
        return []

    bug_vec = bug_vec.reshape((1, -1))
    patched_vec = patched_vec.reshape((1, -1))

    # embedding feature cross
    subtract, multiple, cos, euc = multi_diff_features(bug_vec, patched_vec)
    embedding = np.hstack((subtract, multiple, cos, euc,))

    # embedding = subtraction(bug_vec, patched_vec)

    return list(embedding.flatten()), bugy_all+patched_all

def subtraction(buggy, patched):
    return patched - buggy

def multiplication(buggy, patched):
    return buggy * patched

def cosine_similarity(buggy, patched):
    return paired_cosine_distances(buggy, patched)

def euclidean_similarity(buggy, patched):
    return paired_euclidean_distances(buggy, patched)

def multi_diff_features(buggy, patched):
    subtract = subtraction(buggy, patched)
    multiple = multiplication(buggy, patched)
    cos = cosine_similarity(buggy, patched).reshape((1, 1))
    euc = euclidean_similarity(buggy, patched).reshape((1, 1))

    return subtract, multiple, cos, euc

def output_vec(w2v, bugy_all_token, patched_all_token):

    if w2v == 'bert':
        m = BertClient(check_length=False, check_version=False, port=8190)
        bug_vec = m.encode([bugy_all_token], is_tokenized=True)
        patched_vec = m.encode([patched_all_token], is_tokenized=True)
    elif w2v == 'doc':
        # m = Doc2Vec.load('../model/doc_file_64d.model')
        m = Doc2Vec.load('../model/Doc_frag_ASE.model')
        bug_vec = m.infer_vector(bugy_all_token, alpha=0.025, steps=300)
        patched_vec = m.infer_vector(patched_all_token, alpha=0.025, steps=300)
    else:
        print('wrong model')
        raise

    return bug_vec, patched_vec

def get_diff_files_frag(patch_diff, type):
    # with open(path_patch, 'r') as file:
        lines = ''
        p = r"([^\w_])"
        flag = True
        # try:
        for line in patch_diff:
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
                    if line.startswith('---') or line.startswith('PATCH_DIFF_ORIG=---'):
                        continue
                        # line = re.split(pattern=p, string=line.split(' ')[1].strip())
                        # lines += ' '.join(line) + ' '
                    elif line.startswith('-'):
                        if line[1:].strip().startswith('//'):
                            continue
                        line = re.split(pattern=p, string=line[1:].strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('')
                        line = ' '.join(line)
                        lines += line.strip() + ' '
                    elif line.startswith('+'):
                        # do nothing
                        pass
                    else:
                        line = re.split(pattern=p, string=line.strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('')
                        line = ' '.join(line)
                        lines += line.strip() + ' '
                elif type == 'patched':
                    if line.startswith('+++'):
                        continue
                        # line = re.split(pattern=p, string=line.split(' ')[1].strip())
                        # lines += ' '.join(line) + ' '
                    elif line.startswith('+'):
                        if line[1:].strip().startswith('//'):
                            continue
                        line = re.split(pattern=p, string=line[1:].strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('')
                        line = ' '.join(line)
                        lines += line.strip() + ' '
                    elif line.startswith('-'):
                        # do nothing
                        pass
                    else:
                        line = re.split(pattern=p, string=line.strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('')
                        line = ' '.join(line)
                        lines += line.strip() + ' '
        # except Exception:
        #     print(Exception)
        #     return 'Error'
        return lines

def obtain_ASE_features(path_dataset, w2v):
    total = 0
    generated = 0
    dictionary = pickle.load(open('../CC2Vec/dict.pkl', 'rb'))
    for root, dirs, files in os.walk(path_dataset):
        for file in files:
            if file.endswith('.patch'):
                total += 1
                name = file.split('.')[0]

                feature_name = 'ASE_features_'+w2v + '_' + name.split('_')[0] + '.pickle'
                path_feature = os.path.join(root, feature_name)
                if os.path.exists(path_feature):
                    generated += 1
                    continue

                path_patch = os.path.join(root, file)
                # learned feature
                try:
                    if w2v == 'CC2Vec':
                        learned_vector = lmg_cc2ftr_interface.learned_feature(path_patch, load_model='../CC2Vec/cc2ftr.pt', dictionary=dictionary)
                        if learned_vector != []:
                            learned_vector = list(learned_vector[0])
                    elif w2v == 'bert' or w2v == 'doc':
                        learned_vector, _ = learned_feature(path_patch, w2v)
                except Exception as e:
                    print(e)
                    continue
                if learned_vector == []:
                    continue

                dict = {'ASE':learned_vector}
                with open(path_feature, 'wb') as f:
                    pickle.dump(dict, f)

                generated += 1
                print('generated: {}/{}. new: {}'.format(generated, total, name))
    print('total: {}, generated: {}'.format(total, generated))

def obtain_ASE_features2(path_patch, w2v):
    cnt, succ = 0, 0
    dict_ASE_feature = {}
    datasets = os.listdir(path_patch)
    for dataset in datasets:
        path_dataset = os.path.join(path_patch, dataset)
        benchmarks = os.listdir(path_dataset)
        for benchmark in benchmarks:
            path_benchmark = os.path.join(path_dataset, benchmark)
            tools = os.listdir(path_benchmark)
            for tool in tools:
                path_tool = os.path.join(path_benchmark, tool)
                labels = os.listdir(path_tool)
                for label in labels:
                    path_label = os.path.join(path_tool, label)
                    projects = os.listdir(path_label)
                    for project in projects:
                        path_project = os.path.join(path_label, project)
                        ids = os.listdir(path_project)
                        for id in ids:
                            path_id = os.path.join(path_project, id)
                            patches = os.listdir(path_id)
                            for patch in patches:
                                # parse patch
                                cnt += 1
                                print(cnt)
                                if label == 'Correct':
                                    label_int = 1
                                elif label == 'Incorrect':
                                    label_int = 0
                                else:
                                    raise
                                patch_diff = []
                                path_single_patch = os.path.join(path_id, patch)
                                for root, dirs, files in os.walk(path_single_patch):
                                    for file in files:
                                        if file.endswith('.patch'):
                                            try:
                                                with open(os.path.join(root, file), 'r+') as f:
                                                    patch_diff += f.readlines()
                                            except Exception as e:
                                                print(e)
                                                continue

                                try:
                                    if w2v == 'CC2Vec':
                                        learned_vector = lmg_cc2ftr_interface.learned_feature(path_patch, load_model='../CC2Vec/cc2ftr.pt', dictionary=dictionary)
                                        if learned_vector != []:
                                            learned_vector = list(learned_vector[0])
                                    elif w2v == 'bert' or w2v == 'doc':
                                        learned_vector, _ = learned_feature(patch_diff, w2v)
                                except Exception as e:
                                    print(e)
                                    continue
                                if learned_vector == []:
                                    continue

                                ASE_feature_combined = learned_vector
                                if len(ASE_feature_combined) != 2050:
                                    print('???: {}'.format(len(ASE_feature_combined)))
                                if benchmark == 'Bugsjar' and '+' in project:
                                    project1 = project.split('+')[1]
                                    project1 = project1.lower()
                                    project_id = project1 + '-' + id
                                else:
                                    project_id = project + '-' + id

                                key = project_id
                                key = key.lower()
                                # value = [ASE_feature_combined, label_int]
                                # dict_ASE_feature[key] = value

                                # saved by project id as key
                                patch_id = patch + '-' + project_id + '_' + tool + '_' + dataset

                                if not key in dict_ASE_feature.keys():
                                    dict_ASE_feature[key] = [[patch_id, ASE_feature_combined, label_int]]
                                else:
                                    dict_ASE_feature[key].append([patch_id, ASE_feature_combined, label_int])
                                succ += 1

    print('cnt: {}, succ: {}'.format(cnt, succ))
    # save ASE feature
    with open('../data/ASE_features2_'+w2v+'.pickle', 'wb') as f:
        pickle.dump(dict_ASE_feature, f)


def ASE_features(path_json):
    # ASE_vector = []
    try:
        with open(path_json, 'rb') as f:
            feature_json = pickle.load(f)
            ASE_vector = feature_json['ASE']

    except Exception as e:
        print('name: {}, exception: {}'.format(path_json, e))
        return []

    return ASE_vector

if __name__ == '__main__':
    w2v = 'bert'
    # w2v = 'CC2Vec'
    path_patch = cf = config.Config().path_patch
    # 1. extract ASE feature of all files changed by patches at one time.
    # obtain_ASE_features(path_patch, w2v=w2v)
    # path_patch = '../' + path_patch
    obtain_ASE_features2(path_patch, w2v)


