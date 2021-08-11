import config
import os
from representation.word2vec import Word2vector
import pickle

class Experiment:
    def __init__(self):
        self.cf = config.Config()
        self.path_patch = self.cf.path_patch
        # self.dict_b = {}
        self.bugreport_closure = None

    def save_bugreport(self, project, embedding_method):
        w = Word2vector(embedding_method)
        dict_b = {}
        path = '../preprocess/' + project + '_bugreport.txt'
        with open(path, 'r+') as f:
            for line in f:
                project_id = line.split(',')[0]
                bugReport = line.split(',')[1]
                learned_vector = w.embedding(bugReport)
                dict_b[project_id] = learned_vector
        pickle.dump(dict_b, open('../data/bugreport_dict.pickle', 'wb'))

    def predict_closure(self, path_patch):
        with open('../data/bugreport_dict.pickle', 'rb') as f:
            self.bugreport_closure = pickle.load(f)
        for root, dirs, files in os.walk(path_patch):
            for file in files:
                if file.endswith('.patch'):
                    pass

if __name__ == '__main__':
    e = Experiment()
    e.save_bugreport(project='Closure', embedding_method='bert')
    e.predict_closure(e.cf.path_patch)