import pickle
import os
dirname = os.path.dirname(__file__)

class Config:
    def __init__(self):
        # embedding method
        self.embedding = 'bert'
        # processed dataset
        self.dataset_json = pickle.load(open(os.path.join(dirname, '../data/bugreport_patch_json_' + self.embedding + '.pickle'), 'rb'))

        # original dataset with patches text and commit messages text.
        self.path_patch = '/Users/haoye.tian/Documents/ASE2022withTextUnique'
        # the feature from Tian et al.'s ASE2020 paper for RQ3 DL experiment.
        self.path_ASE2020_feature = '/Users/haoye.tian/Documents/University/data/ASE_features2_bert.pickle'