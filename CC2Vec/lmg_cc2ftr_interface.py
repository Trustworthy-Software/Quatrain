from CC2Vec.lmg_cli import read_args_lmg, get_patch_cc2v, extracted_cc2ftr
from CC2Vec.lmg_padding import processing_data
import pickle

def learned_feature(patch_file, load_model='cc2ftr.pt', 
                        dictionary=None):
    
    params = read_args_lmg().parse_args()
    params.load_model = load_model

    data = [list(), list()]
    data[1].append(get_patch_cc2v(patch_file))
    data[0].append('<NULL>')
    msg, diff = data[0], data[1]
    # dictionary = pickle.load(open(dictionary_data, 'rb'))
    pad_added_code, pad_removed_code = processing_data(code=diff, dictionary=dictionary, params=params)
    dict_msg, dict_code = dictionary

    data = (msg, pad_added_code, pad_removed_code, dict_msg, dict_code)  
    ftr = extracted_cc2ftr(data=data, params=params)
    return ftr

if __name__ == '__main__':
    ftr = learned_feature('/Users/haoye.tian/Documents/University/data/PatchCollectingV3UniqueToken/Developer/Correct/Chart/1/patch1-part1-Chart-1-Developer.patch')
    print(ftr)