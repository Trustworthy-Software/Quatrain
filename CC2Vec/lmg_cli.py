import argparse
import pickle
import re
from os import walk, path
from CC2Vec.lmg_padding import processing_data
from CC2Vec.lmg_utils import mini_batches, commit_msg_label
from tqdm import tqdm
import torch
from CC2Vec.lmg_cc2ftr_model import HierachicalRNN

def read_args_lmg():
    parser = argparse.ArgumentParser()

    # Preprocess mode
    parser.add_argument('-preprocess', action='store_true', help='Preprocess data')
    parser.add_argument('-data', type=str, help='the directory of our raw data')
    parser.add_argument('-output_dir', type=str, help='the path to the output directory of the pre-processed data')

    # Inference mode
    parser.add_argument('-infer', action='store_true', help='Preprocess data')
    parser.add_argument('-load_model', type=str, default=None, help='loading our model')
    parser.add_argument('-patch_file', type=str, help='the patch file')
    parser.add_argument('-dictionary_data', type=str, default='./data/jit/openstack_dict.pkl', help='the directory of our dicitonary data')

    # Number of parameters for reformatting commits        
    parser.add_argument('-code_line', type=int, default=15, help='the number of LOC in each hunk of commit code')
    parser.add_argument('-code_length', type=int, default=40, help='the length of each LOC of commit code')
    parser.add_argument('-batch_size', type=int, default=1, help='batch size. Must be 1')
    parser.add_argument('-embed_size', type=int, default=16, help='the dimension of embedding vector')
    parser.add_argument('-hidden_size', type=int, default=8, help='the number of nodes in hidden states of wordRNN, sentRNN, and hunkRNN')
    parser.add_argument('-hidden_units', type=int, default=256, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout for training PatchNet')
    parser.add_argument('-l2_reg_lambda', type=float, default=1e-5, help='regularization rate')

    # CUDA
    parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=True, help='disable the GPU')

    return parser

def get_files(directory, extension='patch'):
    for (dirpath, dirnames, filenames) in walk(directory):
        for f in filenames:
            if f.endswith('.' + extension):
                yield dirpath + '/' + f

def get_patch_cc2v(patch):
    with open(patch, 'r') as file:
        lines = ''
        p = r"([^\w_])"
        for line in file:
            line = line.strip()
            if line != '':
                if line.startswith('@@') or line.startswith('diff') or line.startswith('index'):
                    continue
                if line == '+++':
                    newline = 'ppp <nl> '
                    lines += newline
                elif line.startswith('---') or line.startswith('-- ') or line.startswith('PATCH_DIFF_ORIG=---'):
                    # newline = re.split(pattern=p,string=line.split(' ')[1].strip())
                    # newline = 'mmm ' + ' '.join(newline) + ' <nl> '

                    # set null
                    newline = 'mmm <nl> '
                    lines += newline
                elif line.startswith('+++') or line.startswith('++ '):
                    # newline = re.split(pattern=p, string=line.split(' ')[1].strip())
                    # newline = 'ppp ' + ' '.join(newline) + ' <nl> '

                    # set null
                    newline = 'ppp <nl> '
                    lines += newline
                else:
                    newline = re.split(pattern=p, string=line.strip())
                    newline = [x.strip() for x in newline]
                    while '' in newline:
                        newline.remove('')
                    newline = ' '.join(newline) + ' <nl> '
                    lines += newline
        return lines

def extracted_cc2ftr(data, params):
    msg, pad_added_code, pad_removed_code, dict_msg, dict_code = data
    labels = commit_msg_label(data=msg, dict_msg=dict_msg)
    batches = mini_batches(X_added_code=pad_added_code, X_removed_code=pad_removed_code, Y=labels, mini_batch_size=params.batch_size, Shuffled=False)

    # return batches
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
        
    params.vocab_code = len(dict_code)
    
    params.class_num = batches[0][2].shape[1]
    params.code_lines = batches[0][0].shape[1]

    # # Device configuration
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HierachicalRNN(args=params)
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(params.load_model))
        model = model.cuda()
    else:
        model.load_state_dict(torch.load(params.load_model, map_location=torch.device('cpu')))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():        
        commit_ftrs = list()
        for i, (batch) in enumerate(tqdm(batches)):
            state_word = model.init_hidden_word()
            state_sent = model.init_hidden_sent()
            state_hunk = model.init_hidden_hunk()

            pad_added_code, pad_removed_code, label = batch          
            commit_ftr = model.forward_commit_embeds_diff(pad_added_code, pad_removed_code, state_hunk, state_sent, state_word)                
            commit_ftrs.append(commit_ftr)
        if torch.cuda.is_available():
            commit_ftrs = torch.cat(commit_ftrs).cuda().detach().numpy()
        else:
             commit_ftrs = torch.cat(commit_ftrs).cpu().detach().numpy()

        return commit_ftrs

if __name__ == '__main__':
    params = read_args_lmg().parse_args()
    if params.preprocess == True:
        data = [list(), list()]
        i = 0
        for file in get_files(params.data):
            try:
                data[1].append(get_patch_cc2v(file))
                data[0].append('<NULL>')
            except:
                i = i + 1
                print(file)

        print(i)

        tmp = params.data.split('/')
        filename = tmp[-1] if tmp[-1] else tmp[-2]
        with open(path.join(params.output_dir, filename + '.pkl'), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    elif params.infer == True:
        data = [list(), list()]
        data[1].append(get_patch_cc2v(params.patch_file))
        data[0].append('<NULL>')
        # data = pickle.load(open(params.patch_file, 'rb'))
        msg, diff = data[0], data[1]
        dictionary = pickle.load(open(params.dictionary_data, 'rb'))
        pad_added_code, pad_removed_code = processing_data(code=diff, dictionary=dictionary, params=params)
        dict_msg, dict_code = dictionary

        data = (msg, pad_added_code, pad_removed_code, dict_msg, dict_code)  
        ftr = extracted_cc2ftr(data=data, params=params)
        print(ftr)
