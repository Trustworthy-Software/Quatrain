import os 
import torch
import numpy as np
import math
import re

def commit_msg_label(data, dict_msg):
    labels_ = np.array([1 if w in d.split() else 0 for d in data for w in dict_msg])
    labels_ = np.reshape(labels_, (int(labels_.shape[0] / len(dict_msg)), len(dict_msg)))
    return labels_

def save(model, save_dir, save_prefix, epochs):
    if not os.path.isdir(save_dir):       
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}.pt'.format(save_prefix, epochs)
    torch.save(model.state_dict(), save_path)

def mini_batches(X_added_code, X_removed_code, Y, mini_batch_size=64, seed=0, Shuffled=True):
    m = Y.shape[0]  # number of training examples
    mini_batches = []

    if Shuffled == True:
        np.random.seed(seed)
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))    
        shuffled_X_added = X_added_code[permutation, :, :]
        shuffled_X_removed = X_removed_code[permutation, :, :]
        
        if len(Y.shape) == 1:
            shuffled_Y = Y[permutation]
        else:
            shuffled_Y = Y[permutation, :]
    else:
        shuffled_X_added = X_added_code
        shuffled_X_removed = X_removed_code
        shuffled_Y = Y

    # Step 2: Partition (X, Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / float(mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):                
        mini_batch_X_added = shuffled_X_added[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        mini_batch_X_removed = shuffled_X_removed[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X_added, mini_batch_X_removed, mini_batch_Y)
        mini_batches.append(mini_batch)        
    return mini_batches

def get_lines(patch):
    with open(patch, 'r') as file:
        lines = ''
        p = r"([^\w_])"
        for line in file:
            if line != '':
                if line.startswith('@@') or line.startswith('diff') or line.startswith('index'):
                    continue
                if line == '+++':
                    newline = ''
                    lines += newline
                elif line.startswith('---') or line.startswith('PATCH_DIFF_ORIG=---'):
                    newline = re.split(pattern=p, string=line.split(' ')[1].strip())
                    newline = ' '.join(newline)
                    lines += newline
                elif line.startswith('+++'):
                    newline = re.split(pattern=p, string=line.split(' ')[1].strip())
                    newline = ' '.join(newline)
                    lines += newline
                else:
                    newline = re.split(pattern=p, string=line.strip())
                    newline = [x.strip() for x in newline]
                    while '' in newline:
                        newline.remove('')
                    newline = ' '.join(newline)
                    lines += newline
        return lines

def get_files_once(folder, extension):
    files = list()
    for dirpath, _, _files in os.walk(folder):  
        for filename in _files: 
            fname = os.path.join(dirpath,filename) 
            if fname.endswith('.' + extension): 
                files.append(fname)
    return files

def update_dict(old_dictionary, params):
    dict_msg, dict_code = old_dictionary
    files = get_files_once(params.new_data_dir, 'patch')
    tokens = list()
    for f in files:
        try:
            lines = get_lines(f)
            tokens.extend(lines.split())
        except:
            print("Cannot read properly file : %s" % f)
    tokens = set(tokens)
    l = len(dict_code)
    keys = dict_code.keys()
    for t in tokens:
        if t not in keys:
            dict_code[t] = l
            l = l + 1
    
    new_dictionary = (dict_msg, dict_code)
    return new_dictionary

