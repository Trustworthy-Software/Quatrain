import experiment.config as config
import os
import shutil
from nltk.tokenize import word_tokenize
import pickle

path_dataset = config.Config().path_patch

def start(path_dataset):
    # drop same patch
    if 'Unique' in path_dataset:
        print('already deduplicated!')
    else:
        dataset_name = path_dataset.split('/')[-1]
        path_dataset, dataset_name = deduplicate_by_token_with_location(dataset_name, path_dataset)

def deduplicate_by_token_with_location(dataset_name, path_dataset):
    new_dataset_name = dataset_name + 'Unique'
    new_dataset_path = path_dataset.replace(dataset_name, new_dataset_name)
    unique_dict = {}
    pre = exception = post = repeat = 0
    pre_co, pre_in, post_co, post_in = 0, 0, 0, 0
    deduplicated_name = []
    for root, dirs, files in os.walk(path_dataset):
        for file in files:
            if file.endswith('.patch'):
                # unlabeled
                if 'iFixR' in root:
                    continue
                path_patch = os.path.join(root, file)
                unique_str = ''
                pre += 1
                if '/Correct/' in path_patch:
                    pre_co += 1
                elif '/Incorrect/' in path_patch:
                    pre_in += 1
                print('{}, name: {}'.format(pre, file.split('.')[0]))
                try:
                    with open(path_patch, 'r') as f:
                        foundAT = False
                        for line in f:
                            if line.startswith('--') or line.startswith('++') or line.strip() == '':
                                continue
                            if not foundAT and not line.startswith('@@ '):
                                continue
                            else:
                                if line.startswith('@@ '):
                                    foundAT = True
                                    # unique_str += ' '.join(word_tokenize(line.strip())) + ' '
                                elif line.startswith('-') or line.startswith('+'):
                                    unique_str += ' '.join(word_tokenize(line[1:].strip())) + ' '
                                else:
                                    unique_str += ' '.join(word_tokenize(line.strip())) + ' '
                except Exception as e:
                    print('Exception: {}'.format(e))
                    exception += 1
                    continue

                if unique_str in unique_dict:
                    unique_dict[unique_str] += 1
                    repeat += 1
                    continue
                else:
                    unique_dict[unique_str] = 0

                    # copy unique to another folder
                    name = file.split('.')[0]
                    new_path = root.replace(dataset_name, new_dataset_name)
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)

                    deduplicated_name.append(name)
                    patch = file
                    commit = name + '.txt'
                    buggy = name + '_s.java'
                    fixed = name + '_t.java'
                    feature = 'features_' + buggy + '->' + fixed + '.json'

                    # try:
                    #     shutil.copy(os.path.join(root, patch), os.path.join(new_path, patch))
                    #     shutil.copy(os.path.join(root, commit), os.path.join(new_path, commit))
                    #     # shutil.copy(os.path.join(root, buggy), os.path.join(new_path, buggy))
                    #     # shutil.copy(os.path.join(root, fixed), os.path.join(new_path, fixed))
                    #     if os.path.exists(os.path.join(root, feature)):
                    #         shutil.copy(os.path.join(root, feature), os.path.join(new_path, feature))
                    # except Exception as e:
                    #     print(e)
                    #     continue

                    post += 1
                    if '/Correct/' in path_patch:
                        post_co += 1
                    elif '/Incorrect/' in path_patch:
                        post_in += 1
    print('pre:{}, post:{} ---- exception:{}, repeat:{}'.format(pre, post, exception, repeat))
    print('pre_correct:{}, pre_incorrect:{}.  post_correct:{}, post_incorrect:{}'.format(pre_co, pre_in, post_co, post_in))
    # print('remember change path in config_.py !!!')
    pickle.dump(deduplicated_name, open('deduplicated_name.pickle', 'wb'))
    return new_dataset_path, new_dataset_name

if __name__ == '__main__':
    start(path_dataset)