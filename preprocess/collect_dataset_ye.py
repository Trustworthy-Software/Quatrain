import json
import os
import csv
import shutil
import re

path_patch = '/Users/haoye.tian/Documents/University/project/ODSExperiment/RawRepairThemAllPatches'
new_root = '/Users/haoye.tian/Documents/PatchNaturalnessYe'

dict_patch_labels = {}
with open('../data/PatchLabelsYe.csv', 'r+') as f:
    patch_labels = csv.reader(f,)
    for item in patch_labels:
        if patch_labels.line_num == 1:
            continue
        dict_patch_labels[item[0]] = item[1]

def download_defects4j(path_patch):
    path_patch = os.path.join(path_patch, 'defects4j')
    root_defects4j = os.path.join(new_root, 'Defects4J')
    print(path_patch)
    for root, dirs, files in os.walk(path_patch):
        for file in files:
            if file.endswith('.patch'):
                name = file.split('.')[0]
                benchmark = root.split('/')[-4]
                project = root.split('/')[-3]
                id = root.split('/')[-2]
                tool = root.split('/')[-1]

                key = '_'.join([tool, name])
                if key not in dict_patch_labels.keys():
                    continue
                value = dict_patch_labels[key]
                if value == 'Overfitting':
                    label = 'Incorrect'
                elif value == 'Correct':
                    label = 'Correct'
                else:
                    raise ('lets debug')
                patchid = 'patch' + name.split('_')[-1]

                new_folder = '/'.join([root_defects4j, tool, label, project, id, patchid])
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                new_file = '-'.join([patchid, project, id]) + '_' + tool + '.patch'

                shutil.copy(os.path.join(root, file), os.path.join(new_folder, new_file))

def download_bears(path_patch):
    path_patch = os.path.join(path_patch, 'Bears')
    root_bears = os.path.join(new_root, 'Bears')
    print(path_patch)
    with open('../data/bears_index_dict_inverse.json', 'r+') as f:
        dict_index = json.load(f)
    for root, dirs, files in os.walk(path_patch):
        for file in files:
            if file.endswith('.patch'):
                name = file.split('.')[0]
                benchmark = root.split('/')[-4]
                project = root.split('/')[-3]
                id = root.split('/')[-2]
                tool = root.split('/')[-1]

                key = '_'.join([tool, name]) + '_'
                if key not in dict_patch_labels.keys():
                    continue
                value = dict_patch_labels[key]
                if value == 'Overfitting':
                    label = 'Incorrect'
                elif value == 'Correct':
                    label = 'Correct'
                else:
                    raise ('lets debug')
                patchid = 'patch' + name.split('_')[-1]

                # change project name to id
                pid_old = project + '-' + id
                projectId = dict_index[pid_old]
                project = projectId.split('-')[0]
                id = projectId.split('-')[1]

                new_folder = '/'.join([root_bears, tool, label, project, id, patchid])
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                new_file = '-'.join([patchid, project, id]) + '_' + tool + '.patch'

                shutil.copy(os.path.join(root, file), os.path.join(new_folder, new_file))

def download_bugsjar(path_patch):
    path_patch = os.path.join(path_patch, 'bugsjar')
    root_bears = os.path.join(new_root, 'Bugsjar')
    print(path_patch)
    cnt = 0
    same = set()
    for root, dirs, files in os.walk(path_patch):
        for file in files:
            if file.endswith('.patch'):
                name = file.split('.patch')[0]
                benchmark = root.split('/')[-4]
                project = root.split('/')[-3]
                id = root.split('/')[-2]
                tool = root.split('/')[-1]

                # files = []
                # with open(os.path.join(root, file), 'r+') as f:
                #     for line in f:
                #         if line.startswith('--- '):
                #             file_changed = line.split(' ')[1].split('\t')[0].split('/')[-1].split('.')[0].strip()
                # debug
                #             if file_changed == '' or file_changed == ' ' or file_changed == '\n':
                #                 print('')
                #                 pass
                #             if file_changed.endswith('java'):
                #                 file_changed = file_changed.replace('java', '')
                #             files.append(file_changed)
                #
                # for file_changed in files:
                #     key = '_'.join([tool, name, file_changed])
                #     if key in dict_patch_labels.keys():
                #         value = dict_patch_labels[key]
                #         cnt += 1
                #         break
                # else:
                #     print(files)
                #     continue

                key = '_'.join([tool, name])
                if not key.endswith('_'):
                    key = key + '_'
                value = ''
                for k, v in dict_patch_labels.items():
                    # if re.match(key, k):
                    if k.startswith(key):
                        value = v
                        cnt += 1
                        if key in same:
                            print('same')
                        else:
                            same.add(key)
                if value == '':
                    continue

                if value == 'Overfitting':
                    label = 'Incorrect'
                elif value == 'Correct':
                    label = 'Correct'
                else:
                    raise ('lets debug')
                patchid = 'patch' + name.split('_')[-1]

                new_folder = '/'.join([root_bears, tool, label, project, id, patchid])
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                new_file = '-'.join([patchid, project, id]) + '_' + tool + '.patch'

                shutil.copy(os.path.join(root, file), os.path.join(new_folder, new_file))
    print(cnt)

# all the generated patches are overfitting/incorrect.
# download_defects4j(path_patch)
# download_bears(path_patch)
download_bugsjar(path_patch)