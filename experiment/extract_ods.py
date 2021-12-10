import os
import shutil
from subprocess import *

import numpy as np

import config
import json

def obtain_ods_features(path_dataset):
    total = 0
    generated = 0
    for root, dirs, files in os.walk(path_dataset):
        for file in files:
            if file.endswith('.patch'):
                total += 1

                name = file.split('.')[0]

                buggy = name + '_s.java'
                fixed = name + '_t.java'

                feature_name = 'features_' + name.split('_')[0] + '.json'
                if os.path.exists(os.path.join(root, feature_name)):
                    generated += 1
                    continue

                frag = root.split('/')
                location = '/'.join(frag[:-2])

                # if name == 'patch1-Closure-43-Developer':
                #     continue

                # cmd = 'java -classpath /Users/haoye.tian/Documents/University/project/coming_tian/' \
                #       'target/coming-0-SNAPSHOT-jar-with-dependencies.jar  fr.inria.coming.main.ComingMain ' \
                #       '-mode features -parameters cross:false -input filespair -location {}:{} -output {}'.format(os.path.join(root,buggy), os.path.join(root,fixed), root)
                cmd = 'java -classpath /Users/haoye.tian/Documents/University/project/coming/' \
                      'target/coming-0-SNAPSHOT-jar-with-dependencies.jar  fr.inria.coming.main.ComingMain ' \
                      '-mode features -parameters cross:false -input files -location {} -output {}'.format(location, root)
                try:
                    with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                        output, errors = p.communicate(timeout=180)
                        # print(output)
                        # if errors:
                        #     raise CalledProcessError(errors, '-1')
                        # if output == '' or 'Exception' in output:
                        if not os.path.exists(os.path.join(root, feature_name)) or output == '':
                            print('Name: {}, Error: {}'.format(name, errors))
                            continue
                except Exception as e:
                    print(e)
                    continue
                generated += 1
                print('generated: {}/{}. new: {}'.format(generated, total, name))
    print('total: {}, generated: {}'.format(total, generated))

def merge_json(path_patch):
    cnt, succ = 0, 0
    dict_ods_feature = {}
    tools = os.listdir(path_patch)
    for tool in tools:
        path_tool = os.path.join(path_patch, tool)
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
                        ods_feature_combined = []
                        path_single_patch = os.path.join(path_id, patch)
                        for root, dirs, files in os.walk(path_single_patch):
                            for file in files:
                                if file.endswith('.json'):
                                    ods_feature = engineered_features(os.path.join(root, file))
                                    if ods_feature != []:
                                        ods_feature_combined.append(ods_feature)
                        if ods_feature_combined == []:
                            # print('no feature')
                            continue
                        else:
                            if len(ods_feature_combined) > 1:
                                print('multiple files')
                            ods_feature_combined = np.sum(ods_feature_combined, axis=0).tolist()
                            key = '-'.join([patch, project, id, tool])
                            value = ods_feature_combined
                            dict_ods_feature[key] = value
                            succ += 1

    print('cnt: {}, succ: {}'.format(cnt, succ))
    # save ods feature
    with open('../data/ODS_feature.json', 'w+') as f:
        json.dump(dict_ods_feature, f)



def engineered_features(path_json):
    other_vector = []
    P4J_vector = []
    repair_patterns = []
    repair_patterns2 = []
    try:
        with open(path_json, 'r') as f:
            feature_json = json.load(f)
            features_list = feature_json['files'][0]['features']
            P4J = features_list[-3]
            RP = features_list[-2]
            RP2 = features_list[-1]

            '''
            # other
            for k,v in other.items():
                # if k.startswith('FEATURES_BINARYOPERATOR'):
                #     for k2,v2 in other[k].items():
                #         for k3,v3 in other[k][k2].items():
                #             if v3 == 'true':
                #                 other_vector.append('1')
                #             elif v3 == 'false':
                #                 other_vector.append('0')
                #             else:
                #                 other_vector.append('0.5')
                if k.startswith('S'):
                    if k.startswith('S6'):
                        continue
                    other_vector.append(v)
                else:
                    continue
            '''

            # P4J
            if not list(P4J.keys())[100].startswith('P4J'):
                raise
            for k,v in P4J.items():
                # dict = P4J[i]
                # value = list(dict.values())[0]
                P4J_vector.append(int(v))

            # repair pattern
            for k,v in RP['repairPatterns'].items():
                repair_patterns.append(v)

            # repair pattern 2
            for k,v in RP2.items():
                repair_patterns2.append(v)

            # for i in range(len(features_list)):
            #     dict_fea = features_list[i]
            #     if 'repairPatterns' in dict_fea.keys():
            #             # continue
            #             for k,v in dict_fea['repairPatterns'].items():
            #                 repair_patterns.append(int(v))
            #     else:
            #         value = list(dict_fea.values())[0]
            #         engineered_vector.append(value)
    except Exception as e:
        print('name: {}, exception: {}'.format(path_json, e))
        return []

    if len(P4J_vector) != 156 or len(repair_patterns) != 26 or len(repair_patterns2) != 13:
        print('name: {}, exception: {}'.format(path_json, 'null feature or shape error'))
        return []

    # return engineered_vector
    return P4J_vector + repair_patterns + repair_patterns2

if __name__ == '__main__':
    path_dataset = cf = config.Config().path_patch
    # 1. extract ods feature of each file changed by patches. note that one patch could change several files.
    # obtain_ods_features(path_dataset)

    # 2. merge feature vectors of different changed files one patch involves
    for benchmark in os.listdir(path_dataset):
        merge_json(os.path.join(path_dataset, benchmark))