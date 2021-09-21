import os
import shutil
import json
import pickle
from representation.word2vec import Word2vector

path_patch_sliced = '/Users/haoye.tian/Documents/University/data/PatchCollectingV1Natural_sliced/'

def patch_bert():
    w2v = Word2vector(method='bert', )
    cnt = 0
    projects = {'Chart': 26, 'Lang': 65, 'Time': 27, 'Closure': 176, 'Math': 106}
    # projects = {'Time': 2,}
    for project, number in projects.items():
        print('Berting {}'.format(project))
        for id in range(1, number + 1):
            tools = os.listdir(path_patch_sliced)
            for label in ['Correct', 'Incorrect']:
                for tool in tools:
                    path_bugid = os.path.join(path_patch_sliced, tool, label, project, str(id))
                    if os.path.exists(path_bugid):
                        patches = os.listdir(path_bugid)
                        for p in patches:
                            path_patch = os.path.join(path_bugid, p)
                            # json_key = '-'.join([path_patch.split('/')[-5], path_patch.split('/')[-4], path_patch.split('/')[-3], path_patch.split('/')[-2], path_patch.split('/')[-1]])
                            if not os.path.isdir(path_patch):
                                continue

                            cnt += 1

                            # change separate character
                            json_key = path_patch + '$.json'
                            json_key_cross = path_patch + '$cross.json'

                            if os.path.exists(json_key) and os.path.exists(json_key_cross):
                                print('exists!')
                                continue

                            try:
                                vector, vector_cross = w2v.convert_single_patch(path_patch)
                            except Exception as e:
                                print('error bert vector: {}'.format(e))
                                continue
                            vector_list = list(map(str, list(vector)))
                            vector_list_cross = list(map(str, list(vector_cross)))

                            with open(json_key, 'w+') as f:
                                jsonstr = json.dumps(vector_list, )
                                f.write(jsonstr)
                            with open(json_key_cross, 'w+') as f:
                                jsonstr = json.dumps(vector_list_cross, )
                                f.write(jsonstr)

                            print('{} json_key: {}'.format(cnt, json_key))

        # pickle.dump(dict, f)
        # f.write(jsonstr)

if __name__ == '__main__':
    patch_bert()