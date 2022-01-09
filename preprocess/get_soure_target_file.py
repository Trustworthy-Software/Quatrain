import os
import json
import shutil
from subprocess import *

def get_source_file(path_patch, benmark):
    found, cnt = 0, 0
    for root, dirs, files in os.walk(path_patch):
        for file in files:
            if file.endswith('.patch'):
                cnt += 1
                name = file.split('.')[0]
                tool = name.split('_')[1]

                half_name = name.split('_')[0]

                if half_name.count('-') == 2:
                    project = half_name.split('-')[1]
                    id = half_name.split('-')[2]
                elif half_name.count('-') == 3:
                    project = half_name.split('-')[1] + '-' + half_name.split('-')[2]
                    id = half_name.split('-')[3]

                with open(os.path.join(root, file), 'r+') as f:
                    for line in f:
                        if line.startswith('--- '):
                            file_changed = line.split(' ')[1].split('\t')[0].split('/')[-1].split('.')[0].strip()

                flag = False
                project = project.lower()
                root_diff = os.path.join(source_github, benmark, project, project.lower()+'_'+id,)
                for root2, dirs2, files2 in os.walk(root_diff):
                    for file2 in files2:
                        name2 = file2.split('.java')[0]
                        if name2 == file_changed:
                            found += 1
                            flag = True
                            break
                # if not flag:
                #     print(file_changed)
    print('cnt: {}, success: {}'.format(cnt, found))

def get_source_bugsjar(path_patch, benmark):
    cnt, found, nofound = 0, 0, 0
    for root, dirs, files in os.walk(path_patch):
        for file in files:
            if file.endswith('.patch'):
                cnt += 1
                print('{}: {}'.format(cnt, file))
                name = file.split('.')[0]
                tool = name.split('_')[1]
                half_name = name.split('_')[0]

                # skip existed source file
                new_name = name + '_s.java'
                target = os.path.join(root, new_name)
                if os.path.exists(target):
                    continue

                if half_name.count('-') == 2:
                    project = half_name.split('-')[1]
                    id = half_name.split('-')[2]
                elif half_name.count('-') == 3:
                    project = half_name.split('-')[1] + '-' + half_name.split('-')[2]
                    id = half_name.split('-')[3]

                with open(os.path.join(root, file), 'r+') as f:
                    for line in f:
                        if line.startswith('--- '):
                            path_file_changed_full = line.split(' ')[1].split('\t')[0].strip()
                            if tool == 'Developer':
                                path_file_changed = path_file_changed_full[2:]
                            else:
                                path_list = path_file_changed_full.split('/')[3:]
                                path_file_changed = '/'.join(path_list)
                if path_file_changed == 'ev/null':
                    with open(os.path.join(root, file), 'r+') as f:
                        for line in f:
                            if line.startswith('+++ '):
                                path_file_changed_full = line.split(' ')[1].split('\t')[0].strip()
                                if tool == 'Developer':
                                    path_file_changed = path_file_changed_full[2:]
                                else:
                                    path_list = path_file_changed_full.split('/')[3:]
                                    path_file_changed = '/'.join(path_list)

                # get branch
                if project == 'Commons+Math' or project == 'Jackrabbit+Oak':
                    project = project.replace('+', '-')
                cmd = 'cd {} && cd {} && git branch -a | grep bugs-dot-jar_ |grep remotes| grep {}'.format(github_repository_bugsjar, project, id)
                try:
                    with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                        project_branch, errors = p.communicate(timeout=300)
                        # print(output)
                        if errors:
                            raise CalledProcessError(errors, '-1')
                except Exception as e:
                    print(e)
                    raise

                # switch branch
                project_branch = project_branch.strip().split('/')[-1]

                cmd = 'cd {} && cd {} && git checkout -f {}'.format(github_repository_bugsjar, project, project_branch)
                try:
                    with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                        result, errors = p.communicate(timeout=300)
                        # print(output)
                        # if errors:
                        #     raise CalledProcessError(errors, '-1')
                except Exception as e:
                    print(e)
                    raise

                # get source file
                path_file = os.path.join(github_repository_bugsjar, project, path_file_changed)
                if os.path.exists(path_file):
                    # print('success')
                    pass
                    # found += 1
                else:
                    nofound += 1
                    print('no')

                cmd = 'cp {} {}'.format(path_file, target)
                try:
                    with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                        result, errors = p.communicate(timeout=300)
                        # print(output)
                        if errors:
                            raise name
                except Exception as e:
                    print(e)
                    # raise
    print('cnt: {}, success: {}'.format(cnt, cnt-nofound))

def get_source_from_bug(path_patch, benmark, github_buggy):

    found, cnt = 0, 0
    for root, dirs, files in os.walk(path_patch):
        for file in files:
            if file.endswith('.patch'):
                cnt += 1
                print(cnt)
                name = file.split('.')[0]
                tool = name.split('_')[1]

                half_name = name.split('_')[0]

                if half_name.count('-') == 2:
                    project = half_name.split('-')[1]
                    id = half_name.split('-')[2]
                elif half_name.count('-') == 3:
                    project = half_name.split('-')[1] + '-' + half_name.split('-')[2]
                    id = half_name.split('-')[3]

                with open(os.path.join(root, file), 'r+') as f:
                    for line in f:
                        if line.startswith('--- '):
                            path_file_changed_full = line.split(' ')[1].split('\t')[0].strip()
                            if tool == 'Developer':
                                path_file_changed = path_file_changed_full[2:]
                            else:
                                path_list = path_file_changed_full.split('/')[3:]
                                path_file_changed = '/'.join(path_list)
                if path_file_changed == 'ev/null':
                    with open(os.path.join(root, file), 'r+') as f:
                        for line in f:
                            if line.startswith('+++ '):
                                path_file_changed_full = line.split(' ')[1].split('\t')[0].strip()
                                if tool == 'Developer':
                                    path_file_changed = path_file_changed_full[2:]
                                else:
                                    path_list = path_file_changed_full.split('/')[3:]
                                    path_file_changed = '/'.join(path_list)

                # get source file
                if benmark == 'Bears':
                    project_id = project+'-'+id
                elif benmark == 'Defects4J':
                    project_id = project + '_' + id
                path_file = os.path.join(github_buggy, project_id, path_file_changed)
                if os.path.exists(path_file):
                    # print('success')
                    found += 1
                else:
                    print('no')
                    # raise
                new_name = name + '_s.java'
                cmd = 'cp {} {}'.format(path_file, os.path.join(root, new_name))
                try:
                    with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                        result, errors = p.communicate(timeout=300)
                        # print(output)
                        if errors:
                            # raise CalledProcessError(errors, '-1')
                            raise name
                except Exception as e:
                    print(e)
                    # raise
    print('cnt: {}, success: {}'.format(cnt, found))

def apply_patch(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.patch'):
                name = file.split('.patch')[0]

                patch = file
                source_file = name + '_s.java'
                target_file = name + '_t.java'
                project = name.split('_')[0].split('-')[1]
                path_patch = os.path.join(root, patch)

                # if project == 'Chart':
                path_source_file = os.path.join(root, source_file)
                path_target_file = os.path.join(root, target_file)

                if not (os.path.exists(path_source_file) or os.path.exists(path_target_file)):
                    continue
                # dos2unix for source file and patch
                os.system('dos2unix  ' + path_source_file)
                shutil.copy(path_source_file, path_target_file)
                os.system('dos2unix  ' + path_patch)
                # else:
                #     path_source_file = os.path.join(root, source_file)
                #     path_target_file = os.path.join(root, target_file)
                #     shutil.copy(path_source_file, path_target_file)

                # cmd = 'patch -p0 {} {}'.format(path_target_file, os.path.join(root, patch))
                cmd = 'patch -u {} -i {}'.format(path_target_file, path_patch)
                try:
                    with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                        output, errors = p.communicate(timeout=300)
                        # print(output)
                        if errors:
                            raise CalledProcessError(errors, '-1')
                        if 'FAILED' in output:
                            print('FAILED')
                            continue
                except Exception as e:
                    print('name: {}, exc: {}'.format(name, e))
                    continue

path = '/Users/haoye.tian/Documents/PatchNaturalnessYeTemp'

if __name__ == '__main__':
    print('start')
    github_buggy_bears = '/Users/haoye.tian/Documents/University/project/Bears-bug'
    github_buggy_defects4j = '/Users/haoye.tian/Documents/University/project/defects4j_buggy'
    github_repository_bugsjar = '/Users/haoye.tian/Documents/University/project/bugs-dot-jar'

    # 1.1 get source file for incorrect patches
    get_source_bugsjar(os.path.join(path, 'Bugsjar'), 'Bugsjar')
    # get_source_from_bug(os.path.join(path, 'Bears'), 'Bears', github_buggy_bears)
    # get_source_from_bug(os.path.join(path, 'Defects4J'), 'Defects4J', github_buggy_defects4j)

    # 1.2 get source file for developer/correct patches
    # get_source_from_bug(os.path.join(path, 'Bears/Developer'), 'Bears', github_buggy_bears)
    get_source_bugsjar(os.path.join(path, 'Bugsjar/Developer'), 'Bugsjar')

    # 2. apply to get target file
    apply_patch(path)