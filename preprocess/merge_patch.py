import os
import shutil

def merge_patch(path_patch, root_folder):
    cnt, succ = 0, 0
    cnt_patch = 0
    dict_patch_text = {}
    datasets = os.listdir(path_patch)
    for dataset in datasets:
        path_dataset = os.path.join(path_patch, dataset)
        benchmarks = os.listdir(path_dataset)
        for benchmark in benchmarks:
            path_benchmark = os.path.join(path_dataset, benchmark)
            tools = os.listdir(path_benchmark)
            for tool in tools:
                path_tool = os.path.join(path_benchmark, tool)
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
                                cnt_patch += 1
                                if benchmark == 'Bugsjar' and '+' in project:
                                        project1 = project.split('+')[1]
                                        project1 = project1.lower()
                                        project_id = project1 + '-' + id
                                else:
                                    project_id = project + '-' + id
                                # print('{}: collecting {}'.format(cnt_patch, project_id))
                                label_int = 1 if label == 'Correct' else 0
                                project_id = project_id.lower()

                                patch_all = ''
                                path_single_patch = os.path.join(path_id, patch)
                                for root, dirs, files in os.walk(path_single_patch):
                                    for file in files:
                                        if file.endswith('.patch'):
                                            try:
                                                with open(os.path.join(root, file), 'r+') as f:
                                                    patch_all += ''.join(f.readlines())
                                            except Exception as e:
                                                print(e)
                                                continue
                                # save complete patch
                                name = patch + '-' + project_id + '_' + tool + '.patch'
                                path_single_patch = path_single_patch.replace(root_folder, root_folder+'_Merged')
                                if not os.path.exists(path_single_patch):
                                    os.makedirs(path_single_patch)
                                with open(os.path.join(path_single_patch, name), 'w+') as f:
                                    f.write(patch_all)


if __name__ == '__main__':
    path_patch = '/Users/haoye.tian/Documents/ISSTA2022withTextUnique'
    root_folder = path_patch.split('/')[-1]
    merge_patch(path_patch, root_folder)