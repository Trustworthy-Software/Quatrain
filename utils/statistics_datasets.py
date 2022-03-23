import os
import shutil

def statistics_patch(path_patch, root_folder, dataset_specific='', benchmark_specific='', tool_specific=''):
    cnt, correct, incorrect = 0, 0, 0
    datasets = os.listdir(path_patch)
    for dataset in datasets:
        if dataset_specific != '' and dataset != dataset_specific:
            continue
        path_dataset = os.path.join(path_patch, dataset)
        benchmarks = os.listdir(path_dataset)
        for benchmark in benchmarks:
            if benchmark_specific != '' and benchmark != benchmark_specific:
                continue
            path_benchmark = os.path.join(path_dataset, benchmark)
            tools = os.listdir(path_benchmark)
            for tool in tools:
                if tool_specific != '' and tool != tool_specific:
                    continue
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
                            # if label == 'Incorrect' and project == 'Closure' and id == '81':
                            #     print(tool)
                            for p in patches:
                                if not p.startswith('patch'):
                                    patches.remove(p)
                            number_patches = len(patches)
                            if tool == 'Developer':
                                cnt += number_patches
                            if label == 'Correct':
                                correct += number_patches
                            elif label == 'Incorrect':
                                incorrect += number_patches
                            else:
                                raise
    print('cnt: {}, correct: {}, incorrect: {}'.format(cnt, correct, incorrect))


if __name__ == '__main__':
    path_patch = '/Users/haoye.tian/Documents/ISSTA2022withTextUnique'
    os.system('find {} -name ".DS_Store" |xargs rm'.format(path_patch))

    # path_patch = '/Users/haoye.tian/Documents/ISSTA2022withTextUnique'
    root_folder = path_patch.split('/')[-1]
    dataset_specific = ''
    benchmark_specific = ''
    tool_specific = ''

    print(tool_specific)
    statistics_patch(path_patch, root_folder, dataset_specific=dataset_specific, benchmark_specific=benchmark_specific, tool_specific=tool_specific)