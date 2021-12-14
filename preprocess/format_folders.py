import os
import shutil


# create a middle folder for patch that changes multiple files. folder tree is like: Chart/1/patch1/patch1#1... and Chart/1/patch1/patch1#2.... folder "patch1" will be created in this script.
def copy_file(path_patch):
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
                        patch_name = patch.split('-')[0]
                        if '#' in patch_name:
                            patch_name = patch_name.split('#')[0]

                        path_single_patch = os.path.join(path_id, patch)
                        for root, dirs, files in os.walk(path_single_patch):
                            for file in files:

                                old_folder = root
                                new_folder = old_folder.replace(path_id, path_id+'/'+patch_name)
                                new_folder = new_folder.replace('PatchNaturalness', 'PatchNaturalness2')
                                if not os.path.exists(new_folder):
                                    os.makedirs(new_folder)

                                old_file = os.path.join(old_folder, file)
                                new_file = os.path.join(new_folder, file)

                                shutil.copy(old_file, new_file)


def format_folder4ODS(path_patch):
    old = path_patch.split('/')[-1]
    new = old + '2'
    for root, dirs, files in os.walk(path_patch):
        for file in files:
            if file.endswith('.patch'):
                name = file.split('.')[0]

                folder0 = name
                folder1 = name.split('_')[0]
                folder2 = name.split('_')[1]

                new_root = root.replace(old, new)
                new_folder = os.path.join(new_root, name, folder1, folder2)
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)

                shutil.copy(os.path.join(root, file), os.path.join(new_folder, file))


path_patch = '/Users/haoye.tian/Documents/PatchNaturalness/Defects4J'
path_patch_ye = '/Users/haoye.tian/Documents/PatchNaturalnessYe'

# copy_file(path_patch)
format_folder4ODS(path_patch_ye)