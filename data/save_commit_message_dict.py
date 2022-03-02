import os
import json
dirname = os.path.dirname(__file__)

file = os.path.join(dirname, './bugreport_patch.txt')
print(file)

def extract_commit_message(file):
    commit_message_dict = {}
    with open(file, 'r+') as f:
        for line in f:
            project_id = line.split('$$')[0].strip()
            bugreport_summary = line.split('$$')[1].strip()
            bugreport_description = line.split('$$')[2].strip()

            patch_id = line.split('$$')[3].strip()
            commit_content = line.split('$$')[4].strip()
            label = int(float(line.split('$$')[5].strip()))

            commit_message_dict[patch_id] = commit_content

    with open(os.path.join(dirname, 'CommitMessage/Generated_commit_message_All.json'), 'w+') as f2:
        json.dump(commit_message_dict, f2)


if __name__ == '__main__':
    extract_commit_message(file)