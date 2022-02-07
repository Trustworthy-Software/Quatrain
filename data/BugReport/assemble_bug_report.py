import os
import json

path = '../BugReport'

def assemble(path):
    bug_reprot_dict = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                with open(os.path.join(root, file), 'r+') as f:
                    for line in f:
                        bugid = line.split('$$')[0]
                        summary = line.split('$$')[1]
                        description = line.split('$$')[2]
                        bug_reprot_dict[bugid] = [summary, description]

    with open('./Bug_Report_All.json', 'w+') as f2:
        json.dump(bug_reprot_dict, f2)


if __name__ == '__main__':
    assemble(path)