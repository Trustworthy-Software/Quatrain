import os
from subprocess import *
import yaml
import re
import artifact_detection_model.transformer.ArtifactRemoverTransformer

# git clone project from 'https://github.com/bugs-dot-jar/bugs-dot-jar'
path_bugs_dot_jar = '/Users/haoye.tian/Documents/University/project/bugs-dot-jar'
save_bug_report = '../data/BugReport/Bugsjar_bugreport.txt'

def checkout(path_bugs_dot_jar):
    projects = os.listdir(path_bugs_dot_jar)
    result = ''
    cnt = 0
    for project in projects:
        if project.startswith('.') or project == 'README.md':
            continue
        path_project = os.path.join(path_bugs_dot_jar, project)
        # get branch list
        cmd = 'cd {} && git branch -a|grep bugs-dot-jar_'.format(path_project)
        try:
            with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                project_branch, errors = p.communicate(timeout=300)
                # print(output)
                if errors:
                    raise CalledProcessError(errors, '-1')
        except Exception as e:
            print(e)
            raise

        project_branch_list = project_branch.split('\n')
        for branch in project_branch_list:
            if not 'remotes/origin/' in branch:
                continue
            b = branch.split('remotes/origin/')[1]
            # print(b)

            # check out branch
            cmd2 = 'cd {} && git checkout -f {} '.format(path_project, b)
            try:
                with Popen(cmd2, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                    output, errors = p.communicate(timeout=300)
                    # print(output)
            except Exception as e:
                print(e)
                raise

            # get bug report path
            path_bug_report = os.path.join(path_project, '.bugs-dot-jar/bug-report.yml')
            with open(path_bug_report, 'r+') as f:
                bug_report = yaml.load(f, Loader=yaml.FullLoader)

            bug_id = bug_report['BugID']
            # summary = bug_report['Summary'].replace('$$', ' ')
            summary = re.sub(r'\r|\n|\$', ' ', bug_report['Summary'])
            description = 'None'
            if bug_report['Description'] != None:
                # description = bug_report['Description'].replace('\n || \r || $$', ' ').replace('$$', ' ')
                # 1. original
                description = re.sub(r'\r|\n|\$', ' ', bug_report['Description'])
                # # 2. cleaned bug report with model
                # model_pretrained = artifact_detection_model.transformer.ArtifactRemoverTransformer.ArtifactRemoverTransformer()
                # original = bug_report['Description']
                # cleaned = model_pretrained.transform2([original])[0]
                # description = re.sub(r'\r|\n|\$', ' ', cleaned)

            project_id = b.replace('bugs-dot-jar_', '')
            # make sure the link between bug id and corresponding bug report is correct
            if bug_id not in project_id:
                raise ('mismatch bug id and bug report')
            # use new project id
            project_id_new = project_id.split('-')[0] + '-' + project_id.split('_')[1]
            result += project_id_new + '$$' + summary.strip() + '$$' + description.strip() + '\n'

            cnt += 1
            print('{}: {}'.format(cnt, project_id))
    with open(save_bug_report, 'w+') as f:
        f.write(result)

if __name__ == '__main__':
    checkout(path_bugs_dot_jar)