import os, json
from subprocess import *
import pickle
from lxml import etree
from github import Github
import time
import requests
from urllib.request import urlopen
from urllib import error
import urllib
import logging
import socket
from bs4 import BeautifulSoup
from lxml import etree

timeout = 30
socket.setdefaulttimeout(timeout)
hdr = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive'}


def extract_defects4j(path_repo):
    dict = {}
    path_dev_commit_id = '../preprocess/Dev_fixed_commit'
    files = os.listdir(path_dev_commit_id)
    for file in files:
        if file.endswith('.txt'):
            project = file.split('.')[0]
            path_file = os.path.join(path_dev_commit_id, file)
            with open(path_file, 'r+') as f:
                for line in f:
                    id, commit_id = line.split(',')[0].strip(), line.split(',')[1].strip()
                    if project == 'Math':
                        path_project = os.path.join(path_repo, 'commons-math.git')
                    elif project == 'Lang':
                        path_project = os.path.join(path_repo, 'commons-lang.git')
                    elif project == 'Closure':
                        path_project = os.path.join(path_repo, 'closure-compiler.git')
                    elif project == 'Mockito':
                        path_project = os.path.join(path_repo, 'mockito.git')
                    elif project == 'Time':
                        path_project = os.path.join(path_repo, 'joda-time.git')
                    else:
                        continue
                    cmd = 'cd {} && git show {} --pretty=format:%s -s '.format(path_project, commit_id)
                    print(cmd)
                    try:
                        with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                            result, errors = p.communicate(timeout=300)
                            # print(output)
                            if errors:
                                raise CalledProcessError(errors, '-1')
                    except Exception as e:
                        print(e)
                        raise

                    project_id = project + '-' + id
                    project_id = project_id.lower()
                    result_list = result.split(' ')
                    if project.lower() in result_list[0].lower():
                        developer_commit_message = ' '.join(result_list[1:])
                    else:
                        developer_commit_message = ' '.join(result_list)
                    dict[project_id] =  developer_commit_message

    with open('../data/CommitMessage/Defects4j_developer_commit_mssage.json', 'w+') as f:
        json.dump(dict, f)

def extract_bugsjar(path_bugs_dot_jar):
    dict = {}
    path_dev_commit_id = '../preprocess/Dev_fixed_commit'
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
            id = b.split('_')[-1]
            commit_id = id
            if commit_id == 'd35d2d85':
                commit_id = 'd35d2d850b'
            cmd = 'cd {} && git show {} --pretty=format:%s -s '.format(path_project, commit_id)
            try:
                with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as p:
                    result, errors = p.communicate(timeout=300)
                    # print(output)
                    if errors:
                        raise CalledProcessError(errors, '-1')
            except Exception as e:
                print(e)
                raise
            if '-' in project:
                project = project.split('-')[1]
            project_id = project + '-' + id
            project_id = project_id.lower()
            result_list = result.split(' ')
            if project.lower() in result_list[0].lower():
                developer_commit_message = ' '.join(result_list[1:])
            else:
                developer_commit_message = ' '.join(result_list)
            dict[project_id] = developer_commit_message

    with open('../data/CommitMessage/Bugsjar_developer_commit_mssage.json', 'w+') as f:
        json.dump(dict, f)

def extract_bears():
    token = os.getenv('GITHUB_TOKEN', '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')
    g = Github(token)

    path_url = 'Project_URL/bears-bugs.json'
    with open(path_url, 'r+') as f:
        bug_info_list = json.load(f)

    cnt, cnt_error = 0, 0
    url_developer_commit = {}
    for bug_dict in bug_info_list:
        cnt += 1

        bugId = bug_dict['bugId'].split('-')[1]
        bugName = bug_dict['bugName']

        bug_url = bug_dict['commits']['fixerBuild']['url']
        sha = bug_dict['commits']['fixerBuild']['sha']
        patch = bug_dict['diff']

        # adapt for github api
        short_link = bug_url.replace('http://github.com/', '')
        short_link = short_link.split('/commit')[0]

        commit_repo = g.get_repo(short_link).get_commit(sha)
        commit_content = commit_repo.commit.message

        commit_content = commit_content.replace('\r', ' ').replace('\n', ' ')
        url_developer_commit['bears-' + bugId] = commit_content

    with open('../data/CommitMessage/Bears_developer_commit_message.json', 'w+') as f:
        json.dump(url_developer_commit, f)


if __name__ == '__main__':
    path_repo = '/Users/haoye.tian/Documents/University/project/defects4j/project_repos'
    extract_defects4j(path_repo)

    extract_bears()

    path_bugs_dot_jar = '/Users/haoye.tian/Documents/University/project/bugs-dot-jar'
    extract_bugsjar(path_bugs_dot_jar)