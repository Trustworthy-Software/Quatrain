import sys
import json
import os
import pickle
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

# def bugRepoDict():
#     brDict={}
#     brDict['Lang'] ='https://issues.apache.org/jira/browse/LANG-'#810'
#     brDict['Math'] ='https://issues.apache.org/jira/browse/MATH-'#790'
#     return brDict

def download4defects4j(project):
    try:
        path_url = 'Project_URL/' + project +'_url.txt'
        with open(path_url, 'r+') as f:
            for line in f:
                id = line.split(',')[0].strip()
                url = line.split(',')[2].strip()
                downloadLink = url
                webRequest(downloadLink, project, id)
    except Exception as e:
        print(e)
        logging.error(e)
        return False

def download4bears():
    path_url = 'Project_URL/bears-bugs.json'
    with open(path_url, 'r+') as f:
        bug_info_list = json.load(f)

    cnt, cnt_error = 0, 0
    url_bug_report = {}
    for bug_dict in bug_info_list:
        cnt += 1

        bugId = bug_dict['bugId'].split('-')[1]
        bugName = bug_dict['bugName']

        bug_url = bug_dict['commits']['fixerBuild']['url']
        patch = bug_dict['diff']

        test_output = ''
        for failureDetails in bug_dict['tests']['failureDetails']:
            if 'detail' in failureDetails.keys():
                test_output += failureDetails['detail'].strip().replace('\n', ' ') + ' '
        # save failing test output
        if test_output == '':
            test_output = 'None'
        testinfo = 'Bears-' + bugId + '$$' + test_output + '\n'
        path_test_info = '../data/Bears_testinfo.txt'
        if not os.path.exists(path_test_info):
            with open('../data/Bears_testinfo.txt', 'a+') as f:
                f.write(testinfo)

        try:
            # avoid frequently request
            time.sleep(0.1)
            logging.info(bug_url)
            req = urllib.request.Request(bug_url, headers=hdr)
            response = urlopen(req)
            the_page = response.read()
        except error.HTTPError as err:
            if err.code == 404:
                print("Error: %s, reason: %s." % (err.code, err.reason))
            elif err.code == 429:
                print("Error: %s, reason: %s." % (err.code, err.reason))
            else:
                print('UNKNOWN.')
                raise

        try:
            er = ''
            bug_report_url_1, bug_report_url_2 = '', ''
            et_html = etree.HTML(the_page)

            div_atrribute_1 = et_html.xpath('//*[@id="repo-content-pjax-container"]/div[2]/div[1]/a')
            div_atrribute_2 = et_html.xpath('//*[@id="repo-content-pjax-container"]/div[2]/div[2]/pre/a')


            if div_atrribute_1 == [] and div_atrribute_2 == []:
                raise ValueError('no bug report')
            if div_atrribute_1 != []:
                bug_report_url_1 = div_atrribute_1[0].attrib['href']
                # get redirect link
                bug_report_url_1 = requests.get(bug_report_url_1, headers=hdr).url
            if div_atrribute_2 != []:
                bug_report_url_2 = div_atrribute_2[0].attrib['href']
                bug_report_url_2 = requests.get(bug_report_url_2, headers=hdr).url

            bug_report_url = ''
            if 'issues' in bug_report_url_1:
                bug_report_url = bug_report_url_1
            elif 'issues' in bug_report_url_2:
                bug_report_url = bug_report_url_2
            if bug_report_url == '':
                raise ValueError('no bug report')
            webRequest(bug_report_url, project='Bears', id=bugId)

            url_bug_report['Bears-' + bugId] = bug_report_url

        except ValueError as e:
            cnt_error += 1
            er = e
            # print(e, end=', ')
        print('cnt: {}, cnt_error: {}, bug_url: {}, {}'.format(cnt, cnt_error, bug_url, er))
    print('cnt: {}, cnt_error: {}'.format(cnt, cnt_error))
    # save url for bears
    with open('../data/bears_url_dict.json', 'w+') as f:
        json.dump(url_bug_report, f)


def webRequest(bug_report_url, project, id):
    # bugID = url.split('/')[-1:]
    url = bug_report_url + '?redirect=false'
    folder = 'Project_File/' +  project+'_xml/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    file = '-'.join([project, id]) + '.pickle.xml'
    brLocation = os.path.join(folder, file)
    if os.path.exists(brLocation):
        # print('existed...')
        pass
        # with open(brLocation, 'rb') as f:
        #     the_page = pickle.load(f)
    else:
        try:
            logging.info(url)
            req = urllib.request.Request(url, headers=hdr)
            response = urlopen(req)
            the_page = response.read()
        except error.HTTPError as err:
            if err.code == 404:
                print("Error: %s, reason: %s." % (err.code, err.reason))
            return None
        pickle.dump(the_page, open(brLocation, "wb"))

def bears_index_dict_construction():
    path_url = 'Project_URL/bears-bugs.json'
    bears_dict = {}
    bears_dict_path = '../data/bears_index_dict.json'
    with open(path_url, 'r+') as f:
        bug_info_list = json.load(f)

    for bug_dict in bug_info_list:
        bugId = bug_dict['bugId']
        bugName = bug_dict['bugName']

        bears_dict[bugId] = bugName
    with open(bears_dict_path, 'w+') as f:
        json.dump(bears_dict, f)

    # with open(bears_dict_path, 'r+') as f:
    #     a = json.load(f)



if __name__ == '__main__':
    # bears_index_dict_construction()

    # 1. Defects4J
    # projects = ['Lang', 'Math', 'Time']
    # for p in projects:
    #     download4defects4j(p)

    # 2. Bears
    download4bears()

