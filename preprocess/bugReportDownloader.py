import os
import pickle

from urllib.request import urlopen
from urllib import error
import urllib
import logging
import socket

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

def downloadAll(project):
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

def webRequest(x, project, id):
    url = x
    # bugID = url.split('/')[-1:]
    url = url + '?redirect=false'

    brLocation = os.path.join('Project_File/', project+'_xml/', '-'.join([project, id]) + '.pickle.xml')
    if os.path.exists(brLocation):
        print('existed...')
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

if __name__ == '__main__':
    downloadAll('Lang')
    downloadAll('Math')
