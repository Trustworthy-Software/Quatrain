import json, os
import urllib
import bs4 as bs
import pickle

def get_json_by_url(folder_path, names, lists):
    if not os.path.exists(folder_path):
        print("Selected folder not exist, try to create it.")
        os.makedirs(folder_path)
    for i in range(len(lists)):
        url = lists[i]
        filename = names[i] + '.json'
        filepath = folder_path + '/' + filename
        if os.path.exists(filepath):
            print("File have already exist. skip")
        else:
            try:
                print("Try downloading file: {}".format(url))
                urllib.urlretrieve(url, filename=filepath)
            except Exception as e:
                print("Error occurred when downloading file, error message:")
                print(e)

def get_json(file_json):
    # from url
    names = []
    lists = []
    with open(file_json, 'r+') as f:
        for line in f:
            id = line.strip().split(',')[0]
            name = 'Closure-' + id
            url = line.strip().split(',')[2]

            names.append(name)
            lists.append(url)

    get_json_by_url('Project_File/Closure_json', names, lists)

def save_bug_report(path_folder):
    result = ''
    json_list = os.listdir(path_folder)
    json_list = sorted(json_list, key=lambda x: int(x.split('.')[0].split('-')[1]),)
    for j in json_list:
        with open(os.path.join(path_folder,j), 'r+') as f:
            name = j.split('.')[0]
            json_dict = json.load(f)
            summary = str(json_dict['summary'])
            description = 'None'

            result += name + '$$' + summary + '$$' + description + '\n'
    print (result)
    with open('../data/BugReport/Closure_bugreport.txt', 'w+') as f:
        f.write(result)


def get_bug_report_closure(file_json):
    get_json(file_json)
    save_bug_report(path_folder='Project_File/Closure_json')

def get_bug_report(project):
    path_xml = 'Project_File/' + project + '_xml'
    xml_names = os.listdir(path_xml)
    xml_names = sorted(xml_names, key=lambda x: int(x.split('.')[0].split('-')[1]),)
    result = ''
    for xml_name in xml_names:
        path_file = path_xml + '/' + xml_name
        project_id = xml_name.split('.')[0]

        with open(path_file, 'rb') as f:
            the_page = pickle.load(f)
        soup = bs.BeautifulSoup(the_page, "html.parser")

        # if project == 'Time':
        #     summary = soup.find('span', {'data-snek-id': 'issue-title'}).text.strip().replace('\n',' ')
        #     description = 'None'
        # else:
        summary = soup.find('h1', {'id': 'summary-val'}).text.strip().replace('\n',' ')
        description = soup.find('div', {'id': 'description-val'}).text.strip().replace('\n',' ')
        result += project_id + '$$' + summary + '$$' + description + '\n'
    print (result)
    with open('../data/BugReport/'+project+'_bugreport.txt', 'w+') as f:
        f.write(result)

def get_bug_report_chartTime(project):
    path_json = 'Project_File/' + project + '_json'
    json_names = os.listdir(path_json)
    json_names = sorted(json_names, key=lambda x: int(x.split('.')[0].split('-')[1]),)
    result = ''
    for json_name in json_names:
        path_file = path_json + '/' + json_name
        project_id = json_name.split('.')[0]
        with open(path_file, 'r+') as f:
            dict = json.load(f)
            summary = dict['summary'].strip()
            description = dict['description'].strip().replace('\n',' ')
        result += project_id + '$$' + summary + '$$' + description + '\n'
    print (result)
    with open('../data/BugReport/'+project+'_bugreport.txt', 'w+') as f:
        f.write(result)

if __name__ == '__main__':
    # '''
    # 1. extract bug report of Closure
    get_bug_report_closure(file_json='Project_URL/Closure_url.txt')

    # 2. extract bug report of  Lang
    get_bug_report(project='Lang')

    # 3. extract bug report of Math
    get_bug_report(project='Math')

    # 4. extract bug report of Chart
    get_bug_report_chartTime(project='Chart')
    # '''
    # 3. extract bug report of Math
    get_bug_report_chartTime(project='Time')