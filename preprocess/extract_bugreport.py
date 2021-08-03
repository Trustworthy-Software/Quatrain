import json, os
import urllib

def get_json_by_url(folder_path, names, lists):
    if not os.path.exists(folder_path):
        print("Selected folder not exist, try to create it.")
        os.makedirs(folder_path)
    for i in range(len(lists)):
        url = lists[i]
        print("Try downloading file: {}".format(url))
        filename = names[i] + '.json'
        filepath = folder_path + '/' + filename
        if os.path.exists(filepath):
            print("File have already exist. skip")
        else:
            try:
                urllib.urlretrieve(url, filename=filepath)
            except Exception as e:
                print("Error occurred when downloading file, error message:")
                print(e)

def get_json(file_json):
    names = []
    lists = []
    with open(file_json, 'r+') as f:
        for line in f:
            id = line.strip().split(',')[0]
            name = 'Closure-' + id
            url = line.strip().split(',')[1]

            names.append(name)
            lists.append(url)

    get_json_by_url('Closure_json', names, lists)

def save_bug_report(path_folder):
    result = ''
    json_list = os.listdir(path_folder)
    json_list.sort()
    for j in json_list:
        with open(os.path.join(path_folder,j), 'r+') as f:
            name = j.split('.')[0]
            json_dict = json.load(f)
            summary = str(json_dict['summary'])

            result += name + ',' + summary + '\n'
    print result
    with open('Closure_bugreport.txt', 'w+') as f:
        f.write(result)


def get_bug_report(file_json):
    get_json(file_json)
    save_bug_report(path_folder='Closure_json')

if  __name__ == '__main__':
    file_json = 'Closure_url.txt'
    get_bug_report(file_json)