import os
import json

with open('./Defects4j_developer_commit_mssage.json') as f1, open('./Bugsjar_developer_commit_mssage.json') as f2, open('./Bears_developer_commit_message.json') as f3:
    first_list = json.load(f1)
    second_list = json.load(f2)
    third_list = json.load(f3)

# for i, v in enumerate(first_list):
#     third_list[i].update(v)
#
# for i, v in enumerate(second_list):
#     third_list[i].update(v)

first_list.update(second_list)
first_list.update(third_list)

with open('./Developer_commit_message.json', 'w+') as f:
    json.dump(first_list, f)