import io
import sys
import pandas as pd
import time
from data_util import ShowProcess, numbr, filter_punc
from constants import Output_DATA_DIR, Origin_DATA_DIR


inputfile = '%s/bugreport_patch.txt' % Output_DATA_DIR
bugreportcommitfile = open(inputfile)
lines = bugreportcommitfile.readlines()
print('Total Data: {} lines'.format(len(lines)))

column_names = ['bug_id', 'bug report text', 'bug report description', 'generated patch id', 'patch text', 'label']
# create a dataframe to store cleaned data
df = pd.DataFrame(columns = column_names)

max_steps = len(lines)
process_bar = ShowProcess(max_steps, 'preprocess finished, Okay!')

for line in lines:
    line2list = line.split('$$')
    line2list[-1] = line2list[-1].replace('\n','')
    #print(line2list)
    line2list[0] = numbr(line2list[0])
    
    #for bug report text
    line2list[1] = filter_punc(line2list[1])
    
    #for bug report description
    line2list[2] = filter_punc(line2list[2])
    
    #for generated patch id
    line2list[3] = numbr(line2list[2])    
    
    #for patch text
    line2list[4] = filter_punc(line2list[3])
    
    # add data into df file
    df_toadd = pd.DataFrame([line2list], columns = column_names)
    df = df.append(df_toadd)

    process_bar.show_process()
    time.sleep(0.01)



outfilename = '%s/bugreport_patch.csv' % Output_DATA_DIR
df.to_csv(outfilename)




