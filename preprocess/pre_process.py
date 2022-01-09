import io
import sys
import pandas as pd

from data_util import ShowProcess, numbr, filter_punc, Prepare_data, writetxt2csv, splitdata
from constants import Output_DATA_DIR, Origin_DATA_DIR


if __name__ == "__main__":
    

    # output bugreport_patch.csv; correct.csv; incorrect.csv
    inputfile = '%s/bugreport_patch.txt' % Output_DATA_DIR
    outfilename = '%s/bugreport_patch.csv' % Output_DATA_DIR
    writetxt2csv(inputfile, outfilename)

    """

    correct_file = '%s/correct.csv' % Output_DATA_DIR
    incorrect_file  = '%s/incorrect.csv' % Output_DATA_DIR

    print(incorrect_file, correct_file)
    pos_sentences, neg_sentences, diction, st = splitdata(correct_file, incorrect_file)
    #print(diction)
    print("*"*50)
    st = sorted([(v[1], w) for w, v in diction.items()])
    print(st[-20:])
    """