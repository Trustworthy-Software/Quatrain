# preprocess
1, collect dataset
2, split patch
3, format folders
4, get source target file

## data_util.py
functions:
- def writetxt2csv( , ): transform file from text to csv
- def Prepare_data( , , is_filter = Ture): Scan data and build vocabulary
- Class ShoProcess(): Process bar function
- def numbr(string): replace '-' to none
- def filter_punc(sentence): remove punctuation
- def word2index(word, diction): given a word, return its index
- def index2word(index, diction): given an index, return its word
- def splitdata( , ): split data and build dictionary
