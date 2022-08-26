#### For complete reproduction, please go to [README.md](https://github.com/Trustworthy-Software/Quatrain#readme).

## Ⅰ) Requirements for Reproduction
### A) Environment
  * python 3.7 (Anaconda recommended)
  * ```pip install -r requirements.txt```

run `sudo apt-get install python3.7-dev` first if you don't have python3.7 dev package.

### B) Data elements 
  download _ASE2022withTextUnique.zip_ (need to be unzipped) and _ASE_features2_bert.pickle_ from [data in Zenodo](https://zenodo.org/record/6946294#.Yub3NWQzZhE "Dataset for Quatrain"), 
  accordingly change the absolute path of these two files in **experiment/config.py** of this repository as below.
  1. self.path_patch ---> ASE2022withTextUnique.  Original dataset with patches text and commit messages text.
  2. self.path_ASE2020_feature ---> ASE_features2_bert.pickle. The feature from Tian et al.'s ASE2020 [paper](https://ieeexplore.ieee.org/abstract/document/9286101) for our RQ3 DL experiment. 

### C) Test
Execute the following command to see if you can successfully obtain RQ1 results (Figure 6 & Table 2).
```
python run.py RQ1
```

## Ⅱ) Custom Prediction
To predict the correctness of your custom patches, you are welcome to use the prediction interface.
### A) Requirements
  * **BERT model client&server:** 24-layer, 1024-hidden, 16-heads, 340M parameters. download it [here](https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip).
    
  * **Environment for BERT server** (different from reproduction)
    * python 3.7 
    * pip install tensorflow==1.14
    * pip install bert-serving-client==1.10.0
    * pip install bert-serving-server==1.10.0
    * pip install protobuf==3.20.1
    * Launch BERT server via `bert-serving-start -model_dir "Path2BertModel"/wwm_cased_L-24_H-1024_A-16 -num_worker=2 -max_seq_len=360 -port 8190`
    * switch the port in [BERT_Port](https://github.com/Trustworthy-Software/Quatrain/blob/main/representation/word2vec.py#L42) in case your port 8190 is occupied.
  * **Bug report text:** developer-written bug report.
  * **Patch description text:** generating patch description for your plausible patches with commit message generation tools, e.g. CodeTrans. [Github](https://github.com/agemagician/CodeTrans) and [API](https://huggingface.co/SEBIS/code_trans_t5_large_commit_generation_transfer_learning_finetune).

### B) Predict
Let's give it a try!
```
python run.py predict $bug_report_text $patch_description_text
```
For instance: `python run.py predict 'Missing type-checks for var_args notation' 'check var_args properly'`
