[![DOI](https://zenodo.org/badge/392095984.svg)](https://zenodo.org/badge/latestdoi/392095984)

Quatrain
=======
Quatrain (Question Answering for Patch Correctness Evaluation), a supervised learning approach that exploits a deep NLP model to classify the
relatedness of a bug report with a patch description.
```bibtex
@article{tian2021is,
  title={Is this Change the Answer to that Problem? Correlating Descriptions of Bug and Code Changes for Evaluating Patch Correctness},
}
```
### Catalogue of Repository
```
artifact_detection_model: a model to detect codes in text.
data: processed and structured dataset.
experiment: scripts to obtain experimental results of paper. 
figure: saved figures for experiment
preprocess: scripts to extract bug reports and commit messages.
representation: embeddings representation model.
utils: scripts to deduplicate dataset.
---------------
requirements.txt: required dependencies.
run.py: entrance to conduct experiment.
```

## Ⅰ) Requirements
### A) Environment 
run `sudo apt-get install python3.7-dev` first if you don't have python3.7 dev package.
  * python 3.7
  * pip install -r requirements.txt

### B) Data elements 
  download _ASE2022withTextUnique.zip_ (need to be unzipped) and _ASE_features2_bert.pickle_ from [Zenodo](https://zenodo.org/record/6946294#.Yub3NWQzZhE "Dataset for Quatrain"), 
  accordingly change the absolute path of files in **experiment/config.py** of this repository as below.
  1. self.path_patch ---> ASE2022withTextUnique.
  2. self.path_ASE2020_feature ---> ASE_features2_bert.pickle.

## Ⅱ) Experiment

To obtain the experimental results of our paper, execute `run.py` with the following parameters:

### A) Sec. 2.2 (Hypothesis validation)
  1. **Figure 3:** Distributions of Euclidean distances between bug and patch descriptions.
```
python run.py hypothesis
```

### B) Sec. 5.1 (RQ1: Effectiveness of Quatrain) 
  1. **Figure 6:** Distribution of Patches in Train and Test Data. 
  2. **Table 2.:** Confusion matrix of Quatrain prediction.
```
python run.py RQ1
```
  3. **The improved F1:** F1 score by re-balancing the test data.
```
python run.py RQ1 balance
```  

### C) Sec. 5.2 (RQ2: Analysis of the Impact of Input Quality on Quatrain)
#### RQ 2.1
  1. **Figure 7:**  Impact of length of patch description to prediction.
```
python run.py RQ2.1
```
#### RQ 2.2
  2. **Figure 8:**  The distribution of probability of patch correctness
on original and random bug report.
  3. **The dropped +Recall:**  22% (241/1073) of developer patches, which were previously predicted as correct, are no longer recalled by Quatrain after they have been associated to a random bug report.
```
python run.py RQ2.2
```
#### RQ 2.3
  4. **Figure 9:**   Impact of distance between generated patch descrip-
tion to ground truth on prediction performance.
  5. **The dropped +Recall:**  The metric (+Recall) drops by 37 percentage points to 45\% when the developer-written descriptions are replaced with CodeTrans-generated descriptions.
```
python run.py RQ2.3
```
  6. **The dropped AUC:**  we evaluated Quatrain in a setting where all developer commit messages were replaced with generated descriptions: the AUC metric dropped by 11 percentage points to 0.774, confirming our findings.
```
python run.py RQ2.3 generate
```

### D) Sec. 5.3 (RQ3: Comparison Against the State of the Art)
#### Sec. 5.3.1 (Comparing against Static Approaches)
  1. **Table 3:** Quatrain vs a DL-based patch classifie.
  2. **New identification:**  Among 9135 patches, our approach identifies 7842 patches, of which 2735 patches cannot be identified by Tian et al.'s approach (RF).
```
python run.py RQ3 DL
```
  3. **Table 4:** Quatrain vs BATS.
  4. **New identification:**  180 out of 345 patches are exclusively identified by Quatrain.
```
python run.py RQ3 BATS
```

#### Sec. 5.3.2 (Comparing against Dynamic Approach)
  1. **Table 5:** Quatrain vs (execution-based) PATCH-SIM.
  2. **New identification:**  Most of the patches (1856/3149) that we identify are not correctly predicted by PATCH-SIM.
```
python run.py RQ3 PATCHSIM
```

### D) Sec. 6.1 (Experimental insights)
  1. **RF with 10-fold:** RandomForest (RF) on the embeddings of the bug report and the patch based on 10-fold cross validation.
  2. **RF with 10-group:** RandomForest (RF) on the embeddings of the bug report and the patch based on 10-group cross validation.
```
python run.py insights
```

## Ⅲ) Dataset
  1. bug report summary: title for bug issue.
  2. bug report description: detailed description for bug issue.
  3. patch description: CodeTrans-generated commit message for patch.

### A) Table 1: Datasets of labelled patches.
* **bugreport_patch.txt:** 9135 (1591:7544) Pairs of Bug report & Commit message. Structured as `bug-id $$ bug report summary $$ bug report description $$ patchId $$ patch description $$ label`
* **bugreport_patch_json_bert.pickle:** Bert embeddings of Pairs of Bug report & Commit message.

### B) Colleted elements
* **BugReport:** Bug reports for Defects4j, Bugsjar, Bears. Structured as `bug-id $$ bug report summary $$ bug report description`.
* **CommitMessage:** Commit messages written by developer or generated by CodeTrans. Structured as `bug-id: commit message` in json file.

[//]: # (* **Patches.zip:** collected patches for Defects4J, Bears, Bugs.jar)

## License

Quatrain is distributed under the terms of the MIT License, see [LICENSE](LICENSE).

[//]: # (### deduplicate.py)
[//]: # (deduplicating same patches.)