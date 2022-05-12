import re

import pandas

from artifact_detection_model.constants import TARGET_NAMES
from artifact_detection_model.regex_cleanup import regex_cleanup, split_by_md_code_block
from artifact_detection_model.utils.Logger import Logger
from artifact_detection_model.utils.paths import TRAINING_SET_BUG_REPORTS_PATH, TEST_SET_BUG_REPORTS_PATH, \
    MD_DOCUMENTATION_PATH

log = Logger()


def get_data_from_issues(df, regex_clean=True):
    print(df.shape)
    df = df[df['body'].str.contains("```", na=False)]
    print(df.shape)
    docs = df['title'] + '\n' + df['body']
    documents = docs.tolist()

    artifacts, text = split_by_md_code_block(documents)

    if regex_clean:
        art, text = regex_cleanup(text)
        artifacts.extend(art)

    return artifacts, text


def get_data_from_documentation(regex_clean=True):
    df = pandas.read_csv(MD_DOCUMENTATION_PATH, compression='zip')
    df = df[~df['doc'].isnull()]
    df['doc'] = df['doc'].astype(str)
    documents = df.pop('doc').values

    artifacts, text = split_by_md_code_block(documents)

    if regex_clean:
        art, text = regex_cleanup(text)
        artifacts.extend(art)

    return artifacts, text


def get_training_and_test_set(bug_tickets=True, documentation=True, regex_clean=True):
    artifacts = []
    texts = []

    if documentation:
        doc_artifacts, doc_texts = get_data_from_documentation(regex_clean=regex_clean)
        artifacts.extend(doc_artifacts)
        texts.extend(doc_texts)

    valid_commit_set_df = pandas.read_csv(TEST_SET_BUG_REPORTS_PATH, compression='zip')
    valid_commit_set_df['body'] = valid_commit_set_df['body'].astype(str)
    valid_commit_set_df['title'] = valid_commit_set_df['title'].astype(str)
    invalid_commit_set_df = pandas.read_csv(TRAINING_SET_BUG_REPORTS_PATH, compression='zip')
    invalid_commit_set_df['body'] = invalid_commit_set_df['body'].astype(str)
    invalid_commit_set_df['title'] = invalid_commit_set_df['title'].astype(str)

    if bug_tickets:
        iss_artifacts, iss_texts = get_data_from_issues(invalid_commit_set_df, regex_clean=regex_clean)
        artifacts.extend(iss_artifacts)
        texts.extend(iss_texts)

    validation_artifacts, validation_texts = get_data_from_issues(valid_commit_set_df)

    print("train artifacts len = " + str(len(artifacts)))
    print("train text len = " + str(len(texts)))

    print("test artifacts len = " + str(len(validation_artifacts)))
    print("test text len = " + str(len(validation_texts)))

    df_tex = pandas.DataFrame({'doc': texts})
    df_tex['target'] = TARGET_NAMES['text']
    df_art = pandas.DataFrame({'doc': artifacts})
    df_art['target'] = TARGET_NAMES['artifact']
    df_train = df_tex.append(df_art.sample(len(df_tex), random_state=42))

    df_tart = pandas.DataFrame({'doc': validation_artifacts})
    df_tart['target'] = TARGET_NAMES['artifact']
    df_ttex = pandas.DataFrame({'doc': validation_texts})
    df_ttex['target'] = TARGET_NAMES['text']
    df_test = df_ttex.append(df_tart.sample(len(df_ttex), random_state=42))
    df_test = df_test.sample(frac=1, random_state=42)

    return df_train, df_test


def get_manual_validation_data_set(path):
    df = pandas.read_csv(path, compression='zip')

    df = df[~df['target'].isnull()]
    df = df[~df['doc'].isnull()]

    target = df.pop('target').values
    data = df.pop('doc').values
    return data, target


def get_nlon_dataset(path, reviewer, balance=False):
    df = pandas.read_csv(path)
    if reviewer == 'agreement':
        df = df[df['Disagreement'] == False]
        reviewer = 'Mika'
    df = df[~df[reviewer].isnull()]
    df['target'] = df[reviewer].replace(2, 0)
    log.s(df['target'].value_counts().to_string())

    if balance:
        df_art = df[df['target'] == 0].copy()
        df_tex = df[df['target'] == 1].copy()
        df = df_art.append(df_tex.sample(len(df_art), random_state=42))

    target = df.pop('target').values
    data = df.pop('Text').values

    out_data = []
    fup = re.compile(r"^([\s>:\-])+") # because they do this aswell https://github.com/M3SOulu/NLoN/blob/2889f7419ddb43f075af0cb5d679981e1d7b5c66/data-raw/make_data.R#L13
    for dat in data:
        out_data.append(fup.sub('', dat))
    return out_data, target
