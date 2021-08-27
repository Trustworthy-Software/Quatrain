# -*- coding: utf-8 -*-
import random
import numpy as np


def rand(length, black_list):
    index = random.randint(1, length - 1)
    while index in black_list:
        index = random.randint(1, length - 1)
    return index


def load_embedding(filename):
    embeddings = list()
    word2idx = dict()
    print('start loading embedding')
    with open(filename, mode='r', encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split(' ')
            embedding = [float(val) for val in arr[1:len(arr)]]
            word2idx[arr[0]] = len(word2idx)
            embeddings.append(embedding)

    embedding_size = len(arr) - 1
    word2idx['UNKNOWN'] = len(word2idx)
    embeddings.append([0] * embedding_size)

    print('embedding loaded')
    return embeddings, word2idx


def load_vocab(filename):
    voc = {}
    for line in open(filename):
        word, _id = line.strip().split('\t')
        voc[word] = _id
    return voc


def load_answers(filename, voc):
    answers = ['<None>']
    for line in open(filename):
        _, sent = line.strip().split('\t')
        answers.append([voc[wid] for wid in sent.split(' ')])
    return answers


def encode_sent(sentence, word2idx, size):
    unknown = word2idx.get("UNKNOWN", 0)
    x = [unknown] * size
    for i, word in enumerate(sentence):
        if word in word2idx:
            x[i] = word2idx[word]
        else:
            x[i] = unknown
        if i >= size - 1:
            break
    return x


def load_train_data(filename, all_answers, voc, word2idx, seq_size):
    questions, pos_answers, neg_answers = [], [], []
    for line in open(filename):
        qsent, ids = line.strip().split('\t')
        ids = [int(_id) for _id in ids.split(' ')]
        question = [voc[idx] for idx in qsent.split(' ')]
        question = encode_sent(question, word2idx, seq_size)
        for _id in ids:
            for i in range(50):
                questions.append(question)
                pos_answer = encode_sent(all_answers[_id], word2idx, seq_size)
                pos_answers.append(pos_answer)
                # Negative samples are randomly sampled
                neg_answer = encode_sent(all_answers[rand(len(all_answers), ids)], word2idx, seq_size)
                neg_answers.append(neg_answer)
    return np.array(questions), np.array(pos_answers), np.array(neg_answers)


def load_test_data(filename, all_answers, voc, word2idx, seq_size):
    questions, answers, labels, qids, aids = [], [], [], [], []
    qid = 0
    for line in open(filename):
        apids, qsent, anids = line.strip().split('\t')
        apids = [int(_id) for _id in apids.split(' ')]
        anids = [int(_id) for _id in anids.split(' ')]
        question = [voc[idx] for idx in qsent.split(' ')]
        question = encode_sent(question, word2idx, seq_size)
        aid = 0
        for _id in apids:
            questions.append(question)
            answer = encode_sent(all_answers[_id], word2idx, seq_size)
            answers.append(answer)
            labels.append(1)
            qids.append(qid)
            aids.append(aid)
            aid += 1
        for _id in anids:
            questions.append(question)
            answer = encode_sent(all_answers[_id], word2idx, seq_size)
            answers.append(answer)
            labels.append(0)
            qids.append(qid)
            aids.append(aid)
            aid += 1
        qid += 1
    return np.array(questions), np.array(answers), np.array(labels), np.array(qids), np.array(aids)


def batch_iter(questions, pos_answers, neg_answers, batch_size):
    data_size = len(questions)
    batch_num = int(data_size / batch_size)
    for batch in range(batch_num):
        result_questions, result_pos_answers, result_neg_answers = [], [], []
        for i in range(batch * batch_size, min((batch + 1) * batch_size, data_size)):
            result_questions.append(questions[i])
            result_pos_answers.append(pos_answers[i])
            result_neg_answers.append(neg_answers[i])
        yield result_questions, result_pos_answers, result_neg_answers


def test_batch_iter(questions, answers, batch_size):
    data_size = len(questions)
    batch_num = int(data_size / batch_size)
    for batch in range(batch_num):
        result_questions, result_answers = [], []
        for i in range(batch * batch_size, min((batch + 1) * batch_size, data_size)):
            result_questions.append(questions[i])
            result_answers.append(answers[i])
        yield result_questions, result_answers


if __name__ == '__main__':
    embeddings, word2idx = load_embedding('vectors.nobin')
    voc = load_vocab('D:\\DataMining\\Datasets\\insuranceQA\\V1\\vocabulary')
    all_answers = load_answers('D:\\DataMining\\Datasets\\insuranceQA\\V1\\answers.label.token_idx', voc)
    # questions, pos_answers, neg_answers = load_train_data('D:\\DataMining\\Datasets\\insuranceQA\\V1\\question.train.token_idx.label', all_answers, voc, word2idx, 100)
    questions, answers, labels, qids = load_test_data('D:\\DataMining\\Datasets\\insuranceQA\\V1\\question.test1.label.token_idx.pool', all_answers, voc, word2idx, 100)
