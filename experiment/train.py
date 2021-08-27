# -*- coding: utf-8 -*-
import time
import numpy as np
import data_helpers
from qalstm import QALSTM
import tensorflow as tf
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import pickle

def main():
    trained_model = "checkpoints/model.ckpt"
    embedding_size = 100  # Word embedding dimension
    epochs = 5
    batch_size = 64  # Batch data size
    rnn_size = 50  # Number of hidden layer neurons
    sequence_length = 300  # Sentence length
    learning_rate = 0.01  # Learning rate
    lrdownRate = 0.9
    margin = 0.1
    attention_matrix_size = 100
    gpu_mem_usage = 0.75
    gpu_device = "/gpu:0"
    cpu_device = "/cpu:0"

    # embeddings, word2idx = data_helpers.load_embedding('vectors.nobin')
    # voc = data_helpers.load_vocab('/Users/haoye.tian/Documents/University/data/insuranceQA/V1/vocabulary')
    # all_answers = data_helpers.load_answers('/Users/haoye.tian/Documents/University/data/insuranceQA/V1/answers.label.token_idx', voc)
    # questions, pos_answers, neg_answers = data_helpers.load_train_data('/Users/haoye.tian/Documents/University/data/insuranceQA/V1/question.train.token_idx.label', all_answers, voc, word2idx, sequence_length)
    # data_size = len(questions)
    # permutation = np.random.permutation(data_size)
    # questions = questions[permutation, :]
    # pos_answers = pos_answers[permutation, :]
    # neg_answers = neg_answers[permutation, :]

    # tian
    embedding_method = 'bert'
    dataset = pickle.load(open('../data/bugreport_commit_' + embedding_method + '.pickle', 'rb'))
    questions = np.array(dataset[0]).reshape((len(dataset[0]), -1))
    answers = np.array(dataset[1]).reshape((len(dataset[1]), -1))
    data_size = len(questions)

    with tf.Graph().as_default(), tf.device(gpu_device):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_usage)
        session_conf = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        model = QALSTM(batch_size, sequence_length, embeddings, embedding_size, rnn_size, margin, attention_matrix_size)
        with tf.Session(config=session_conf).as_default() as sess:  # config=session_conf
            saver = tf.train.Saver()

            print("Start training")
            sess.run(tf.global_variables_initializer())  # Initialize all variables
            for epoch in range(epochs):
                print("The training of the %s iteration is underway" % (epoch + 1))
                batch_number = 1
                for question, pos_answer, neg_answer in data_helpers.batch_iter(questions, pos_answers, neg_answers, batch_size):
                    start_time = time.time()
                    feed_dict = {
                        model.q: question,
                        model.ap: pos_answer,
                        model.an: neg_answer,
                        model.lr: learning_rate
                    }
                    _, loss, acc = sess.run([model.train_op, model.loss, model.acc], feed_dict)
                    duration = time.time() - start_time
                    print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tAcc %2.3f' % (epoch + 1, batch_number * batch_size, data_size, duration, loss, acc))
                    batch_number += 1
                learning_rate *= lrdownRate
                saver.save(sess, trained_model)
            print("End of the training")


if __name__ == '__main__':
    main()
