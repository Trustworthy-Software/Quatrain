# -*- coding: utf-8 -*-
import tensorflow as tf


class QALSTM(object):

    def __init__(self, batch_size, sequence_length, embeddings, embedding_size, rnn_size, margin, attention_matrix_size):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.margin = margin
        self.attention_matrix_size = attention_matrix_size

        self.q = tf.placeholder(tf.int32, shape=[None, self.sequence_length])  # question
        self.ap = tf.placeholder(tf.int32, shape=[None, self.sequence_length])  # positive answer
        self.an = tf.placeholder(tf.int32, shape=[None, self.sequence_length])  # negative answer
        self.qtest = tf.placeholder(tf.int32, shape=[None, self.sequence_length])  # question to test
        self.atest = tf.placeholder(tf.int32, shape=[None, self.sequence_length])  # answer to test
        self.lr = tf.placeholder(tf.float32)

        with tf.name_scope("embedding_layer"):
            # Map the word index to the word embedding
            embeddings = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")
            q_embed = tf.nn.embedding_lookup(embeddings, self.q)
            ap_embed = tf.nn.embedding_lookup(embeddings, self.ap)
            an_embed = tf.nn.embedding_lookup(embeddings, self.an)
            qtest_embed = tf.nn.embedding_lookup(embeddings, self.qtest)
            atest_embed = tf.nn.embedding_lookup(embeddings, self.atest)

        with tf.variable_scope("bilstm", reuse=tf.AUTO_REUSE):
            q_lstm = self.bidirectional_lstm(q_embed, self.rnn_size)
            ap_lstm = self.bidirectional_lstm(ap_embed, self.rnn_size)
            an_lstm = self.bidirectional_lstm(an_embed, self.rnn_size)
            qtest_lstm = self.bidirectional_lstm(qtest_embed, self.rnn_size)
            atest_lstm = self.bidirectional_lstm(atest_embed, self.rnn_size)

        with tf.variable_scope("attention_encoder", reuse=tf.AUTO_REUSE):
            qp_atted, ap_atted = self.attention_encoder(q_lstm, ap_lstm)
            qn_atted, an_atted = self.attention_encoder(q_lstm, an_lstm)

        self.poscosine = self.calc_cosine(qp_atted, ap_atted)
        self.negcosine = self.calc_cosine(qn_atted, an_atted)
        self.loss, self.acc = self.calc_loss_and_acc(self.poscosine, self.negcosine)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        qtest_atted, atest_atted = self.attention_encoder(qtest_lstm, atest_lstm)
        self.scores = self.calc_cosine(qtest_atted, atest_atted)

    def bidirectional_lstm(self, x, hidden_size):
        input_x = tf.transpose(x, [1, 0, 2])
        input_x = tf.unstack(input_x)
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input_x, dtype=tf.float32)
        output = tf.stack(output)
        output = tf.transpose(output, [1, 0, 2])
        return output

    def max_pooling(self, lstm_out):
        height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])       # (step, length of input for one step)

        # do max-pooling to change the (sequence_length) tensor to 1-length tensor
        lstm_out = tf.expand_dims(lstm_out, -1)
        output = tf.nn.max_pool(
            lstm_out,
            ksize=[1, height, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID')

        output = tf.reshape(output, [-1, width])

        return output

    def attention_encoder(self, input_q, input_a):
        # h_q = int(input_q.get_shape()[1])  # length of question
        w = int(input_q.get_shape()[2])  # length of input for one step
        h_a = int(input_a.get_shape()[1])  # length of answer

        output_q = self.max_pooling(input_q)  # (b,w)

        reshape_q = tf.expand_dims(output_q, 1)  # (b,1,w)  b:batch size
        reshape_q = tf.tile(reshape_q, [1, h_a, 1])  # (b,h_a,w)
        reshape_q = tf.reshape(reshape_q, [-1, w])  # (b*h_a, w)
        reshape_a = tf.reshape(input_a, [-1, w])  # (b*h_a,w)

        Wam = tf.get_variable(initializer=tf.truncated_normal([2 * self.rnn_size, self.attention_matrix_size], stddev=0.1), name='Wam')
        Wqm = tf.get_variable(initializer=tf.truncated_normal([2 * self.rnn_size, self.attention_matrix_size], stddev=0.1), name='Wqm')
        Wms = tf.get_variable(initializer=tf.truncated_normal([self.attention_matrix_size, 1], stddev=0.1), name='Wms')

        M = tf.tanh(tf.add(tf.matmul(reshape_q, Wqm), tf.matmul(reshape_a, Wam)))
        M = tf.matmul(M, Wms)  # (b*h_a,1)

        S = tf.reshape(M, [-1, h_a])  # (b,h_a)
        S = tf.nn.softmax(S)  # (b,h_a)

        S_diag = tf.matrix_diag(S)  # (b,h_a,h_a)
        attention_a = tf.matmul(S_diag, input_a)  # (b,h_a,w)

        output_a = self.max_pooling(attention_a)  # (b,w)

        return tf.tanh(output_q), tf.tanh(output_a)

    def calc_cosine(self, q, a):
        norm_q = tf.sqrt(tf.reduce_sum(tf.multiply(q, q), 1))
        norm_a = tf.sqrt(tf.reduce_sum(tf.multiply(a, a), 1))
        mul_q_a = tf.reduce_sum(tf.multiply(q, a), 1)
        cosine = tf.div(mul_q_a, tf.multiply(norm_q, norm_a))
        return cosine

    def calc_loss_and_acc(self, poscosine, negcosine):
        # the target function
        zero = tf.fill(tf.shape(poscosine), 0.0)
        margin = tf.fill(tf.shape(poscosine), self.margin)
        with tf.name_scope("loss"):
            losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(poscosine, negcosine)))
            loss = tf.reduce_sum(losses)

        # cal accurancy
        with tf.name_scope("acc"):
            correct = tf.equal(zero, losses)
            acc = tf.reduce_mean(tf.cast(correct, "float"), name="acc")
        return loss, acc
