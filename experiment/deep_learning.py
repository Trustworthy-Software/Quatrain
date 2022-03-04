import random
import numpy as np
import keras
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from gensim.models import Word2Vec
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dot, Dropout, Reshape, Flatten, Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential
import tensorflow as tf


from sklearn import preprocessing
from keras import models
from keras.layers import MaxPool1D, Activation, Dense, Flatten, Input, Multiply, Permute, RepeatVector, Reshape, Concatenate, Conv1D, LSTM, Dot
# from keras.utils import plot_model
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score


def get_dnn(dimension):
    input_embeddings_tensor = Input(shape=(dimension,))
    embeddings_tensor = Dense(512, activation='tanh')(input_embeddings_tensor)  # 100为神经元
    # for _ in range(3):   # DNN层数，该为3层
    embeddings_tensor = Dense(128, activation='tanh')(embeddings_tensor)
    embeddings_tensor = Dense(128, activation='tanh')(embeddings_tensor)
    # embeddings_tensor = Dense(64, activation='tanh')(embeddings_tensor)
    output_tensor = Dense(1, activation='sigmoid')(embeddings_tensor)

    model = models.Model(inputs=input_embeddings_tensor, outputs=output_tensor)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

    # plot_model(model, to_file='./model.png', show_shapes=True)
    return model

def get_dnn_4engineered(dimension):
    input_engineered_tensor = Input(shape=(dimension,))
    engineered_tensor = Dense(128, activation='tanh')(input_engineered_tensor)
    engineered_tensor = Dense(64, activation='tanh')(engineered_tensor)
    # engineered_tensor = Dense(1024, activation='tanh')(engineered_tensor)

    # engineered_tensor = Dense(512, activation='sigmoid')(engineered_tensor)
    output_tensor = Dense(1, activation='sigmoid')(engineered_tensor)

    model = models.Model(inputs=input_engineered_tensor, outputs=output_tensor)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

    # plot_model(model, to_file='./model.png', show_shapes=True)
    return model

def get_cnn():
    input_engineered_tensor = Input(shape=(4497,))
    engineered_tensor = Reshape((4497, 1))(input_engineered_tensor)

    engineered_tensor = Conv1D(256, (50), activation='relu')(engineered_tensor)
    engineered_tensor = MaxPool1D((8))(engineered_tensor)
    # engineered_tensor = Conv1D(128, (50), activation='relu')(engineered_tensor)
    # engineered_tensor = MaxPool1D((8))(engineered_tensor)

    engineered_tensor = Flatten()(engineered_tensor)
    output_tensor = Dense(1, activation='sigmoid')(engineered_tensor)

    model = models.Model(inputs=input_engineered_tensor, outputs=output_tensor)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

    # plot_model(model, to_file='./model.png', show_shapes=True)
    return model

def get_wide_deep(dimension_learned, dimension_engineered):
    input_embeddings_tensor = Input(shape=(dimension_learned,))
    embeddings_tensor = Dense(512, activation='tanh')(input_embeddings_tensor)  # 100为神经元
    # for _ in range(3):   # DNN层数，该为3层
    embeddings_tensor = Dense(128, activation='tanh')(embeddings_tensor)
    embeddings_tensor = Dense(128, activation='tanh')(embeddings_tensor)
    # embeddings_tensor = Dense(64, activation='tanh')(embeddings_tensor)

    input_fe_tensor = Input(shape=(dimension_engineered,))
    # engineered_tensor = Dense(1, activation='sigmoid')(input_fe_tensor)

    concat_tensor = Concatenate()([embeddings_tensor, input_fe_tensor])
    output_tensor = Dense(1, activation='sigmoid')(concat_tensor)
    model = models.Model(inputs=[input_embeddings_tensor, input_fe_tensor], outputs=output_tensor)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

    # plot_model(model, to_file='../model/wide_deep.png', show_shapes=True)
    tf.keras.utils.plot_model(model, to_file="my_model.png", show_shapes=True)

    return model

def get_rnn_qa(dimension_bug_report, dimension_commit_message):
    seq_maxlen = 360
    vocab_size = dimension_bug_report
    inputs = Input(shape=(seq_maxlen, vocab_size))
    encoded = LSTM(EMBED_SIZE)(inputs)
    decoded = RepeatVector(seq_maxlen)(encoded)
    decoded = LSTM(vocab_size, return_sequences=True)(decoded)
    autoencoder = Model(inputs, decoded)

    autoencoder.compile("adadelta", loss="binary_crossentropy")

    autoencoder.fit(Xstrain, Xstrain, nb_epoch=NBR_EPOCHS, batch_size=BATCH_SIZE,
                    shuffle=True, validation_data=(Xstest, Xstest))

    # plot_model(model, to_file='../model/wide_deep.png', show_shapes=True)

    return model

def get_qa_attention(dimension_bug_report, dimension_commit_message):
    seq_maxlen = dimension_bug_report[0]
    QA_EMBED_SIZE = dimension_bug_report[1]
    # question
    qenc = Sequential()
    # qenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,input_length=seq_maxlen,weights=[embedding_weights]))
    qenc.add(Bidirectional(LSTM(QA_EMBED_SIZE, return_sequences=True), merge_mode="sum", input_shape=dimension_bug_report))
    # qenc.add(Dropout(0.3))
    # qenc.add(Convolution1D(QA_EMBED_SIZE // 2, 5, padding="valid"))
    # qenc.add(MaxPooling1D(pool_size=2, padding="valid"))
    # qenc.add(Dropout(0.3))

    # answer
    aenc = Sequential()
    aenc.add(Bidirectional(LSTM(QA_EMBED_SIZE, return_sequences=True),merge_mode="sum",input_shape=dimension_commit_message))
    # aenc.add(Dropout(0.3))
    # aenc.add(Convolution1D(QA_EMBED_SIZE // 2, 5, padding="valid"))
    # aenc.add(MaxPooling1D(pool_size=2, padding="valid"))
    # aenc.add(Dropout(0.3))

    # attention
    attOut = Dot(axes=1, normalize=True)([qenc.output, aenc.output])
    attOut = Flatten()(attOut)  # shape is now only (samples,)
    attOut = Dense((qenc.output_shape[1] * (dimension_bug_report[1] )))(attOut)
    attOut = Reshape((qenc.output_shape[1], dimension_bug_report[1] ))(attOut)

    flatAttOut = Flatten()(attOut)
    flatQencOut = Flatten()(qenc.output)

    similarity = Dot(axes=1, normalize=True)([flatQencOut, flatAttOut])
    concat_tensor = Concatenate()([Flatten()(qenc.output), Flatten()(aenc.output), similarity])
    # concat_tensor = Concatenate()([similarity])

    output_tensor = Dense(1, activation='sigmoid')(concat_tensor)


    model = models.Model([qenc.input, aenc.input], output_tensor)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])
    # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    tf.keras.utils.plot_model(model, to_file="my_model.png", show_shapes=True)


    # # attention model
    # attn = Sequential()
    # attn.add(Merge([qenc, aenc], mode="dot", dot_axes=[1, 1]))
    # attn.add(Flatten())
    # attn.add(Dense((seq_maxlen * dimension_bug_report)))
    # attn.add(Reshape((seq_maxlen, dimension_bug_report)))
    #
    # model = Sequential()
    # model.add(Merge([qenc, attn], mode="sum"))
    # model.add(Flatten())
    # model.add(Dense(2, activation="softmax"))
    #
    # model.compile(optimizer="adam", loss="categorical_crossentropy",
    #               metrics=["accuracy"])
    #
    # print("Training...")
    return model

def balanceLoss(yTrue,yPred):
    import keras.backend as K
    loss = K.binary_crossentropy(yTrue,yPred)
    scaledTrue = (2*yTrue) + 1
        #true values are 3 times worth the false values
        #contains 3 for true and 1 for false

    return scaledTrue * loss



def get_dnn_dnn(dimension_learned, dimension_engineered):
    input_embeddings_tensor = Input(shape=(dimension_learned,))
    embeddings_tensor = Dense(512, activation='tanh')(input_embeddings_tensor)  # 100为神经元
    # for _ in range(3):   # DNN层数，该为3层
    embeddings_tensor = Dense(128, activation='tanh')(embeddings_tensor)
    embeddings_tensor = Dense(128, activation='tanh')(embeddings_tensor)

    # embeddings_tensor = Dense(1, activation='sigmoid')(embeddings_tensor)

    input_engineered_tensor = Input(shape=(dimension_engineered,))
    engineered_tensor = Dense(128, activation='tanh')(input_engineered_tensor)
    engineered_tensor = Dense(64, activation='tanh')(engineered_tensor)
    # engineered_tensor = Dense(512, activation='tanh')(engineered_tensor)

    # engineered_tensor = Dense(128, activation='sigmoid')(engineered_tensor)

    concat_tensor = Concatenate()([embeddings_tensor, engineered_tensor])
    output_tensor = Dense(1, activation='sigmoid')(concat_tensor)
    model = models.Model(inputs=[input_embeddings_tensor, input_engineered_tensor], outputs=output_tensor)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])

    # plot_model(model, to_file='../model/dnn_dnn.png', show_shapes=True)

    return model
