from __future__ import print_function
import os
import sys
import numpy as np
import csv
import json
import pickle
import codecs
import tensorflow as tf
import collections
import h5py
from time import time
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.backend.common import epsilon
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import TensorBoard

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + 'glove.twitter.27B/'
TEXT_DATA_DIR = BASE_DIR + 'hate-speech-and-offensive-language-master/data/'
MAX_SEQUENCE_LENGTH = 120
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def categorical_crossentropy(target, output, from_logits=False):
    """Categorical crossentropy between an output tensor and a target tensor.
    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
    # Returns
        Output tensor.
    """
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output,
                                axis=len(output.get_shape()) - 1,
                                keep_dims=True)
        # manual computation of crossentropy
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return -tf.reduce_sum(0.11 * target * tf.log(output) + (1-target)*tf.log(1.0-output),axis=len(output.get_shape()) - 1)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)


print('Indexing word vectors.')

embeddings_index = {}

# first we will create a map from words to vector
f = open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype = 'float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# Prepare the data and their labels.

texts = [] #list of text data
labels_index = {} #dictionary mapping label name to id
labels = [] #list of label ids
hate_text = [] #list of hate texts
non_hate_text = [] #list of non-hate text
f = csv.reader(open(os.path.join(BASE_DIR, 'clean_data_1.csv')))
for line in f:
    if line[2] == 'neither':
        labels.append(0)
        texts.append(line[1])
    elif line[2] == 'sexism':
        labels.append(1)
        # print(line[1])
        texts.append(line[1])
    elif line[2] == 'racism':
        labels.append(2)
        texts.append(line[1])

freq = collections.Counter()
freq.update(labels)
print(freq)
print("hate speech",len(labels))
print('Found %s texts.' % len(texts))

tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

# pad the sequences to a threshold
data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH)
# train_data = pad_sequences(train_sequences, maxlen = MAX_SEQUENCE_LENGTH)
# valid_data = pad_sequences(valid_sequences, maxlen = MAX_SEQUENCE_LENGTH)
# test_data = pad_sequences(test_sequences, maxlen = MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_valid_samples = int(0.2 * data.shape[0])

x_temp_train = data[ : -num_valid_samples]
y_temp_train = labels[ : -num_valid_samples]

num_testing_samples = int(0.2 * x_temp_train.shape[0])
x_train = x_temp_train[ : -num_testing_samples]
y_train = y_temp_train[ : -num_testing_samples]

'''
x_test = x_temp_train[-num_testing_samples : ]
y_test = y_temp_train[-num_testing_samples : ]

print(x_test, y_test)
'''

x_valid = data[-num_valid_samples : ]
y_valid = labels[-num_valid_samples : ]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding matrix will be all zeros
        embedding_matrix[i] = embedding_vector

# load pre trained word embeddings into an embedding layers
print (embedding_matrix.shape)
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights = [embedding_matrix],
                            input_length = MAX_SEQUENCE_LENGTH,
                            trainable=False)


print('Training model.')

#Training

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
print('sequence size = ', (embedded_sequences.shape))
x = Conv1D(120, 5, activation='elu')(embedded_sequences)
print('shape of x1', x.shape)
x = MaxPooling1D(3)(x)
print('shape of x_pool1', x.shape)
x = Conv1D(120, 5, activation='elu')(x)
print('shape of x2', x.shape)
x = MaxPooling1D(3)(x)
x = Flatten()(x)
print('shape of x_flatten', x.shape)
x = Dense(120, activation='elu')(x)
print('shape of x_dense', x.shape)
preds = Dense(3, activation='softmax')(x)

model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',
optimizer='rmsprop',
metrics=['acc'])
scores = []
def convert(sentence, tokenizer):
    sequences = tokenizer.texts_to_sequences([sentence])
    data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH)
    return data

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit(x_train, y_train, batch_size=128, epochs=8, validation_data=(x_valid, y_valid), callbacks=[tensorboard])
print(model.summary())

# scores1 = model.evaluate(x_test, y_test, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores1[1]*100))
#scores.append(scores1[1] * 100)
# print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))
# for num in range(x_test.shape[0]):
#
#     print(model.predict(convert(x_test[num], tokenizer)))
# serialize model to JSON
model_json = model.to_json()
with open("modelv1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save("modelv1.h5")
print("Saved model to disk")

print("Saving tokenizer")
pickle.dump(tokenizer, open("tokenizerv1.pkl","wb"))
