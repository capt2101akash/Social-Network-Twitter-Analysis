from __future__ import print_function
import os
import sys
import numpy as np
import csv
import json
import pickle
import codecs
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.backend.common import epsilon
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

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
        return - tf.reduce_sum((1/19.0) * target * tf.log(output) + (1-target)*tf.log(1.0-output),axis=len(output.get_shape()) - 1)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)



BASE_DIR = ''
GLOVE_DIR = BASE_DIR + 'glove.twitter.27B/'
TEXT_DATA_DIR = BASE_DIR + 'hate-speech-and-offensive-language-master/data/'
MAX_SEQUENCE_LENGTH = 120
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

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
f = csv.reader(open(os.path.join(BASE_DIR, 'clean_data.csv')))
for line in f:
    labels.append(line[2])
    texts.append(line[0])
print("hate speech",len(labels))

print('Found %s texts.' % len(texts))

# vectorize the text samples into a 2D integer vectorize
num_train_data_1 = int(0.8 * len(hate_text))
num_train_data_2 = int(0.8 * len(non_hate_text))
train_list1 = hate_text[:num_train_data_1]
train_list1_label = []
train_list2_label = []
for i in range(0, num_train_data_1):
    train_list1_label.append(1)
for i in range(0, num_train_data_2):
    train_list2_label.append(0)

train_list2 = non_hate_text[:num_train_data_2]

train_data_final = train_list1 + train_list2
train_data_label = train_list1_label + train_list2_label

test_list1 = hate_text[num_train_data_1:]
test_list2 = non_hate_text[num_train_data_2:]
test_list1_label = []
test_list2_label = []
for i in range(0, len(hate_text) - num_train_data_1):
    test_list1_label.append(1)
for i in range(0, len(non_hate_text) - num_train_data_2):
    test_list2_label.append(0)
test_data_final = test_list1 + test_list2
test_data_label = test_list1_label + test_list2_label
count_one = 0
count_zero = 0
num_valid_data = int(0.2 * len(train_data_final))
for i in range(len(train_data_final)):
    if train_data_label[i] == 1:
        count_one +=1
    else:
        count_zero += 1
print('1s : ', count_one)
print('0s : ', count_zero)
num_valid_hate = int(0.2 * count_one)
num_valid_non_hate = int(0.2 * count_zero)

valid_data_list1 = train_data_final[:num_valid_hate]
valid_data_list2 = train_data_final[count_one:num_valid_non_hate]
valid_data_final = valid_data_list1 + valid_data_list2

valid_data_list1_label = train_data_label[:num_valid_hate]
valid_data_list2_label = train_data_label[count_one:num_valid_non_hate]
valid_data_label = valid_data_list1_label + valid_data_list2_label

train_data_final_list1 = train_data_final[num_valid_hate:count_one]
train_data_final_list2 = train_data_final[num_valid_non_hate:]
train_data_final = train_data_final_list1 + train_data_final_list2

train_data_label_list1 = train_data_label[num_valid_hate:count_one]
train_data_label_list2 = train_data_label[num_valid_non_hate:]
train_data_label = train_data_label_list1 + train_data_label_list2

print('valid data: ', len(valid_data_final))
print('valid label size: ', len(valid_data_label))

print('train label size: ',len(train_data_label))
print('train data size: ', len(train_data_final))

print('test data size: ', len(test_data_final))
print('test label size: ', len(test_data_label))

tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
train_sequences = tokenizer.texts_to_sequences(train_data_final)
valid_sequences = tokenizer.texts_to_sequences(valid_data_final)
test_sequences = tokenizer.texts_to_sequences(test_data_final)
word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

# pad the sequences to a threshold
data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH)
train_data = pad_sequences(train_sequences, maxlen = MAX_SEQUENCE_LENGTH)
valid_data = pad_sequences(valid_sequences, maxlen = MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen = MAX_SEQUENCE_LENGTH)

train_data_label = to_categorical(np.asarray(train_data_label))
test_data_label = to_categorical(np.asarray(test_data_label))
valid_data_label = to_categorical(np.asarray(valid_data_label))

print('Shape of train_data tensor:', train_data.shape)
print('Shape of valid_data tensor:', valid_data.shape)
print('Shape of test_data tensor:', test_data.shape)
#split data into a training and validation set
#print (data.shape[0])



train_indices = np.arange(train_data.shape[0])
np.random.shuffle(train_indices)
train_data = train_data[train_indices]
train_data_label = train_data_label[train_indices]

test_indices = np.arange(test_data.shape[0])
np.random.shuffle(test_indices)
test_data = test_data[test_indices]
test_data_label = test_data_label[test_indices]

valid_indices = np.arange(valid_data.shape[0])
np.random.shuffle(valid_indices)
valid_data = valid_data[valid_indices]
valid_data_label = valid_data_label[valid_indices]

#
# hate_data_indices = np.arange(hate_data.shape[0])
# np.random.shuffle(hate_data_indices)
# hate_data = hate_data[indices]
#
# num_test_samples = int(0.2 * hate_data.shape[0])
#
# x_train = data[:-num_test_samples]
# y_train = labels[:-num_test_samples]
# x_test = data[-num_test_samples:]
# y_test = labels[-num_test_samples:]
#
# num_valid_samples = int(VALIDATION_SPLIT * x_train.shape[0])
#
# x_val = x_train[:-num_valid_samples]
# y_val = y_train[:-num_valid_samples]
# x_train = x_train[-num_valid_samples:]
# y_train = y_train[-num_valid_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
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
                            trainable=True)

print('Training model.')

#Training

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
print (embedded_sequences)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)

x = MaxPooling1D(3)(x)

x = Conv1D(128, 5, activation='relu')(x)

x = MaxPooling1D(3)(x)

x = Conv1D(128, 5, activation='relu')(x)

x = MaxPooling1D(3)(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)

model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',
optimizer='rmsprop',
metrics=['acc'])

def convert(sentence, tokenizer):
    sequences = tokenizer.texts_to_sequences([sentence])
    data = pad_sequences(sequences, maxlen = MAX_SEQUENCE_LENGTH)
    return data

model.fit(train_data, train_data_label, batch_size=128, epochs=5, validation_data=(valid_data, valid_data_label))
print(model.evaluate(test_data, test_data_label))
print(model.predict(convert("I love you!!", tokenizer)))
# serialize model to JSON
model_json = model.to_json()
with open("modelv1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelv1.h5")
print("Saved model to disk")

print("Saving tokenizer")
pickle.dump(tokenizer, open("tokenizerv1.pkl","wb"))
