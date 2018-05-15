import sys
import os
import numpy as np
import csv
import json
import pickle
import codecs
import tensorflow as tf
import collections
import h5py
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

tokenizer = Tokenizer(num_words = 20000)

def convert(sentence, tokenizer):
    #print(sentence)
    #tokenizer.fit_on_texts(sentence)
    sequences = tokenizer.texts_to_sequences([sentence])
    #print(sequences)
    data = pad_sequences(sequences, maxlen = 120)
    return data

BASE_DIR = ''
model = load_model("modelv1.h5")


f = csv.reader(open(os.path.join(BASE_DIR, 'test_data.csv')))
texts = []
labels = []
neither = 0
sexism = 0
racism = 0
neither_pred = 0
sexism_pred = 0
racism_pred = 0
for line in f:
    texts.append(line[1])
    labels.append(line[2])
    if line[2] == 'neither':
        neither+=1
    elif line[2] == 'sexism':
        sexism+=1
    elif line[2] == 'racism':
        racism+=1
tokenizer.fit_on_texts(texts)
#sequences = tokenizer.texts_to_sequences(texts)
#word_index = tokenizer.word_index

index = 0
for tweets in texts:
    #print("Pr                       Actual\n")
    if model.predict(convert(tweets, tokenizer)).argmax() == 0:
        if labels[index] == 'neither':
            neither_pred += 1
    elif model.predict(convert(tweets, tokenizer)).argmax() == 1:
        if labels[index] == 'sexism':
            sexism_pred += 1
    elif model.predict(convert(tweets, tokenizer)).argmax() == 2:
        if labels[index] == 'racism':
            racism_pred += 1
    #print(model.predict(convert(tweets, tokenizer)).argmax(), labels[index], '\n')
    index+=1

print(neither, neither_pred)
print(sexism, sexism_pred)
print(racism, racism_pred)
