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
csv.field_size_limit(sys.maxsize)
tokenizer = Tokenizer(num_words = 20000)

def convert(sentence, tokenizer):
    #print(sentence)
    #tokenizer.fit_on_texts(sentence)
    sequences = tokenizer.texts_to_sequences([sentence])
    #print(sequences)
    data = pad_sequences(sequences, maxlen = 120)
    return data

BASE_DIR = '/home/research-pc/Hate_speech/cities'
model = load_model("modelv1.h5")


# texts = []
# for line in reader:
#         texts.append(line[0])
# #print(texts)
# tokenizer.fit_on_texts(texts)
# f.close()
# count = 0
#
# f1 = open(os.path.join(BASE_DIR,'ahemdabad_final.csv'), 'w+')
# writer = csv.writer(f1)
# for tweets in texts:
#     writer.writerow([model.predict(convert(tweets, tokenizer)).argmax(), tweets])

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith("_1.csv"):
            f = open(os.path.join(root, file))
            reader = csv.reader(f)

            texts = []
            for line in reader:
                    texts.append(line[1])
            #print(texts)
            tokenizer.fit_on_texts(texts)
            f.close()
            count = 0
            print(file)
            f1 = open(os.path.join(BASE_DIR, file+'_final.csv'), 'w+')
            writer = csv.writer(f1)
            for tweets in texts:
                try:
                    #print(model.predict(convert(tweets, tokenizer))[0][0])
                    writer.writerow([model.predict(convert(tweets, tokenizer)).argmax(),model.predict(convert(tweets, tokenizer))[0][0],model.predict(convert(tweets, tokenizer))[0][1],model.predict(convert(tweets, tokenizer))[0][2], tweets])
                except :
                    print("Error - "+tweets)
            # for line in f1:
            #     # print(line[0], model.predict(convert(line[0], tokenizer)).argmax())
            #     # print(line[0], model.predict(convert(line, tokenizer)).argmax())
            #     if model.predict(convert(line[0], tokenizer)).argmax() == 0:
            #         write.writerow([line[0], 'neither'])
            #     elif model.predict(convert(line[0], tokenizer)).argmax() == 1:
            #         write.writerow([line[0], 'sexism'])
            #     elif model.predict(convert(line[0], tokenizer)).argmax() == 2:
            #         write.writerow([line[0], 'racism'])
            #
            #     count+=1
