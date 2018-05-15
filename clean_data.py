from __future__ import print_function

import os
import sys
import csv

BASE_DIR = ''

f1 = csv.reader(open(os.path.join(BASE_DIR, 'crawled_tweets.csv')))
values = {}
for line in f1:
    values.update({line[0]:line[1]})

f3 = csv.reader(open(os.path.join(BASE_DIR, 'dataset_2.csv')), delimiter= ',')
labels = {}

for line in f3:
    #labels.update({line[0]: max(set([line[1],line[1],line[2],line[3],line[4]]))})
    labels.update({line[0]: line[1]})
f2 = open(os.path.join(BASE_DIR, 'clean_data.csv'), 'a')
lab = csv.writer(f2)
for key, value in values.items():
    if key in labels:
        lab.writerow([key,value,labels.get(key)])

f2.close()
