from __future__ import print_function

import os
import sys
import csv

BASE_DIR = ''

f = csv.reader(open(os.path.join(BASE_DIR, 'dataset_2.csv')))
f1 = open(os.path.join(BASE_DIR, 'tweet_ids.txt'), 'w')
for line in f:
    f1.write(line[0])
    f1.write('\n')
f1.close()
