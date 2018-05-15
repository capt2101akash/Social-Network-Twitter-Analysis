from __future__ import print_function

import os
import sys
import csv

BASE_DIR = ''

f1 = csv.reader(open(os.path.join(BASE_DIR, 'clean_data.csv')))

f2 = open(os.path.join(BASE_DIR, 'test_data.csv'), 'w')
write = csv.writer(f2)

values = {}
num = 0
for line in f1:
    if num <= 1200:
        #print(line)
        write.writerow([line[0], line[1], line[2]]);
    num+=1;
