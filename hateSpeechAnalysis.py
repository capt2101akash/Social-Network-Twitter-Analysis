import os
import csv
import sys

BASE_DIR = '/home/research-pc/Hate_speech/cities'

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith('csv_final.csv'):
            f = open(os.path.join(root, file), 'r+')
            sexism = 0
            racism = 0
            for line in f:
                if line[0] == '1':
                    sexism+=1
                elif line[0] == '2':
                    racism+=1
            print(file)
            print("    "+"sexism = ", sexism)
            print("    "+"racism = ", racism)
            f.close()
