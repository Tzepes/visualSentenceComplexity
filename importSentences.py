import csv
import numpy as np

train_data = open("./data/train.csv")
test_data = open("./data/test.csv")

train_ids = []
test_ids = []

train_sentences = []
test_sentences = []

train_scores = []

train_reader = csv.reader(train_data)
test_reader = csv.reader(test_data)

for row in train_reader:
    train_ids.append(row[0])
    train_sentences.append(row[1])
    train_scores.append(row[-1])  # Get the last column

for row in test_reader:
    test_ids.append(row[0])
    test_sentences.append(row[1])

print(train_sentences)