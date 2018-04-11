#!/usr/bin/env python3
import numpy as np
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy import sparse
"""
We assume : if index i != 0,
then for every index j < i, there is a document
in which j:nb where nb > 0
"""


f = open("BaseReuters-29")

all_lab = []
row = []
column = []
freq = []
doc_size = 0

for line in f:  # Takes each line (document))
    doc_size += 1
    l = line.split()  # Takes every component of the document
    all_lab.append(int(l[0]))  # Put the class of the doc in all_lab
    for i in range(1, len(l)):  # for every component of the vector
        feat = l[i].split(":")  # separate index and value
        row.append(doc_size-1)  # current row
        column.append(int(feat[0]))  # process the index
        freq.append(int(feat[1]))  # process the value (at the index)
vocab_size = np.max(column)  # max index whose value is not 0
max_class = np.max(row)  # nb of lines
# print(vocab_size)

reuters = sparse.csr_matrix((freq, (row, column)),
                            shape=(doc_size, vocab_size+1))
""" Count the nb of doc in the classes
for nb in range(1, max(all_lab)+1):
    print("Nb of doc in class " + str(nb)
          + ": " + str(all_lab.count(nb)))
"""

x_train, x_test, y_train, y_test = train_test_split(reuters, all_lab,
                                                    test_size=18203)
clf = BernoulliNB()
clf.fit(x_train, y_train)
res = clf.score(x_test, y_test)
print("BernoulliNB\nScore : %.2f %% (random : %.2f %%)" % (res * 100, 100.0/29.0))


clf = MultinomialNB()
clf.fit(x_train, y_train)
res = clf.score(x_test, y_test)
print("\nMultinomialNB\nScore : %.2f %% (random : %.2f %%)" % (res * 100, 100.0/29.0))
"""
formulas : http://scikit-learn.org/stable/modules/naive_bayes.html
"""

# 5 fold
skf = StratifiedKFold(n_splits=5, shuffle=True)
j = 1
average = 0.0
for train_index, test_index in skf.split(reuters, all_lab):
    x_train, x_test = reuters[train_index], reuters[test_index]
    y_train, y_test = [all_lab[i] for i in train_index],\
                      [all_lab[i] for i in test_index]
    clf = BernoulliNB()
    clf.fit(x_train, y_train)
    res = clf.score(x_test, y_test)
    average += res
    print("\nMultinomialNB 5-Folds, Fold %d\nScore : %.2f %% (random : %.2f %%)"\
          % (j, res * 100, 100.0/29.0))
    j += 1
average /= 5.0
print("\nAverage score of 5-Folds cross-validation : %.2f %% (random : %.2f %%)"\
      % (average * 100, 100.0/29.0))
