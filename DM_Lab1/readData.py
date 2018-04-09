import numpy as np
import random as rd
from scipy import sparse
"""
We assume : if index i != 0,
then for every index j < i, there is a document
in which j:nb where nb > 0
"""


def check_data(X, Y, nb_class=29):
    for i in range(1, nb_class+1):
        ok = False
        for x in X:
            if int(x.split()[0]) == i:
                ok = True
                break
        if not ok:
            print("Class " + str(i) + " not found in training data")
            return True
        ok = False
        for y in Y:
            if int(y.split()[0]) == i:
                ok = True
                break
        if not ok:
            print("Class " + str(i) + " not found in test data")
            return True
    return False


f = open("BaseReuters-29")

all_lab = []
row = []
column = []
freq = []
doc_size = 0

copy_f = []
for line in f:  # Takes each line (document)
    copy_f.append(line)
    doc_size += 1
    l = line.split()  # Takes every component of the document
    all_lab.append(int(l[0]))  # Put the class of the doc in all_lab
    for i in range(1, len(l)):  # for every component of the vector
        feat = l[i].split(":")  # separate index and value
        row.append(doc_size)  # current row
        column.append(int(feat[0]))  # process the index
        freq.append(int(feat[1]))  # process the value (at the index)

vocab_size = np.max(column)  # max index whose value is not 0
max_class = np.max(row)  # nb of lines
print(vocab_size)
"""
for nb in range(1, max(all_lab)+1):
    print("Nb of doc in class " + str(nb)
          + ": " + str(all_lab.count(nb)))
"""
# Split
again = True
training_data = None
test_data = None
while again:
    rd.shuffle(copy_f)
    training_data = copy_f[:52501]
    test_data = copy_f[52501:]
    again = check_data(training_data, test_data)
print("Data splitted successfully")

all_lab.append(1)
reuters = sparse.csr_matrix((freq, (row, column)),
                            shape=(doc_size+1, vocab_size+1))
