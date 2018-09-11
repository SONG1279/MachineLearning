import numpy as np
from sklearn import svm


def load_data(filename):
    dataset = []
    label = []
    file = open(filename)
    for line in file.readlines():
        lineArr = line.strip().split('\t')
        m = len(lineArr)
        dataset.append(lineArr[0:m-1])
        label.append(lineArr[-1])
    return np.array(dataset, dtype=np.float64), \
           np.array(label, dtype=np.int).reshape(-1,1)

x,y = load_data('testSet.txt')
clf = svm.SVC(C=1.0)
clf.fit(x, y)
test = np.mat([[6, -4]])
print(test.shape)
pred_y = clf.predict(test)
print(pred_y)
print('----------------以上为项目数据的分类------------------')

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.3)
clf = svm.SVC(C=1.0,kernel='linear')
clf.fit(X, y)
plt.figure(figsize=(12,4), dpi=144)

()











# X, y = make_blobs(n_samples=100, centers=2,random_state=0, cluster_std=0.3)
# print(len(X), len(X[0]), len(y))


