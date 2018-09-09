import numpy as np


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


trainMat, trainLabel = loadDataSet('horseColicTraining2.txt')
testMat, testLabel = loadDataSet('horseColicTest2.txt')

trainMat, trainLabel = np.mat(trainMat), np.mat(trainLabel).T
testMat, testLabel = np.mat(testMat), np.mat(testLabel).T

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score


for i in [1, 10, 20, 30, 40, 50, 60, 70]:
    clf = AdaBoostClassifier(n_estimators=i)
    clf.fit(trainMat, trainLabel)
    score = clf.score(testMat, testLabel)
    print('in iterate ', i, ' score =', score)
