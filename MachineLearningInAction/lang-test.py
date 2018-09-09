import numpy as np

datMat = np.matrix(
    [[1., 2.1],
     [2., 1.1],
     [1.3, 1.],
     [1., 1.],
     [2., 1.]])
a = np.ones((np.shape(datMat)[0], 1))
print(a)
print('-------------')
b = np.ones(np.shape(datMat)[0])
print(b)