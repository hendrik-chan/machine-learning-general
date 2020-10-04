#https://www.kdnuggets.com/2018/05/wtf-tensor.html
import numpy as np

x = np.array(42)
print(x)
print(x.shape)
print('A scalar is of rank %d' %(x.ndim))

x = np.array([1, 1, 2, 3, 5, 8])
print(x)
print(x.shape)
print('A vector is of rank %d' %(x.ndim))

x = np.array([[1, 4, 7],
              [2, 5, 8],
              [3, 6, 9]])
print(x)
print(x.shape)
print('A matrix is of rank %d' %(x.ndim))

x = np.array([[[1, 4, 7],
               [2, 5, 8],
               [3, 6, 9]],
              [[10, 40, 70],
               [20, 50, 80],
               [30, 60, 90]],
              [[100, 400, 700],
               [200, 500, 800],
               [300, 600, 900]]])
print(x)
print(x.shape)
print('This tensor is of rank %d' %(x.ndim))