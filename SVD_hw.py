from numpy.linalg import svd, det
import numpy as np

A = np.array([[3,2,2],[2,3,-2]])
print(A)
u, s, vh = svd(A)
print(u)
det_U = det(u)
print(det_U)