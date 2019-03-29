from solve_equations import *
import numpy as np

A = np.array([[1,1,1],[3,3,4],[2,1,3]])
B = np.array([[6],[20],[13]])

X_gauss = gauss_elimination(A,B)
assert np.array_equal(X_gauss, np.array([[3],[1],[2]])), "error: Gauss elimination answer isn't matching"
print('Gauss elimination answer matches')
