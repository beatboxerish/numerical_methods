# this file contains some algorithms that solve a system of equations

import numpy as np


def ge(n=3):
    '''
    Generate a nxn sysytem of equations. Eg: AX = B where A is nxn matrix
    and B is nx1 matrix. X is the matrix that we will try to find with the
    subsequent functions.
    '''
    # generate random nxn matrix(A). Note: The values will be within 0 and 1
    A = np.random.rand(n, n)
    # generate B
    B = np.random.randint(1, 11, (n, 1))
    return A, B


def gauss_elimination(A, B):
    '''
    This function uses the Gauss Elimination method to solve a system of n
    linear equations. Pivoting can be equal to 'None' or 'partial'.

    This function returns values for X for any given A and B where AX = B.
    '''
    n = A.shape[0]

    # making A an upper triangular matrix
    for i_of_pivot in range(n):
        pivot = A[i_of_pivot, i_of_pivot]

        if pivot == 0:
            if i_of_pivot != n - 1:  # it isnt the last row
                A_new = A[i_of_pivot, :].copy()
                A[i_of_pivot, :] = A[i_of_pivot + 1, :]
                A[i_of_pivot + 1, :] = A_new
                B_new = B[i_of_pivot].copy()
                B[i_of_pivot] = B[i_of_pivot + 1, :]
                B[i_of_pivot + 1] = B_new
                pivot = A[i_of_pivot, i_of_pivot]

        for i in range(n):  # to decide which row to use operations on, i.e, i stands for the ith row
            if i > i_of_pivot:  # this allows us to take only the rows after the row where the pivot entry is
                B[i] = B[i] - B[i_of_pivot]/pivot * A[i, i_of_pivot] 
                # The below code changes A's elements and that changes A[i,i_of_pivot] to 0 making the above useless
                A[i, :] = A[i,:] - A[i_of_pivot,:]/pivot * A[i,i_of_pivot]

    # using back-substitution to get X
    X = np.zeros(n).reshape(n, 1) # form initial X
    for i in range(n, 0,-1): # we go from the nth row to the 1st row
        A_temp = A[i-1, i-1:] # this contains the values of A that we shall use to subtract and divide B with
        X[i-1] = 1
        A_temp = A_temp.reshape(-1,) * X[i-1:].reshape(-1,)
        if len(A_temp) == 1:
            X[i-1] = B[i-1]/A_temp[0]
        else:
            X[i-1] = (B[i-1] - A_temp[1:].sum())/A_temp[0]
    return X
