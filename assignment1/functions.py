"""
Sample code automatically generated on 2020-11-16 18:48:06

by www.matrixcalculus.org

from input

d/dW X * W = eye\otimes X

where

W is a matrix
X is a matrix

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(W, X):
    assert isinstance(W, np.ndarray)
    dim = W.shape
    assert len(dim) == 2
    W_rows = dim[0]
    W_cols = dim[1]
    assert isinstance(X, np.ndarray)
    dim = X.shape
    assert len(dim) == 2
    X_rows = dim[0]
    X_cols = dim[1]
    assert W_rows == X_cols

    functionValue = (X).dot(W)
    gradient = np.einsum('ik, jl', X, np.eye(W_cols, W_cols))

    return functionValue, gradient

def checkGradient(W, X):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.ones((3, 3)) #np.random.randn(3, 3)
    print(delta)
    f1, _ = fAndG(W + t * delta, X)
    f2, _ = fAndG(W - t * delta, X)
    f, g = fAndG(W, X)
    print('tensordot', np.tensordot(g, np.eye(3, 3), axes=2))
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=2)))

def generateRandomData():
    W = np.random.randn(3, 3)
    X = np.random.randn(3, 3)

    return W, X

if __name__ == '__main__':
    W, X = generateRandomData()
    functionValue, gradient = fAndG(W, X)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(W, X)
