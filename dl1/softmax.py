import numpy as np

X = np.random.randn(10,3)

W = np.random.randn(3,5)

Z = np.tanh(X.dot(W))

V = np.random.randn(5,3)

A = Z.dot(V)

expA = np.exp(A)

out = expA/expA.sum(axis=1, keepdims=True)
