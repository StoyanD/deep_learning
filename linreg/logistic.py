import numpy as np

N = 100
D = 2

X = np.random.randn(N,D)
print "X"
print X
ones = np.array([[1]*N]).T
print "ones"
print ones
Xb = np.concatenate((ones, X), axis=1)
print "Xb"
print Xb
w = np.random.randn(D + 1)
print "w"
print w
z = Xb.dot(w)
print "z"
print z
def sigmoid(z):
    return 1/(1 + np.exp(-z))

print sigmoid(z)
