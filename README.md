# Python-FastICA
Implementation and optimization of FastICA algorithm in Python

Colone the GitHub repository first, and then follow the steps below to install the FastICA package. An test example is also provided.

!pip install .

from fastica_lz import fastica_lz as lz

test(lz.fastica_s(X_test,f = "logcosh",n_comp =2, alpha = 1,maxit = 200, tol = 0.0001)['S'])
