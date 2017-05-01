# STA 663 Final Project

## Implementation and optimization of FastICA algorithm in Python

#### Blaire Li, Liwen Zhang

This is Blaire Li and Liwen Zhang's Final Project Report for Duke STA-663.

Please clone our repsository and download dependencies to run our report, "Independent_Component_Analysis.ipynb".


To intall our FastICA package, please clone the GitHub repository first, and then follow the steps below. An test example is also provided.

!pip install .

from fastica_lz import fastica_lz as lz

test(lz.fastica_s(X_test,f = "logcosh",n_comp =2, alpha = 1,maxit = 200, tol = 0.0001)['S'])
