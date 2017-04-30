import scipy
import numpy as np


def sym_decorrelation(W):
    """ Symmetric decorrelation """
    K = np.dot(W, W.T)
    s, u = np.linalg.eigh(K) 
    W = (u @ np.diag(1.0/np.sqrt(s)) @ u.T) @ W
    return W

def g_logcosh(wx,alpha):
    """derivatives of logcosh"""
    return np.tanh(alpha * wx)
def gprime_logcosh(wx,alpha):
    """second derivatives of logcosh"""
    return alpha * (1-np.square(np.tanh(alpha*wx)))
# exp
def g_exp(wx,alpha):
    """derivatives of exp"""
    return wx * np.exp(-np.square(wx)/2)
def gprime_exp(wx,alpha):
    """second derivatives of exp"""
    return (1-np.square(wx)) * np.exp(-np.square(wx)/2)


def fastICA_3(X, f,alpha=None,n_comp=None,maxit=200, tol=1e-04):
    """FastICA algorithm for several units"""
    n,p = X.shape
    #check if n_comp is valid
    if n_comp is None:
        n_comp = min(n,p)
    elif n_comp > min(n,p):
        print("n_comp is too large")
        n_comp = min(n,p)
       
    #centering
    #by subtracting the mean of each column of X (array).
    X = X - X.mean(axis=0)[None,:]
    X = X.T
 
    #whitening
    s = np.linalg.svd(X @ (X.T) / n)
    D = np.diag(1/np.sqrt(s[1]))
    k = D @ (s[0].T)
    k = k[:n_comp,:]
    X1 = k @ X
   
    # initial random weght vector
    w_init = np.random.normal(size=(n_comp, n_comp))
    W = sym_decorrelation(w_init)
 
    lim = 1
    it = 0
   
    # The FastICA algorithm
    while lim > tol and it < maxit :
        wx = W @ X1
        if f =="logcosh":
            gwx = g_logcosh(wx,alpha)
            g_wx = gprime_logcosh(wx,alpha)
        elif f =="exp":
            gwx = g_exp(wx,alpha)
            g_wx = gprimeg_exp(wx,alpha)
        else:
            print("doesn't support this approximation negentropy function")
        W1 = np.dot(gwx,X1.T)/X1.shape[1] - np.dot(np.diag(g_wx.mean(axis=1)),W)
        W1 = sym_decorrelation(W1)
        it = it +1
        lim = np.max(np.abs(np.abs(np.diag(W1 @ W.T))) - 1.0)
        W = W1
 
    S = W @ X1
    A = scipy.linalg.pinv2(W @ k)   
    return{'X':X1.T,'A':A.T,'S':S.T}