
import numpy as np
import matplotlib.pyplot as plt

# Source matrix
a = ((np.arange(200)+1)-100)/100
a = np.concatenate((a,a,a,a,a), axis=0) 
b = np.sin((np.arange(1000)+1)/20)
S_test= np.vstack((b,a)).T
# Mixing matrix
A = np.array([0.291, 0.6557, -0.5439, 0.5572]).reshape((2, 2))
# test data
X_test = S_test @ A

def test(ic):
    np.random.seed(1)
    plt.subplot(121)
    plt.plot(np.arange(1000)+1, ic[:,0])
    plt.title("IC 1")
    plt.subplot(122)
    plt.plot(np.arange(1000)+1, ic[:,1])
    plt.title("IC 2")
pass