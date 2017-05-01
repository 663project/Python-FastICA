
# Generate observation data
import numpy as np
from scipy import signal

np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)                   # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))          # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S_compare = np.c_[s1, s2, s3]
S_compare += 0.2 * np.random.normal(size=S_compare.shape)  # Add noise

S_compare /= S_compare.std(axis=0)                      # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X_compare = np.dot(S_compare, A.T)                      