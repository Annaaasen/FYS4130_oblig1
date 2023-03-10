import numpy as np 
import matplotlib.pyplot as plt 
from scipy import optimize
import plot_utils


T = np.linspace(0, 3.5, 1000)
a = 1/T 

S = a/(np.exp(a) + 1) + np.log(1 + np.exp(-a))

plt.plot(T, S)
plt.xlabel("T")
plt.ylabel("S")
plt.show()