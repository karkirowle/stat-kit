

import numpy as np
import datagen as datagen
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pandas import Series
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller


import time



import dimensional_causality as dc
from scipy.signal import argrelextrema

def autocorr(x):
    result = np.correlate(x, x, mode='full')[len(x)-1:]
    return result

def period(x,fs):
    x_c = autocorr(x)
    y = argrelextrema(x_c, np.greater)
    period = y[0][0] / fs
    print(period)
    return y[0][0] 


# 0. Generate the repressilator data
k_range = range(10, 40, 1)
t,x,y,z = datagen.repressilator(tmax=100,fs=200,noise=0)

# 1a. Check for stationarity
x_df = Series(x)
result = adfuller(x_df)
p_value = result[1]
print("ADF p-value:",p_value)

# 1b. Check for L2 distance (same scale)

l2_xy = np.sqrt(np.mean((x-y)**2))
print("L2 distance of X and Y:", l2_xy)

# 2. Obtain period using ACF
print("Lag period:", period(x,100))

plt.subplot(1,2,1)
plt.plot(x)
plt.plot(y)
plt.plot(z)
plt.subplot(1,2,2)
plt.plot(t,x)
plt.plot(t,y)
plt.plot(t,z)
plt.show()

# Based on Taken's embedding Theorem
emb_dim = 2 * 3 + 1 # 3 states

# Period based on automatic detection or eyeballing the period
tau = period(x,100)


probs, dims, stdevs = dc.infer_causality(x, y, emb_dim, tau, k_range)


