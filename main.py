

import linreg as l
import numpy as np
import datagen as datagen
import causality
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#random_predictor = np.random.normal(0,1,(10,15))
#random_response = np.random.normal(0,1,(10,1))

X,Y, W = datagen.linear_system(5,0.01,100)

import time



import dimensional_causality as dc

k_range = range(10, 40, 1)
print([i for i in k_range])
t,x,y,z = datagen.repressilator(res=0.001,noise=0)

plt.plot(t,x,'r')
plt.plot(t,y,'g')
plt.plot(t,z,'b')
plt.show()
plt.acorr(x)
plt.show()
emb_dim = 2 * 3 + 1 # 3 states

probs, dims, stdevs = dc.infer_causality(x, y, 4, 1, k_range)
#probs, dims, stdevs = dc.infer_causality(y, z, 4, 9, k_range)
#probs, dims, stdevs = dc.infer_causality(z, x, 4, 9, k_range)

print(probs)

#t,x,y,z = datagen.lorentz_attractor()

fig = plt.figure()
#ax = fig.gca(projection='3d')


plt.plot(t,x,'r')
plt.plot(t,y,'b')
plt.plot(t,z,'g')
#ax.set_axis_off()

plt.show()
#w_pred = l.linear_regression(X,Y)
