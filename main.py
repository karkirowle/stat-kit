

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



start = time.time()
t,x,y,z = datagen.repressilator()
#t,x,y,z = datagen.lorentz_attractor()

fig = plt.figure()
#ax = fig.gca(projection='3d')


plt.plot(t,x,'r')
plt.plot(t,y,'b')
plt.plot(t,z,'g')
#ax.set_axis_off()

plt.show()
#w_pred = l.linear_regression(X,Y)
end = time.time()
print(end - start)

#causality.shadow_manifold(x)
causality.test_knn()
