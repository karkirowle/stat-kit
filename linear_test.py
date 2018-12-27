

import linreg as l
import numpy as np
import datagen as datagen
import causality
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plot as p
#random_predictor = np.random.normal(0,1,(10,15))
#random_response = np.random.normal(0,1,(10,1))

X,Y, W = datagen.linear_system(1,0.01,100)

w_pred = l.linear_regression(X,Y,gradient=False)

# Bayesian Linear Regression
alpha = 1
beta = 1
gamma = 1
convergence = 0.000001
mu = 10

while True:
    mu_new,sigma = l.bayesian(X,Y,alpha,beta)
    if (np.abs(mu_new - mu) < convergence):
        break
    else:
       mu = mu_new
       alpha,gamma,beta = l.evidence_iteration(alpha,gamma,beta,X,Y,mu)

print("Normal Equation", w_pred)
print("Bayesian Estimated", mu)
print("True", W)

#evidence = l.bayesian_evidence(X,Y,mu,alpha,beta)
#print("Mean", mu)
#print("Variance", sigma)
#print("Evidence", evidence)
#p.normal_plot(mu[0],sigma[0])

#print(mu.shape)
#print(sigma.shape)

#idx = np.argsort(X[:,0])
#x_sorted = X[idx,0]
#fit_sorted = np.sort(fitted[idx])
# TODO: Sort
#plt.scatter(X[:,0],Y)
#plt.plot(x_sorted,fit_sorted)
#plt.show()
