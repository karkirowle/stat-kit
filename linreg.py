# Linear regression


import numpy as np

def linear_regression(predictor,response):
	"""
	Performs linear regression using
		predictor: (Float/Double) X \in R^{m*n}
		response: (Float/Double) Y \in R^{m*p}
		
	Returns a weight matrix W \in R^{n*p}
	The standard linear form is thus Y = X * W 
	"""
	
	X = predictor
	Y = response
	
	if underdetermined(predictor):
		W = gradient_descent(X,Y)
	else:
		W = normal_equation(X,Y)
	
	return W

def normal_equation(X,Y):
	"""
	Solves the least squares normal equations given the matrices X and Y.
	"""
	D = X.T.dot(X)
	W = np.linalg.inv(D).dot(X.T).dot(Y)
	return W

def gradient_descent(X,Y,mu=0.01,convergence=0.1):
	"""
	Solves the least squares problem with a naive gradient descent solution.
	"""
	N = X.shape[1]
	P = Y.shape[1]
	W = np.random.normal(0,1,(N,P))
	
	# Sum of squared error loss function - implicit Gaussainity imposed
	error = (X.dot(W) - Y)**2
	while (np.linalg.norm(error) > convergence):
		W = W - mu * X.T.dot(error) 
		error = (X.dot(W) - Y)**2
	
	return W
	
def underdetermined(predictor):
	"""
	Given a predictor matrix tells if it is underdetermined.
	"""
	
	rank = np.linalg.matrix_rank(predictor)
	return rank < predictor.shape[1]
	

