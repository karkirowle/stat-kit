# Linear regression


import numpy as np

def linear_regression(predictor,response,l2=0,gradient=False):
	"""
	Performs linear regression using
		predictor: (Float/Double) X \in R^{m*n}
		response: (Float/Double) Y \in R^{m*p}
		l2: (Float/Double) Amount of L2 regularisation

	Returns a weight matrix W \in R^{n*p}
	The standard linear form is thus Y = X * W 
	"""
	X = predictor
	Y = response
	
	if (underdetermined(predictor) or gradient):
                # TODO: L2 is not implemented
		W = gradient_descent(X,Y)
	else:
		W,_ = normal_equation(X,Y,l2)
	
	return W

def normal_equation(X,Y,l2):
	"""
	Solves the least squares normal equations given the matrices X and Y.
	"""
	D = np.linalg.inv(X.T.dot(X) + l2)
	W = D.dot(X.T).dot(Y)
	return W,D

def cost_func(X,Y,W,alpha,beta,q):
        """
        Returns cost,grad,Hessian
        q - refers to the q-th Minkowski loss, q >=1
        alpha - regulariser
        beta
        """
        print(q)
        assert q >= 1, "Uninterpretable Minkowski space"

        cost_val = beta/2 * np.sum((X.dot(W)- Y)**q) + alpha/2 * W.T.dot(W)
        grad = beta/2 * q * X.T.dot((X.dot(W) - Y)**(q-1)) + alpha * W

        # Because of different differential rule
        if (q == 2):
                hessian = beta * X.T.dot(X) + alpha
        else:
                hessian = beta / 2 * q * (q-1) * X.T.dot(X) + alpha

        # TODO: This probably does not work for q = 1
        return cost_val,grad,hessian

def gradient_descent(X,Y,mu=0.01,convergence=0.1):
        """
        Solves the least squares problem with a naive gradient descent solution.
        The cost function is strictly Euclidean, so it imposes implicit
        Gaussian distribution on the data.
        """
        N = X.shape[1]
        P = Y.shape[1]
        W = np.random.normal(0,1,(N,P))
	
	# Sum of squared error loss function - implicit Gaussainity imposed
        error,grad,hessian = cost_func(X,Y,W,1,0,2)
        while (error[0] > convergence):
                W = W - mu * grad
                error,grad,hessian = cost_func(X,Y,W,1,0,2)

        return W

def underdetermined(predictor):
	"""
	Given a predictor matrix tells if it is underdetermined.
	"""
	
	rank = np.linalg.matrix_rank(predictor)
	return rank < predictor.shape[1]
	
def bayesian(predictor,response,alpha,beta):
        """
        Wrapper for the Bayesian Linear regression
        beta - noise precision parameter (scalar)
        alpha - prior (scalar)
        predictor - (Datapoints X Parameter) (N x M) 
        response - (Datapoints X Response dim) (N x P)
        Implemented according to Bishop Eq. 3.53 and 3.54
        """

        N = predictor.shape[0]
        M = predictor.shape[1]
        

        # Bishop equations can be recasted to normal equations
        #sigma_post_inv = alpha * identity + beta * predictor.T.dot(predictor)
        #sigma_post1 = np.linalg.inv(sigma_post_inv)
        #mu_post1 = beta * sigma_post1 * predictor.T.dot(response)
        mu_post, sigma_post = normal_equation(predictor,response,alpha/beta)
        
        return mu_post, sigma_post

def evidence_iteration(alpha,gamma,beta,predictor,response,w):
        """
        Calculates the new hyperparameters given the 
        Parameters:
        alpha
        gamma
        w

        Calculates the new alpha and gamma
        """

        N = predictor.shape[0]

        # TODO: Eigenvalue calculation
        eigval = np.linalg.eig(beta * (predictor.T).dot(predictor))
        alpha = gamma / w.T.dot(w)
        gamma = np.sum(eigval / (eigval + alpha))

        inv_beta = 1/(N-gamma) * np.sum(response - predictor.dot(w))
        beta = 1 / inv_beta
        
        return alpha,gamma,beta
def bayesian_evidence(predictor,response,W,alpha,beta):
        """
        Calculates the model evidence for a Bayesian linear regression
        with a normally distributed prior.
        alpha - 
        beta -
        predictor - X
        Based on Bishop Equaton (3.86)
        """
        X = predictor
        Y = response
        N = predictor.shape[0]
        M = predictor.shape[1]

        # TODO: E(Mn)
        error,grad,hessian = cost_func(X,Y,W,alpha,beta,2)
        
        emn,_,_ = cost_func(X,Y,W,1,0,2)
        log_det_hessian = np.log(np.linalg.det(hessian))/2
        
        # Full evidence 
        evidence = M * np.log(alpha) / 2
        + N * np.log(beta) / 2
        - emn
        - log_det_hessian
        - N * np.log(2 * 3.14) / 2
        return evidence

def r_squared(w,predictor,response):
        """
	Performs r squared calculation 
		predictor: (Float/Double) X \in R^{m*n}
		response: (Float/Double) Y \in R^{m*p}
                weight matrix: W \in R^{n*p}
	The standard linear form is thus Y = X * W 
        
        Returns the r-squared and the adjusted r-squared
        """
        N = predictor.shape[0] # samples
        M = predictor.shape[1] # parameter order
        y_actual = response
        y_predicted = predictor.dot(w)
        y_mean = np.mean(response)
        r = 1 - (np.sum((y_actual - y_predicted)**2) / np.sum((y_actual-y_mean)**2))
        adjusted = 1 - ((1-r)*(N-1))/(N-M-1)
        return r, adjusted
