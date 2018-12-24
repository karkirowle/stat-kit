import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Data generator


def linear_system(order,noise,timesteps):
	"""
	Generates a linear system with a given order (number of coefficients)
	and given level of noise (variance of a Gaussian, and the signal is some set number of timesteps.
	"""
	
	# The weights of the system are sampled from a Gaussian distribution
	W = np.random.normal(0,1,(order,1))

	# Random sampled time series are used as X
	X = np.random.normal(0,0.01,(timesteps,order))
	
	Y = X.dot(W) + np.random.normal(0,noise,(timesteps,1))
	
	return X,Y,W
	
def lorentz_attractor(tmax=100,res=0.01):
	"""
	Generates the X,Y,Z time series for a Lorentz attractor.
	tmax - gives the maximum time step for the ODE solver
	res - time resolution of the simulation
	
	"""

	def lorenz_model(all,t):
		sigma = 10
		beta = 2.667	
		ro = 28
		
		x,y,z = all

		dxdt = -sigma * (x - y)
		dydt = ro * x - y - x*z 
		dzdt = x * y - beta * z
		
		return dxdt,dydt,dzdt
	
	# Lorenz paramters and initial conditions
	
	
	# Maximum time point and total number of time points
	n = int(np.ceil(100 / res))
	# solve ODE
	t = np.linspace(0,tmax, n)
	
	time_series = odeint(lorenz_model,(0,1,1.05),t)
	x,y,z = time_series.T
	print(x.shape)
	print(y.shape)
	print(z.shape)
	
	return t,x,y,z
	
def repressilator(tmax=100,res=0.01):
	"""
	Generates the X,Y,Z time series for a repressilator.
	
	
	"""

	def rep_model(all,t):
		d = 0.1
		a = 0.3
		b = 0.2
		c = 0.4
		x,y,z = all

		dxdt = a /(1 + z)**2 - d * x 
		dydt = b/(1 + x)**2 - d * y
		dzdt = c/(1 + y)**2 - d * z 
		
		return dxdt,dydt,dzdt
	
	# Lorenz paramters and initial conditions
	
	
	# Maximum time point and total number of time points
	n = int(np.ceil(100 / res))
	# solve ODE
	t = np.linspace(0,tmax, n)
	
	time_series = odeint(rep_model,(0.1,0.2,0.3),t)
	x,y,z = time_series.T
	print(x.shape)
	print(y.shape)
	print(z.shape)
	
	return t,x,y,z