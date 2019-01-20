
import numpy as np
class AdamOpt():

	def __init__(self,weights, alpha=0.1, beta1=0.9, beta2=0.999, epsilon = 1e-8):


		assert beta1 < 1
		assert beta2 < 1
		assert epsilon > 0
		assert alpha > 0
		
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.alpha = alpha
		
		self.m = np.zeros(weights.shape)
		self.r = np.zeros(weights.shape)
		self.w = weights
		
		self.t = 1

	def update(self,grad):

		self.m = self.beta1 * self.m +  (1 - self.beta1) * grad
		self.r = self.beta2 * self.r +  (1 - self.beta2) * grad**2
		m_hat = self.m / ( 1 - self.beta1**self.t )
		r_hat = self.r / ( 1 - self.beta2**self.t )
		
		self.w = self.w - (self.alpha * m_hat) / np.sqrt(r_hat + self.epsilon)
		self.t = self.t + 1
		return self.w
