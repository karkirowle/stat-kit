
# Causality

import numpy as np
import matplotlib.pyplot as plt
import sys
		
class KNN:
	"""
	Implements a k nearest neighbour classifier
	It is not a classifier in the strictest sense, as it returns the actual closest point in the training set, rather than
	a classification index
	"""
	def __init__(self):
		self.cucc = 0
		
	def fit(self,train):
		"""
		
		Stores the training set which
		
		Parameters
		----------
		train : 2d numpy array (N,D)
		
		"""
		self.train = train
	
	def classify(self,query, k=1):
		"""
		Returns the closest points from the dictionary of training points
		based on an Euclidean evaluation measure
		
		Parameters
		----------
		query : 2d numpy Float32 array (N,D)
		
		k : Number of neighbours to return, must be smaller than train length
		
		Returns:
		points : 2d numpy Float32 array (N,D) 
		"""
		
		N_train = self.train.shape[0]
		D_train = self.train.shape[1]
		
		assert k < N_train
		
		N_query = query.shape[0]
		D_query = query.shape[1]
		
		assert D_train == D_query
		
		points = np.zeros((N_query,D_query,k))
		
		for i,points1 in enumerate(query):
		
			distances = np.zeros((N_train))
			for j,points2 in enumerate(self.train):
			
				# Calculate Euclidean distances
				distances[j] = np.sqrt(np.sum((points2-points1)**2))
				
			# Sort, and return k largest indices in the form idx[:k]
			# Argpartition is effectively only a guarantee to sort the first k, and then select using idx[:k]
			idx = np.argpartition(distances, k)
			points[i,:,:] = self.train[idx[:k],:].T
	
		return points

def shadow_manifold(x, dim=2, lag=1):
	"""
	Creates a shadow manifold plot object with an embedding dimension dim=2
	and time series lag 1.
	"""
	
	# Two time series are crated (dim=2), one with lag, and one cropped to match time series dimensiosn
	x_lag = x[lag:]
	x_crop = x[:-lag]
	
	plt.plot(x_crop, x_lag)
	plt.show()
	
def test_knn():
	"""
	Tests the K nearest neighbour classifier with an example training numpy 2d array
	and prints out the results
	"""
	
	knn_1 = KNN()
	
	train = np.array([[1,3],[1,2],[1,1]]) # 2x2 array
	query = np.array([[1,1.1],[1,1.9]]) # query point, should return [1,1] 
	
	knn_1.fit(train)
	classification = knn_1.classify(query,2)
	
	print(classification)