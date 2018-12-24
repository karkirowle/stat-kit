
import unittest
import datagen as datagen
import numpy as np
class TestLinReg(unittest.TestCase):

	def test_linreg(self):
		self.assertEqual(True,True)
		
	def test_datagen(self):
		# Makes sure that the generated data is indeed a linear system
		X,Y,W = datagen.linear_system(10,0,1000)
		error = np.sum(np.dot(X,W) - Y)
		print(error)
		self.assertEqual(0,error)
		
if __name__ == '__main__':
    unittest.main()