import numpy as np
import node

class Constant(node.Node):
	"""A class to model a constant in a Bayesian Network,
    
	Arguments
	----------	
	value : numpy.array
    
	Attributes
	----------
    
	Notes
	----------
	Essentially a wrapper around a numpy array. This allows us to  make __mul__ and __rmul__ behave in the way we want.
	
	A constant should be the parent of other nodes only.  
	"""
	def __init__(self,value):
		node.Node.__init__(self,value.shape)
		self.shape = value.shape
		self.value = value
		self.value_xxT = np.dot(value,value.T)
		self.value_xTx = np.dot(value.T,value)
	
	def pass_down_Ex(self):
		return self.value
	def pass_down_ExxT(self):
		return self.value_xxT
	def pass_down_ExTx(self):
		return self.value_xTx