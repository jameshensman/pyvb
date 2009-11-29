class Node:
	"""base class for a node
	
	Arguments
	----------	
	shape : tuple
		shape of the node
	
	Attributes
	----------
	children : list
		A list of the children of this node
	shape : tuple
		shape of this node

    References
	----------
	C. M. Bishop (2006) Pattern Recognition and Machine Learning
	"""
	def __init__(self, shape):
		self.children = []
		self.shape = shape
	
	def addChild(self,child):
		"""add a child to this node	
			
		Arguments
		----------
		child : node
			child node to add to this node
		"""
		self.children.append(child)

	
	def __add__(self,other):
		return Addition(self,other)
	
	def __mul__(self,other):
		return Multiplication(self,other)
	def __rmul__(self,other):
		return Multiplication(other,self)