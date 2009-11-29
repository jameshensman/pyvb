# -*- coding: utf-8 -*-
import numpy as np

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

class Addition(Node):
	"""creates a node by adding two other nodes together (or adding a np array to a node).  

	Arguments
	----------
	x1 : numpy.array or node
		the first node
	x2 : numpy.array or node
		the second node

	Attributes
	----------

	Notes
	----------

	See Also
	--------

	References
	----------

	Examples
	--------

	"""
	def __init__(self,x1,x2):
		assert x1.shape == x2.shape, "Bad shapes for addition"
		Node.__init__(self, x1.shape)
		if type(x1) == np.ndarray:
			self.x1 = Constant(x1)
		else:
			self.x1 = x1

		if type(x2) == np.ndarray:
			self.x2 = Constant(x2)
		else:
			self.x2 = x2

		x1.addChild(self)
		x2.addChild(self)

	def pass_up_ex(self,requester):
		"""Return the 'mu' update message of the child node, modified by the co-parent
		expected value of the co-parent

		Arguments
		----------
		requester : node
			requester is either parent of this node
		"""
		sumMu = sum([e.pass_up_ex(self) for e in self.children])
		sumC = sum([e.pass_up_prec(self) for e in self.children])
		if requester is self.x1:
			return sumMu - np.dot(sumC,self.x2.pass_down_Ex())
		elif requester is self.x2:
			return sumMu - np.dot(sumC,self.x1.pass_down_Ex())

	def pass_up_prec(self, requester):
		"""return the sum of the precision matrices for the children of this
		node. - This is the 'prec' update message to the parent node.
		"""
		# TODO aint no dependence on the argument 'requester' 
		#careful - pass_up prec is common to all (Gaussian-like) nodes, and some of them need to know the requester.

		#get prec from children to pass upwards
		sumC = sum([e.pass_up_prec(self) for e in self.children])
		return sumC

	def pass_down_Ex(self):
		""" Return the sum of the expected values of the parent nodes

		Notes
		----------
		<A+B> = <A> + <B>
		"""
		return self.x1.pass_down_Ex()+self.x2.pass_down_Ex()

	def pass_down_ExxT(self):
		"""Return the expected value of the 'outer procuct' of the sum of the parent node

		Notes
		----------
		Is this how I use latex, Mike?
		$ <(A+B)(A+B)>^\top = <AA^\top> + <BB^\top> + <A><B>^\top + <B><A>^\top $""" 
		return self.x1.pass_down_ExxT() + self.x2.pass_down_ExxT() + np.dot(self.x1.pass_down_Ex(),self.x2.pass_down_Ex().T) + np.dot(self.x2.pass_down_Ex(), self.x1.pass_down_Ex().T)

class Multiplication(Node):
	def __init__(self,x1,x2):

		m1,n1 = x1.shape
		m2,n2 = x2.shape
		assert n1 == m2, "incompatible multiplication dimensions"
		assert n2 == 1, "right hand object must be a vector"
		Node.__init__(self, (m1,n2))
		if type(x1) == np.ndarray:
			self.x1 = Constant(x1)
		else:
			self.x1 = x1

		if type(x2) == np.ndarray:
			self.x2 = Constant(x2)
		else:
			self.x2 = x2

		x1.addChild(self)
		x2.addChild(self)

	def pass_up_ex(self,requester):
		""" Pass up the 'mu' message to the parent.

		Notes
		----------
		1) get the mu message from the child(ren)
		2) modify my appropriate co-parent
		3) pass it up the network
		"""
		sumMu = sum([e.pass_up_ex(self) for e in self.children])
		if requester is self.x1:
			if self.x1.shape[1] == 1:#lhs is column: therefore rhs is scalar: easy enough
				return float(self.x2.pass_down_Ex())*sumMu
			elif self.x1.shape[0] == 1:#lhs is a transposed vector (or hstacked scalars?)  
				return self.x2.pass_down_Ex().T*float(sumMu)
			else: #lhs is a matrix!
				raise NotImplementedError,"Hstack objects not done yet"
		elif requester is self.x2:
			return  np.dot(self.x1.pass_down_Ex().T,sumMu)


	def pass_up_prec(self,requester):
		"""
		Pass up the 'prec' message to the requesting parent

		1) get prec message from child(ren)
		2) modify it by the co-parent
		3) pass up."""
		sumC = sum([e.pass_up_prec(self) for e in self.children])
		if requester is self.x1:
			if self.x1.shape[1]==1:# one column (rhs scalar): easy enough
				x2x2T = self.x2.pass_down_ExxT()# this must be scalar in this case?
				return sumC*float(x2x2T)
			elif self.x1.shape[0] == 1:#lhs is a transpose (or hstack of scalars?)
				return float(sumC)*self.x2.pass_down_ExxT()
			else:
				raise NotImplementedError,"Objects with width (transposes, hstacks) not supported yet"
		elif requester is self.x2:
			if self.x1.shape[1]==1:#left object has only one column...
				x1x1T = self.x1.pass_down_ExxT()
				return np.trace(np.dot(x1x1T,sumC))
			elif self.x1.shape[0] == 1:# lhs is transpose (or hstacked scalars)
				return float(sumC)*self.x1.pass_down_ExTx()
			else:#lhs is a matrix.
				if isinstance(self.x1,Constant):
					return np.dot(self.x1.pass_down_Ex().T,np.dot(sumC,self.x1.pass_down_Ex()))
				else:
					raise NotImplementedError,"Objects with width (transposes, hstacks) not supported yet"

	def pass_down_Ex(self):
		return np.dot(self.x1.pass_down_Ex() , self.x2.pass_down_Ex())

	def pass_down_ExxT(self):
		if self.x1.shape[1] == 1:#rhs is scalar: this is quite easy
			return self.x1.pass_down_ExxT() * float(self.x2.pass_down_ExxT())
		elif self.x1.shape[0] == 1:#lhs is transposed vector (or hstacked scalar?)
			#print np.trace(np.dot(self.x2.pass_down_ExxT(),self.x1.pass_down_ExTx()))
			return np.trace(np.dot(self.x2.pass_down_ExxT(),self.x1.pass_down_ExTx()))
		else:
			raise NotImplementedError, "hstacks, transposes etc are not implememted yet"

class Constant(Node):
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
		Node.__init__(self,value.shape)
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