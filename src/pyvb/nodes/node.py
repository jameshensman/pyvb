# -*- coding: utf-8 -*-
# Copyright 2009 James Hensman and Michael Dewar
# Licensed under the Gnu General Public license, see COPYING
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
	A : numpy.array or node
		the first node
	B : numpy.array or node
		the second node

	Attributes
	----------

	Notes
	----------
	numpy assarys are automagically wrapped in a Constant class for convenience

	See Also
	--------

	References
	----------

	Examples
	--------

	"""
	def __init__(self,A,B):
		assert A.shape == B.shape, "Bad shapes for addition"
		Node.__init__(self, A.shape)
		if type(A) == np.ndarray:
			self.A = Constant(A)
		else:
			self.A = A

		if type(B) == np.ndarray:
			self.B = Constant(B)
		else:
			self.B = B

		A.addChild(self)
		B.addChild(self)

	def pass_up_m1_m2(self,requester):
		"""Pass up both m1 and m2 messages
		TODO: proper docstring
		
		Arguments
		----------
		requester : node
			requester is either parent of this node (self.A or self.B)
		"""
		child_messages = [e.pass_up_m1_m2(self) for e in self.children]
		m1 = np.sum([e[0] for e in child_messages],0)
		if requester == self.A:
			m2 = np.sum([e[1] for e in child_messages],0) - np.dot(m1,self.B.pass_down_Ex())
		else:
			m2 = np.sum([e[1] for e in child_messages],0) - np.dot(m1,self.A.pass_down_Ex())
		return (m1,m2)
		
	def pass_down_Ex(self):
		""" Return the sum of the expected values of the parent nodes

		Notes
		----------
		<A+B> = <A> + <B>
		"""
		return self.A.pass_down_Ex()+self.B.pass_down_Ex()

	def pass_down_ExxT(self):
		"""Return the expected value of the 'outer procuct' of the sum of the parent nodes

		Notes
		----------
		$ <(A+B)(A+B)>^\top = <AA^\top> + <BB^\top> + <A><B>^\top + <B><A>^\top $""" 
		return self.A.pass_down_ExxT() + self.B.pass_down_ExxT() + np.dot(self.A.pass_down_Ex(),self.B.pass_down_Ex().T) + np.dot(self.B.pass_down_Ex(), self.A.pass_down_Ex().T)

class Multiplication(Node):
	"""creates a node by multiplying two other nodes together.  

	Arguments
	----------
	A : numpy.array or node
		the first node
	B : numpy.array or node
		the second node
		

	Attributes
	----------

	Notes
	----------
	numpy assarys are automagically wrapped in a Constant class for convenience

	See Also
	--------
	Addition

	References
	----------

	Examples
	--------
	A = nodes.Gaussian(1,np.random.randn(1,1),np.eye(1))
	B = nodes.Constant(np,random.randn(1,1)
	C = A*B# returns an instance if this class
	
	"""
	def __init__(self,A,B):
		m1,n1 = A.shape
		m2,n2 = B.shape
		assert n1 == m2, "incompatible multiplication dimensions"
		assert n2 == 1, "right hand object must be a vector"
		Node.__init__(self, (m1,n2))
		if type(A) == np.ndarray:
			self.A = Constant(A)
		else:
			self.A = A

		if type(B) == np.ndarray:
			self.B = Constant(B)
		else:
			self.B = B

		A.addChild(self)
		B.addChild(self)

	def pass_up_m1_m2(self,requester):
		"""Pass up both m1 and m2 messages to the requesting aprent
		
		Arguments
		----------
		requester : node
			requester is either parent of this node (self.A or self.B)"""
		child_messages = [e.pass_up_m1_m2(self) for e in self.children]
		sum_child_m1s = np.sum([e[0] for e in child_messages],0)
		sum_child_m2s = np.sum([e[1] for e in child_messages],0)
		
		if requester is self.A:
			BBT = self.B.pass_down_ExxT()
			if self.A.shape[1] == 1:#lhs is column: therefore rhs is scalar - easy enough
				m1 = sum_child_m1s*float(BBT)
				m2 =  float(self.B.pass_down_Ex())*sum_child_m2s
			elif self.A.shape[0] == 1:#lhs is a transposed vector (or hstacked scalars?)  
				m1 = float(sumC)*BBT
				m2 =  self.B.pass_down_Ex().T*float(sum_child_m2s)
			else: #lhs is a hstack matrix! Return a tuple and tel it deal with it.
				return sum_child_m1s, sum_child_m2s, self.B.pass_down_Ex(), self.B.pass_down_ExxT()
		else:#requester must be self.B
			m2 = np.dot(self.A.pass_down_Ex().T,sum_child_m2s)#that was easy :)
			if self.A.shape[1]==1:#lhs has only one column, rhs is scalar
				AAT = self.A.pass_down_ExxT()
				m1 = np.trace(np.dot(AAT,sum_child_m1s))
			elif self.A.shape[0] == 1:# lhs is transpose (or hstacked scalars) : therefore product is scalar
				return float(sum_child_m1s)*self.A.pass_down_ExTx()
			else:#lhs is a matrix - need to do <A.T * sumC * A>
				if isinstance(self.A,Constant):
					return np.dot(self.A.pass_down_Ex().T,np.dot(sum_child_m1s,self.A.pass_down_Ex()))
				else: #lhs must be a hstack?
					dim = self.B.shape[0]
					m1 = np.zeros((dim,dim))
					for i in range(dim):
						for j in range(dim):
							if i==j:
								m1[i,j] = np.trace(np.dot(self.A.parents[i].pass_down_ExxT(),sum_child_m1s))
							else:
								
								m1[i,j] = np.trace(np.dot(np.dot(self.A.parents[i].pass_down_Ex(),self.A.parents[j].pass_down_Ex().T),sum_child_m1s))
		
		return m1,m2
				
	def pass_up_m2(self,requester):
		""" Pass up the 'm2' message to the parent.

		Notes
		----------
		1) get the m2 message from the child(ren)
		2) modify my appropriate co-parent
		3) pass it up the network
		"""
		sum_m2 = sum([e.pass_up_m2(self) for e in self.children])
		if requester is self.A:
			if self.A.shape[1] == 1:#lhs is column: therefore rhs is scalar - easy enough
				return float(self.B.pass_down_Ex())*sum_m2
			elif self.A.shape[0] == 1:#lhs is a transposed vector (or hstacked scalars?)  
				return self.B.pass_down_Ex().T*float(sum_m2)
			else: #lhs is a hstack matrix! Return a tuple and tel it deal with it.
				return sum_m2,self.B.pass_down_Ex()
				
		elif requester is self.B:
			return  np.dot(self.A.pass_down_Ex().T,sum_m2)


	def pass_up_m1(self,requester):
		"""
		Pass up the 'm1' message to the requesting parent

		1) get m1 message from child(ren)
		2) modify it by the co-parent
		3) pass up."""
		sumC = sum([e.pass_up_m1(self) for e in self.children])
		if requester is self.A:
			BBT = self.B.pass_down_ExxT()# this must be scalar in this case?
			if self.A.shape[1]==1:# one column (rhs scalar): easy enough
				return sumC*float(BBT)
			elif self.A.shape[0] == 1:#lhs is a transpose (or hstack of scalars?)
				return float(sumC)*BBT
			else:#lhs is a matrix. pass up the data for the hstack (or simliar ) instance to deal with
				return sumC,self.B.pass_down_ExxT()
		elif requester is self.B:
			if self.A.shape[1]==1:#lhs has only one column, rhs is scalar
				AAT = self.A.pass_down_ExxT()
				return np.trace(np.dot(AAT,sumC))
			elif self.A.shape[0] == 1:# lhs is transpose (or hstacked scalars) : therefore product is scalar
				return float(sumC)*self.A.pass_down_ExTx()
			else:#lhs is a matrix.
				if isinstance(self.A,Constant):
					return np.dot(self.A.pass_down_Ex().T,np.dot(sumC,self.A.pass_down_Ex()))
				else: #lhs must be a hstack?
					#need to do <A.T * sumC * A>
					dim = self.B.shape[0]
					ret = np.zeros((dim,dim))
					for i in range(dim):
						for j in range(dim):
							if i==j:
								ret[i,j] = np.trace(np.dot(self.A.parents[i].pass_down_ExxT(),sumC))
							else:
								
								ret[i,j] = np.trace(np.dot(np.dot(self.A.parents[i].pass_down_Ex(),self.A.parents[j].pass_down_Ex().T),sumC))
					return ret
					
					

	def pass_down_Ex(self):
		"""Return the expected value of the product of the two parent nodes.
		
		Notes
		----------
		<AB> = <A><B>
		"""
		return np.dot(self.A.pass_down_Ex() , self.B.pass_down_Ex())

	def pass_down_ExxT(self):
		"""Return the Expected value of the 'outer' product of the product of the two parent nodes.
		
		Notes
		----------
		<(AB)(AB)^\top> = eek. TODO
		"""
		if self.A.shape[1] == 1:#rhs is scalar: this is quite easy
			return self.A.pass_down_ExxT() * float(self.B.pass_down_ExxT())
		elif self.A.shape[0] == 1:#lhs is transposed vector (or hstacked scalar?)
			return np.trace(np.dot(self.B.pass_down_ExxT(),self.A.pass_down_ExTx()))
		else:
			#lhs is matrix.
			if isinstance(self.A,Constant):
				return np.dot(self.A.pass_down_Ex(),np.dot(self.B.pass_downExxT(),self.A.pass_down_Ex()))
			else:#lhs is hstack
				BBT = self.B.pass_down_ExxT()
				dim = self.A.shape[0]
				ret = np.zeros((dim,dim))
				dim2 = BBT.shape[0]
				for i in range(dim2):
					for j in range(dim2):
						if i==j:
							ret += self.A.parents[i].pass_down_ExxT()*float(BBT[i,i])
						else:
							ret += np.dot(self.A.parents[i].pass_down_Ex(),self.A.parents[j].pass_down_Ex().T)*float(BBT[i,j])
				return ret

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