# -*- coding: utf-8 -*-
# Copyright 2009 James Hensman and Michael Dewar
# Licensed under the Gnu General Public license, see COPYING
import numpy as np

class ConjugacyError(ValueError):
	
	def __init__(self,message):
		ValueError.__init__(self,message)
	
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
	
	Notes
	----------
	Forgive him, Guido, for he knows not how to capitalise

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
		
	

#it's a shame we can;t override the __rmul__ command - when a numpy array gets multiplied by an object with an __rmul__, it passes each of it's elements to that rmul seperately. otherwise we could just do:
# def __rmul__(self,other):
#     return Multiplication(other,self)
#I wonder if there's a way to stop the numpy array from doing this (throw it an error?)
#
# brainwave - could add a 'constant' class. 

class Gaussian(Node):
	""" A node to model a Gaussian random variable
		
	Arguments
	----------
	dim : int
		description
	pmu : array or node  # TODO adding a 'Constant' class would make this less ambiguous. (I hate 'if' statements) 
		prior mean
	pprec : array or node # TODO as above
		prior precision matrix

	Attributes
	----------
	children : list
		a list of the children of this node
	observed : boolean
		flag to say if this node is observed
	qmu : 
		decription
	qprec : 
		description

	Notes
	----------

	See Also
	--------
	pyvb.node : parent class
	"""
	def __init__(self,dim,pmu,pprec):
		Node.__init__(self,(dim,1))
		#Deal with prior mu parent (pmu)
		assert pmu.shape==self.shape,"Parent node (or array) has incorrect dimension"
		if type(pmu)==np.ndarray:
			self.mean_parent = Constant(pmu)
		elif sum([isinstance(pmu,e) for e in [Gaussian, Addition, Multiplication,Constant]]):
			self.mean_parent = pmu
			pmu.addChild(self)
		else:
			raise ConjugacyError,'bad'
	
		#Deal with prior precision parent (pprec)
		assert pprec.shape == (self.shape[0],self.shape[0]), "Parent precision array has incorrect dimension"
		if type(pprec)==np.ndarray:
			self.precision_parent = Constant(pprec)
		elif isinstance(pprec,Gamma):
			self.precision_parent = pprec
			pprec.addChild(self)
		elif isinstance(pprec,DiagonalGamma):
			self.precision_parent = pprec
			pprec.addchild(self)
		elif type(prec)==Wishart:
			raise NotImplementedError
			# TODO
		else:
			raise ConjugacyError
		
		self.observed=False
		
		#randomly initialize solution...
		self.qmu = np.random.randn(self.shape[0],1)
		self.qprec = np.eye(self.shape[0])*np.random.rand()
	
	def observe(self,val):
		"""assigns an observation to the node
			
		Arguments
		----------
		val : numpy.array
			observation vector of the same dimension as the node
			
		Notes
		----------
		By providing an observation to this node, the attribute `observed`
		is set to True and is treated appropriately when updating etc.
		"""
		assert val.shape == self.shape,"Bad shape for observation data"
		self.observed = True
		self.obs_value = val
		self.obs_xxT = np.dot(val,val.T)
	
	def update(self):
		# don't update this node if it's an observed one
		if self.observed:
			return
		# get parent messages
		pmu = self.mean_parent.pass_down_Ex()
		pprec = self.precision_parent.pass_down_Ex()
		# get Child messages
		child_exs = [e.pass_up_ex(self) for e in self.children]
		child_precs = [e.pass_up_prec(self) for e in self.children]
		# here's the calculation
		self.qprec = pprec + sum(child_precs) #that's it?
		#weighted_exs = np.dot(pprec,pmu) + sum([np.dot(c,x) for x,c in zip(child_exs,child_precs)])
		weighted_exs = np.dot(pprec,pmu) + sum(child_exs)
		self.qmu = np.linalg.solve(self.qprec,weighted_exs)
	
	def pass_down_Ex(self):
		if self.observed:
			return self.obs_value
		else:
			return self.qmu
	    
	def pass_down_ExxT(self):
		if self.observed:
			return self.obs_xxT
		else:
			return np.dot(self.qmu,self.qmu.T) + np.linalg.inv(self.qprec)
			
	def pass_down_ExTx(self):
		if self.observed:
			return self.obs_xTx
		else:
			return np.trace(self.pass_down_ExxT())
	
	def pass_up_ex(self,requester):
		if self.observed:
			return np.dot(self.precision_parent.pass_down_Ex(),self.obs_value)
		else:
			return np.dot(self.qprec,self.qmu)
	    
	def pass_up_prec(self,requester):
		if self.observed:
			return self.precision_parent.pass_down_Ex()
		else:
			return self.qprec
	
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
			#return  np.dot(self.x1.pass_down_Ex().T,sumMu)
			return  np.dot(self.x1.pass_down_Ex(),sumMu) # TODO is this the source of the Kalman filter problem?
	
	
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
				return float(sumC)*self.get_x1Tx1()
			else:#lhs is a matrix.
				if isinstance(self.x1,Constant):
					return np.dot(self.x1.pass_down_Ex().T,np.dot(sumC,self.x1.pass_down_Ex()))
				else:
					raise NotImplementedError,"Objects with width (transposes, hstacks) not supported yet"
    
	def pass_down_Ex(self):
		return np.dot(self.x1.pass_down_Ex() , self.x2.pass_down_Ex())
	
	def pass_down_ExxT(self):
		if self.x1.shape[1] == 1:#rhs is scalar: this is quite easy
			return self.x.pass_down_ExxT() * float(self.x2.pass_down_ExxT())
		elif self.x1.shape[0] == 1:#lhs is transposed vector (or hstacked scalar?)
			print np.trace(np.dot(self.get_x2x2T(),self.get_x1Tx1()))
			return np.trace(np.dot(self.x2.pass_down_ExxT(),self.x1.pass_down_ExTx()))
		else:
			raise NotImplementedError, "hstacks, transposes etc are not implememted yet"

class hstack(Node):
	def __init__(self,parents):
		assert type(parents)==list
		dims = [e.shape[0] for e in parents]
		assert np.all(dims[0]==np.array(dims)),"dimensions incompatible"
		self.parents = parents
		shape = (dims[0],len(parents))
		Node.__init__(self, shape)
		
	def get_Ex(self):
		return np.hstack(e.get_Ex() for e in parents)
		
	def get_Exxt(self):
		return # TODO
	
class Transpose(Node):
	
	def __init__(self,parent):
		"""I'm designing this to sit between a Gaussian Node and a multiplication node (for inner products)"""
		assert isinstance(parent, Gaussian), "Can only transpose Gaussian Nodes..."
		self.parent = parent
		shape = self.parent.shape[::-1]
		Node.__init__(self, shape)
		parent.addChild(self)
	def pass_down_Ex(self):
		return self.parent.pass_down_Ex().T
	def pass_down_ExxT(self):
		return self.parent.pass_down_ExTx()
	def pass_down_ExTx(self):
		return self.parent.pass_down_ExxT()
	def pass_up_prec(self,requester):
		#get messages from the child node(s), undo the transpose nonsense, passup
		Csum = sum([c.pass_up_prec(self) for c in self.children])
		return Csum # TODO : check this?
	def pass_up_ex(self,requester):
		return sum([e.pass_up_ex(self) for e in self.children]).T
		
		
		
	
	    
class Gamma:
	"""
	A Class to represent a Gamma random variable in a VB network
	
	

	Gamma does not inherrit from Node because it cannot be added, muliplied etc"""
	def __init__(self,dim,a0,b0):
		self.shape = (dim,dim)
		self.a0 = a0
		self.b0 = b0
		self.children = []
		self.update_a()#initialise q to correct value
		self.qb = np.random.rand()#randomly initialise solution
	
	def addChild(self,child):
		self.children.append(child)
		self.update_a()#re-initialise q to correct value
		
	def update_a(self):
		self.qa = self.a0
		for child in self.children:
			self.qa += 0.5*child.shape[0]
    
	def update(self):
		"""update only the b parameter, since the a parameter can be done is closed form and does not need to be iterated. Note the use of trace() allows for children whose shape is not (1,1)"""
		self.qb = self.b0
		for child in self.children:
			self.qb += 0.5*np.trace(child.pass_down_ExxT()) + 0.5*np.trace(child.mean_parent.pass_down_ExxT()) - np.trace(np.dot(child.pass_down_Ex(),child.mean_parent.pass_down_Ex().T))
    
	def pass_down_Ex(self):
		return np.eye(self.shape[0])*self.qa/self.qb
		
class DiagonalGamma:
	"""A class to implemet a diagonal prior for a lumtivariate Gaussian. Effectively a series of Gamma distributions"""
	def __init__(self,dim,a0s,b0s):
		self.shape = (dim,dim)
		assert a0s.size==self.shape[0]
		assert b0s.size==self.shape[0]
		self.a0s = a0s.flatten()
		self.b0s = b0s.flatten()
		self.children = []
		self.update_a()#initialise q to correct value
		self.qb = np.random.rand()#randomly initialise solution
		
	def addchild(self,child):
		assert child.shape == (self.dim,1)
		self.children.append(child)
		self.update_a()
	def update_a(self):
		self.qa = self.a0
		for child in self.children:
			self.qa += 0.5
	def update(self):
		self.qb = self.qb0
		for child in self.children:
			self.qb += 0.5*np.diag(child.pass_down_ExxT()) + 0.5*np.diag(child.get_pExxT()) - np.diag(np.dot(child.pass_down_Ex(),child.get_pmu().T))
		
	def pass_down_Ex(self):
		return np.diag(self.qa/self.qb)
	    
class Wishart:
	def __init__(self,dim,v0,w0):
		self.dim = dim
		assert w0.shape==(self.dim,self.dim)
		self.v0 = v0
		self.w0 = w0
		self.children = []
		self.update_v()#initialise qv to correct value
		#randomly initialise solution (for qw)
		l = np.random.randn(self.dim,1)#randomly initialise solution
		self.qw = np.dot(l,l.T)
		
	def addchild(self,child):
		assert child.shape == (self.dim,1)
		self.children.append(child)
		self.update_a()
	def update_a(self):
		self.qv = self.a0
		for child in self.children:
			self.qa += 0.5
	def update(self):
		self.qb = self.qb0
		for child in self.children:
			self.qb += 0.5*child.pass_down_ExxT() + 0.5*child.get_pExxT() - np.dot(child.pass_down_Ex(),child.get_pmu().T)
		
	def pass_down_Ex(self):
		return # TODO
	    
	
	
	
	
	
	