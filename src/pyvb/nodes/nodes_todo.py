# -*- coding: utf-8 -*-
# Copyright 2009 James Hensman and Michael Dewar
# Licensed under the Gnu General Public license, see COPYING
import numpy as np

class ConjugacyError(ValueError):
	
	def __init__(self,message):
		ValueError.__init__(self,message)	
	
class hstack(Node):
	def __init__(self,parents):
		assert type(parents)==list
		dims = [e.shape[0] for e in parents]
		assert np.all(dims[0]==np.array(dims)),"dimensions incompatible"
		self.parents = parents
		shape = (dims[0],len(parents))
		Node.__init__(self, shape)
		
	def pass_down_Ex(self):
		return np.hstack(e.pass_down_Ex() for e in parents)
		
	def pass_down_ExxT(self):
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
	    
	
	
	
	
	
	