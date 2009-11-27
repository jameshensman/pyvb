# -*- coding: utf-8 -*-
# Copyright 2009 James Hensman
# Licensed under the Gnu General Public license, see COPYING
import numpy as np

class ConjugacyError(ValueError):
	"""This does very little at the moment"""
	def __init__(self,message):
		ValueError.__init__(self,message)
	
class node:
	def __init__(self):
		pass
	def addChild(self,other):
		self.children.append(other)
	def __add__(self,other):
		return Addition(self,other)
	def __mul__(self,other):
		return Multiplication(self,other)

#it's a shame we can;t override the __rmul__ command - when a numpy array gets multiplied by an object with an __rmul__, it passes each of it's elements to that rmul seperately. otherwise we could just do:
# def __rmul__(self,other):
#     return Multiplication(other,self)
#I wonder if there's a way to stop the numpy array from doing this (throw it an error?)
#
# brainwave - could add a 'constant' class. 
    
class Gaussian(node):
	def __init__(self,dim,pmu,pprec):
		node.__init__(self)
		self.shape = (dim,1)
	
		#Deal with prior mu parent (pmu)
		if type(pmu)==np.ndarray:
			assert pmu.shape==self.shape,"Parent mean array has incorrect dimension"
			self.get_pmu = lambda :pmu
			self.get_pExxT = lambda : np.dot(pmu,pmu.T)
		elif sum([isinstance(pmu,e) for e in [Gaussian, Addition, Multiplication]]):
			assert pmu.shape==self.shape,"Parent mean node has incorrect dimension"
			self.get_pmu = pmu.pass_down_Ex
			self.get_pExxT = pmu.pass_down_ExxT
			pmu.addChild(self)
		else:
			raise ConjugacyError,'bad'
	
		#Deal with prior precision parent (pprec)
		if type(pprec)==np.ndarray:
			assert pprec.shape == (self.shape[0],self.shape[0]),"Parent precision array has incorrect dimension"
			self.get_pprec = lambda :pprec
		elif isinstance(pprec,Gamma):
			self.get_pprec = lambda : np.eye(self.shape[0])*pprec.get_Ex()
			pprec.addChild(self)
		elif isinstance(pprec,DiagonalGamma):
			self.get_pprec = pprec.get_Ex
			assert pprec.dim==self.shape[0]
			pprec.addchild(self)
		elif type(prec)==Wishart:
			raise NotImplementedError
			# TODO
		else:
			raise ConjugacyError
	
		self.children = []
		self.observed=False
		
		#randomly initialize solution...
		self.qmu = np.random.randn(self.shape[0],1)
		self.qprec = np.eye(self.shape[0])*np.random.rand()
	
	def observe(self,val):
		assert val.shape == self.shape,"Bad shape for observation data"
		self.observed=True
		self.obs_value = val
		self.obs_xxT = np.dot(val,val.T)
	
	def update(self):
		if self.observed:
			return # don't update this node if it's an observed one
		#get parent messages
		pmu = self.get_pmu()
		pprec = self.get_pprec()
		#get Child messages
		child_exs = [e.pass_up_ex(self) for e in self.children]
		child_precs = [e.pass_up_prec(self) for e in self.children]
		#here's the calculation
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
	
	def pass_up_ex(self,requester):
		if self.observed:
			return np.dot(self.get_pprec(),self.obs_value)
		else:
			return np.dot(self.qprec,self.qmu)
	    
	def pass_up_prec(self,requester):
		if self.observed:
			return self.get_pprec()
		else:
			return self.qprec
	
class Addition(node):
	def __init__(self,x1,x2):
		node.__init__(self)
		self.x1 = x1
		self.x2 = x2
		self.children = []
		assert x1.shape == x2.shape, "Bad shapes for addition"
		self.shape = x1.shape
		if type(x1) == np.ndarray:
			self.get_x1 = lambda : x1
		else:
			self.get_x1 = self.x1.pass_down_Ex
			x1.addChild(self)
		if type(x2) == np.ndarray:
			self.get_x2 = lambda : x2
		else:
			self.get_x2 = self.x2.pass_down_Ex
			x2.addChild(self)
	    
	def pass_up_ex(self,requester):
		"""return the sum of the expected value of the child nodes, minus the expected value of the co-parent"""
		sumMu = sum([e.pass_up_ex(self) for e in self.children])
		sumC = sum([e.pass_up_prec(self) for e in self.children])
		if requester is self.x1:
			return sumMu - np.dot(sumC,self.get_x2())
		elif requester is self.x2:
			return sumMu - np.dot(sumC,self.get_x1())
	
	def pass_up_prec(self,requester):
		#get prec from children to pass upwards
		sumC = sum([e.pass_up_prec(self) for e in self.children])
		return sumC
	
	def pass_down_Ex(self):
		return self.get_x1()+self.get_x2()
    
	def pass_down_ExxT(self):
		Ex1,Ex2 = self.get_x1() , self.get_x2()
		return self.x1.pass_down_ExxT()+self.x2.pass_down_ExxT() + np.dot(Ex1,Ex2.T) + np.dot(Ex2,Ex1.T)
	

class Multiplication(node):
	def __init__(self,x1,x2):
		node.__init__(self)
		m1,n1 = x1.shape
		m2,n2 = x2.shape
		assert n1 == m2, "incompatible multiplication dimensions"
		assert n2 == 1, "right hand object must be a vector"
		self.shape = (m1,n2)
		self.x1 = x1
		self.x2 = x2
		self.children = []
		if type(x1) == np.ndarray:
			self.get_x1 = lambda : x1
			self.get_x1x1T = lambda : np.dot(x1,x1.T)
			self.get_x1Tx1 = lambda : np.dot(x1.T,x1)
		else:
			self.get_x1 = x1.pass_down_Ex
			self.get_x1x1T = x1.pass_down_ExxT
			self.get_x1Tx1 = x1.pass_down_ExTx
			x1.addChild(self)
		if type(x2) == np.ndarray:
			self.get_x2 = lambda : x2
			self.get_x2x2T = lambda : np.dot(x2,x2.T)
		else:
			self.get_x2 = x2.pass_down_Ex
			self.get_x2x2T = x2.pass_down_ExxT
			x2.addChild(self)
    
	def pass_up_ex(self,requester):
		""""""
		sumMu = sum([e.pass_up_ex(self) for e in self.children])
		#sumC = sum([e.pass_up_prec(self) for e in self.children]) - not needed here
		if requester is self.x1:
			if self.x1.shape[1] == 1:#lhs is column: therefore rhs is scalar: easy enough
				return float(self.get_x2())*sumMu
			elif self.x1.shape[0] == 1:#lhs is a transposed vector (or hstacked scalars?)  
				return self.get_x2().T*float(sumMu)
			else: #lhs is a matrix!
				raise NotImplementedError,"Hstack objects not done yet"
		elif requester is self.x2:
			return  np.dot(self.get_x1().T,sumMu)
	
	
	def pass_up_prec(self,requester):
		"""get prec from children to pass upwards, modify it by the co-parent"""
		sumC = sum([e.pass_up_prec(self) for e in self.children])
		if requester is self.x1:
			if self.x1.shape[1]==1:# one column (rhs scalar): easy enough
				x2x2T = self.get_x2x2T()# this must be scalar in this case?
				return sumC*float(x2x2T)
			elif self.x1.shape[0] == 1:#lhs is a transpose (or hstack of scalars?)
				return float(sumC)*self.get_x2x2T()
			else:
				raise NotImplementedError,"Objects with width (transposes, hstacks) not supported yet"
		elif requester is self.x2:
			if self.x1.shape[1]==1:#left object has only one column...
				x1x1T = self.get_x1x1T()
				return np.trace(np.dot(x1x1T,sumC))
			elif self.x1.shape[0] == 1:# lhs is transpose (or hstacked scalars)
				return float(sumC)*self.get_x1Tx1()
			else:#lhs is a matrix.
				if type(self.x1) is np.ndarray:
					return np.dot(self.get_x1().T,np.dot(sumC,self.get_x1()))
				raise NotImplementedError,"Objects with width (transposes, hstacks) not supported yet"
    
	def pass_down_Ex(self):
		return np.dot(self.get_x1() , self.get_x2())
	
	def pass_down_ExxT(self):
		if self.x1.shape[1] == 1:#rhs is scalar: this is quite easy
			return self.get_x1x1T() * float(self.get_x2x2T())
		elif self.x1.shape[0] == 1:#lhs is transposed vector (or hstacked scalar?)
			print np.trace(np.dot(self.get_x2x2T(),self.get_x1Tx1()))
			return np.trace(np.dot(self.get_x2x2T(),self.get_x1Tx1()))
		else:
			raise NotImplementedError, "hstacks, transposes etc are not implememted yet"

class hstack(node):
	def __init__(self,parents):
		node.__init__(self)
		assert type(parents)==list
		dims = [e.shape[0] for e in parents]
		assert np.all(dims[0]==np.array(dims)),"dimensions incompatible"
		self.parents = parents
		self.shape = (dims[0],len(parents))
		self.children = []
	def get_Ex(self):
		return np.hstack(e.get_Ex() for e in parents)
	def get_Exxt(self):
		return # TODO
	
class Transpose(node):
	def __init__(self,parent):
		"""I'm designing this to sit between a Gaussian Node and a multiplication node (for inner products)"""
		assert isinstance(parent, Gaussian), "Can only transpose Gaussian Nodes..."
		self.parent = parent
		self.shape = self.parent.shape[::-1]
		self.children = []
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
	"""Gamma does not inherrit from node because it cannot be added, muliplied etc"""
	def __init__(self,a0,b0):
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
		"""update only the b parameter..."""
		self.qb = self.b0
		for child in self.children:
			self.qb += 0.5*np.trace(child.pass_down_ExxT()) + 0.5*np.trace(child.get_pExxT()) - np.trace(np.dot(child.pass_down_Ex(),child.get_pmu().T))
    
	def get_Ex(self):
		return self.qa/self.qb
	
class DiagonalGamma:
	def __init__(self,dim,a0s,b0s):
		self.dim = dim
		assert a0s.size==self.dim
		assert b0s.size==self.dim
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
		
	def get_Ex(self):
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
		
	def get_Ex(self):
		return 
	    
	
	
	
	
	
	