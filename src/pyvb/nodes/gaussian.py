# -*- coding: utf-8 -*-
from node import *
from nodes_todo import *
import numpy as np

class Gaussian(Node):
	""" A node to model a Gaussian random variable
		
	Arguments
	----------
	dim : int
		description
	pmu : array or node  (arrays get wrapped in a Constant class)
		prior mean
	pprec : array or node 
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
		elif sum([isinstance(pmu,e) for e in [Gaussian, Addition, Multiplication, Constant]]):
			self.mean_parent = pmu
		else:
			raise ConjugacyError,'bad'
	
		#Deal with prior precision parent (pprec)
		assert pprec.shape == (self.shape[0],self.shape[0]), "Parent precision array has incorrect dimension"
		if type(pprec)==np.ndarray:
			self.precision_parent = Constant(pprec)
		elif isinstance(pprec,Constant):
			self.precision_parent = pprec
		elif isinstance(pprec,Gamma):
			self.precision_parent = pprec
		elif isinstance(pprec,DiagonalGamma):
			self.precision_parent = pprec
		elif type(pprec)==Wishart:
			raise NotImplementedError
			# TODO
		else:
			raise ConjugacyError
		
		self.mean_parent.addChild(self)
		self.precision_parent.addChild(self)
		
		self.observed=False
		
		#randomly initialize solution...
		self.qmu = np.random.randn(self.shape[0],1)
		self.qprec = np.eye(self.shape[0])*np.random.rand()
		self.qcov = np.linalg.inv(self.qprec)
	
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
		child_m2s = [e.pass_up_m2(self) for e in self.children]
		child_m1s = [e.pass_up_m1(self) for e in self.children]
		# here's the calculation
		self.qprec = pprec + sum(child_m1s) #that's it!
		self.qcov = np.linalg.inv(self.qprec)
		#weighted_exs = np.dot(pprec,pmu) + sum([np.dot(c,x) for x,c in zip(child_exs,child_precs)])
		weighted_exs = np.dot(pprec,pmu) + sum(child_m2s)
		self.qmu = np.dot(self.qcov,weighted_exs)
	
	def pass_down_Ex(self):
		"""Returns the Expected value of this node"""
		if self.observed:
			return self.obs_value
		else:
			return self.qmu
	    
	def pass_down_ExxT(self):
		"""Returns the expected value of the 'outer' product of this node.
		
		Notes
		----------
		<x x.T> = <x> <x>.T + \Sigma_x """
		if self.observed:
			return self.obs_xxT
		else:
			return np.dot(self.qmu,self.qmu.T) + self.qcov
			
	def pass_down_ExTx(self):
		"""Returns the expected value of the 'inner' product of this node.
		
		Notes
		----------
		<x.T x> = \textit{tr}(<x x.T>)
		"""
		if self.observed:
			return self.obs_xTx
		else:
			return np.trace(self.pass_down_ExxT())
	
	def pass_up_m1(self,requester):
		"""pass up the message 'm1'. See the doc for explanation."""
		#if self.observed:
			#return self.precision_parent.pass_down_Ex()
		#else:
		return self.precision_parent.pass_down_Ex()
		
	def pass_up_m2(self,requester):
		"""pass up the message 'm2'. See the doc for explanation."""
		if self.observed:
			return np.dot(self.precision_parent.pass_down_Ex(),self.obs_value)
		else:
			return np.dot(self.precision_parent.pass_down_Ex(),self.qmu)
			#return np.dot(self.qprec,self.qmu)
	    
