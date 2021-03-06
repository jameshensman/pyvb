# -*- coding: utf-8 -*-
# Copyright 2009 James Hensman and Michael Dewar
# Licensed under the Gnu General Public license, see COPYING
from node import *
from nodes_todo import *
import numpy as np
from scipy.linalg import cho_factor, cho_solve

class Gaussian(Node):
	""" A node to model a Gaussian random variable
		
	Arguments
	----------
	dim : int
		The dimensionality of the node
	pmu : array or node  (arrays get wrapped in a Constant class)
		'prior' mean
	pprec : array or node 
		'prior' precision matrix

	Attributes
	----------
	children : list
		a list of the children of this node
	observed : boolean
		flag to say if this node is observed
	partially_observed : boolean
		flag to say if this node is _partially_ observed
	qmu : 
		Mean of the variational posterior
	qcov : 
		Covariance of the variational posterior
		

	Notes
	----------

	See Also
	--------
	pyvb.Node : parent class
	pyvb.nodes.Addition, Multiplication, Hstack: Operation classes for this node.
	"""
	def __init__(self,dim,pmu,pprec):
		Node.__init__(self,(dim,1))
		#Deal with prior mu parent (pmu)
		assert pmu.shape==self.shape,"Parent node (or array) has incorrect dimension"
		if type(pmu)==np.ndarray:
			self.mean_parent = Constant(pmu)
		elif isinstance(pmu,(Gaussian, Addition, Multiplication, Constant)):
			self.mean_parent = pmu
		else:
			raise ConjugacyError,"mean parent for a Gaussian node should be one of:\nGaussian\nConstant\nAddition\nMultiplication\nnumpy array. \n\n"+str(type(pmu))+" is invalid"
	
		#Deal with prior precision parent (pprec)
		assert pprec.shape == (self.shape[0],self.shape[0]), "Parent precision array has incorrect dimension"
		if type(pprec)==np.ndarray:
			self.precision_parent = Constant(pprec)
		elif sum([isinstance(pprec,e) for e in Gamma,DiagonalGamma,Wishart,Constant]):
			self.precision_parent = pprec
		else:
			raise ConjugacyError,"Precision parent for a Gaussian node should be one of:\nGamma\nDiagonalGamma\nWishart\nConstant\nnumpy array. \n\n"+str(type(pprec))+" is invalid"
		
		self.mean_parent.addChild(self)
		self.precision_parent.addChild(self)
		
		self.observed=False
		self.partially_observed=False
		
		#randomly initialize solution...
		self.qmu = np.random.randn(self.shape[0],1)
		self.qprec = np.eye(self.shape[0])*np.random.rand()
		self.qcov = np.linalg.inv(self.qprec)
	
	def observe(self,val):
		"""Assigns an observation to the node.
			
		Arguments
		----------
		val : numpy.array
			observation vector of the same dimension as the node.  
			NaN parts val are treated as missing data
			
		Notes
		----------
		By providing an observation to this node, the attribute `observed`
		is set to True and is treated appropriately when updating etc.
		"""
		assert val.shape == self.shape,"Bad shape for observation data"
		
		if np.isnan(val).all():#there are no actual data - it's an array of nans...
			return
		elif np.isnan(val).any():#some missing data
			self.partially_observed = True
			self.obs_value = val
			self.obs_index = np.nonzero(1-np.isnan(val))[0]
			self.missing_index = np.nonzero(np.isnan(val))[0]
		else:#full observation
			self.observed = True
			self.qmu = val
			self.qcov = np.zeros(self.qcov.shape)
	
	def update(self):
		"""Update the (approximate, variational) posterior distribution of this node using VB messag passing
		
		Notes
		----------
		"""
		# don't update this node if it's an observed one
		if self.observed:
			return
		# get parent messages
		pmu = self.mean_parent.pass_down_Ex()
		pprec = self.precision_parent.pass_down_Ex()
		# get Child messages
		child_messages = [e.pass_up_m1_m2(self) for e in self.children]
		# here's the calculation
		qprec = pprec + np.sum([e[0] for e in child_messages],0) #that's it!
		qprec_chol = cho_factor(qprec)
		self.qcov = cho_solve(qprec_chol,np.eye(self.shape[0]))
		self.q_ln_det = .5/np.log(np.prod(np.diag(qprec_chol[0])))

		weighted_exs = np.dot(pprec,pmu) + np.sum([e[1] for e in child_messages],0)
		self.qmu = np.dot(self.qcov,weighted_exs)
		
		if self.partially_observed:
			#calculate the posterior probability of q given the observations.
			#find covariance of observed parts
			cov_obs = self.qcov.take(self.obs_index,0).take(self.obs_index,1)
			cov_obs_inv = np.linalg.inv(cov_obs)
			#find covariance between observed data and all data:
			cov_obs_all = self.qcov.take(self.obs_index,1)
			#calculate marginals and set into qcov,qmu and qprec
			self.qmu += np.dot(cov_obs_all,np.dot(cov_obs_inv,self.obs_value[self.obs_index,:]-self.qmu[self.obs_index,:]))
			self.qcov -= np.dot( cov_obs_all,np.dot(cov_obs_inv,cov_obs_all.T))
			
	def log_lower_bound(self):
		"""Calculate and return this node's contribution to the lower bound of the log of the model evidence
		"""
		
		#expected value of joint probability
		ret = -0.5*self.shape[0]*np.log(2*np.pi) \
			+ 0.5*self.precision_parent.pass_down_lndet()\
			-0.5*np.trace(np.dot(self.precision_parent.pass_down_Ex(),self.pass_down_ExxT() + self.mean_parent.pass_down_ExxT() \
			-2* np.dot(self.qmu,self.mean_parent.pass_down_Ex().T)) ) 
		if not (self.observed or self.partially_observed):
			#relative entropy from prior
			ret -= -0.5*self.shape[0]*np.log(2*np.pi) - 0.5*self.q_ln_det - 0.5*self.shape[0]
		elif self.partially_observed:
			#relative entropy of unobserved part
			ret -= 0.5*len(self.missing_index)*np.log(2*np.pi) -0.5*np.log(np.linalg.det(self.qcov.take(self.missing_index,0).take(self.missing_index,1))) - 0.5*len(self.missing_index)
		return ret	
			
	
	def pass_down_Ex(self):
		"""Returns the Expected value of this node.  
		
		Notes
		----------
		If the node has been observed, the expected value has been set to the observation value"""
		return self.qmu 
	    
	def pass_down_ExxT(self):
		"""Returns the expected value of the 'outer' product of this node.
		
		Notes
		----------
		<x x.T> = <x> <x>.T + \Sigma_x """
		return np.dot(self.qmu,self.qmu.T) + self.qcov
			
	def pass_down_ExTx(self):
		"""Returns the expected value of the 'inner' product of this node.
		
		Notes
		----------
		<x.T x> = \textit{tr}(<x x.T>)
		"""
		return np.trace(self.pass_down_ExxT())
	
	def pass_up_m1_m2(self,requester):
		"""
		Pass up both messages m1 and m2"""
		pp =  self.precision_parent.pass_down_Ex()
		return (pp,np.dot(pp,self.qmu))
		
class DiagonalGaussian(Gaussian):
	"""A class to represent a diagonal matrix, whose (diagonal) values are jointly Gaussian distributed
	
	Notes
	----------
	The qmu and qprec values are kept as for a normal Gaussian Class.
	
	See Also
	----------
	nodes.Gaussian """
	def __init__(self,dim,pmu,pprec):
		Gaussian.__init__(self,dim,pmu,pprec)
		self.shape = (self.shape[0],self.shape[0])
	def pass_down_Ex(self):
		return np.diag(self.qmu[:,0])
	def pass_down_ExxT(self):
		return np.diag(self.qmu[:,0]*self.qmu[:,0] + np.diag(self.qcov)) 
	def pass_down_ExTx(self):
		return self.pass_down_ExxT()
		
		
	    
