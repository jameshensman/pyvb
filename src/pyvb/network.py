# -*- coding: utf-8 -*-
from nodes import *
class Network:
	"""A class to represent a Variational Bayesian Network. 
	
	Arguments
	----------
	nodes - list
		the nodes in the network
		
	Attributes
	----------
	nodes - list
		Pointers to instances of all the nodes contained in the network
		
	Notes
	----------
	It is possible to create a network by instantiating the class with a subset of the network's nodes and calling self.fetch_network(). This is rather slow for big networks. 
	
	Before learning, the network class creates a list of _iterable_ nodes, a subset of nodes with applicable update() and log_lower_bound() functions.  
	"""
		
	def __init__(self,nodes=[]):
		self.nodes = []
		[self.addnode(n) for n in nodes]
		
		
	def addnode(self,n):
		"""Add a node (or list of nodes) to the network"""
		if type(n) is list:
			self.nodes.extend(n)
		else:
			self.nodes.append(n)
			
	def find_iterable(self):
		"""make a list of all of the nodes which are to be updated"""
		self.iterable_nodes = [e for e in self.nodes if isinstance(e,Gaussian) or isinstance(e,Gamma) or isinstance(e,DiagonalGamma) or isinstance(e,Wishart)]
			
	
	def learn(self,niters,tol=1e-3):
		"""Iterate through the iterable nodeds in the network, updating each until the log_lower_bound converges or max_iter is reached.  """
		self.find_iterable()
		print 'Found' + str(len(self.iterable_nodes))+' iterable nodes\n'
		
		old_llb = -np.inf
		for i in range(niters):
			for n in self.iterable_nodes:
				n.update()
			self.llb = np.sum([n.log_lower_bound() for n in self.iterable_nodes])	
			
			print niters-i,self.llb
			#check for convergence
			if self.llb-old_llb < tol:
				print "Convergence!"
				break
			old_llb = self.llb
	
	def fetch_network(self):
		"""Find all of the nodes connected to the nodes in the network
		
		Notes
		---------
		This is rather slow at the moment"""
		N_starting_nodes = len(self.nodes)
		
		new_nodes = True
		while new_nodes:
			new_nodes = 0
			for n in self.nodes:
				if isinstance(n,Gaussian):
					new_children = [e for e in n.children if not e in self.nodes]
					new_parents = [e for e in [n.mean_parent, n.precision_parent] if not e in self.nodes]
					new_nodes += len(new_children)+len(new_parents)
					self.nodes.extend(new_children)
					self.nodes.extend(new_parents)
					
				elif sum([isinstance(n,Addition), isinstance(n,Multiplication)]):
					new_children = [e for e in n.children if not e in self.nodes]
					new_parents = [e for e in [n.A, n.B] if not e in self.nodes]
					new_nodes += len(new_children)+len(new_parents)
					self.nodes.extend(new_children)
					self.nodes.extend(new_parents)
					
				elif isinstance(n,hstack):
					new_children = [e for e in n.children if not e in self.nodes]
					new_parents = [e for e in n.parents if not e in self.nodes]
					new_nodes += len(new_children)+len(new_parents)
					self.nodes.extend(new_children)
					self.nodes.extend(new_parents)
					
				if sum([isinstance(n,Gamma), isinstance(n,DiagonalGamma), isinstance(n,Wishart)]):
					new_children = [e for e in n.children if not e in self.nodes]
					new_nodes += len(new_children)
					self.nodes.extend(new_children)
					
		print "Found "+str(len(self.nodes)-N_starting_nodes)+" new nodes."  
				
			
				
		
		
