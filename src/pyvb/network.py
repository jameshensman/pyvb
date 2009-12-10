# -*- coding: utf-8 -*-
from nodes import *
class Network:
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
		self.find_iterable()
		print 'found' + str(len(self.iterable_nodes))+' iterable nodes'
		
		old_llb = -np.inf
		for i in range(niters):
			for n in self.iterable_nodes:
				n.update()
			llb = np.sum([n.log_lower_bound() for n in self.iterable_nodes])	
			
			print niters-i,llb
			#check for convergence
			if llb-old_llb < tol:
				break
			old_llb = llb
	
	def fetch_network(self):
		"""find all of the nodes connected to the nodes in the network"""
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
				
			
				
		
		
