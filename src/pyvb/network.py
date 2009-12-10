# -*- coding: utf-8 -*-
from nodes import *
class Network:
	def __init__(self,nodes=[]):
		self.nodes = []
		[self.addnode(n) for n in nodes]
		
		
	def addnode(self,n):
		"""Add a node (or list of nodes) to the network"""
		for n in nodes:
			if type(n) is list:
				self.nodes.extend(n)
			else:
				self.nodes.append(n)
		
			
	
	def learn(self,niters):
		for i in range(niters):
			for n in self.nodes:
				n.update()
	
	def fetch_network(self):
		"""find all of the nodes connected to the nodes in the network"""
		N_starting_nodes = len(self.nodes)
		
		new_nodes = True
		while new_nodes:
			new_nodes = 0
			for n in self.nodes:
				if sum(isinstance(n,Gaussian), isinstance(n,Addition), isinstance(n,Multiplication)):
					new_children = [e for e in n.children if not e in self.nodes]
					new_parents = [e for e in [n.mean_parent, precision_parent] if not e in self.nodes]
					new_nodes += len(new_children)+len(new_parents)
					self.nodes.extend(new_children)
					self.nodes.extend(new_parents)
				if sum(isinstance(n,Gamma), isinstance(n,DiagonalGamma), isinstance(wishart)):
					new_children = [e for e in n.children if not e in self.nodes]
					new_nodes += len(new_children)
					self.nodes.extend(new_children)
					
		print "Found "+str(len(self.nodes)-N_starting_nodes)+" new nodes."  
				
			
				
		
		
