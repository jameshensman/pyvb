# -*- coding: utf-8 -*-
class network:
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
				
		
		
