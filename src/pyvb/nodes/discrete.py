# -*- coding: utf-8 -*-

class Dirichlet:
	def __init__(self,alpha0):
		self.alpha0 = np.array(alpha0).flatten()
		self.dim = self.alpha0.size
	def pass_down_Ex(self):
		pass
	def update(self):
		pass
	def log_lower_bound(self):
		pass
	
class Mixture:
	def __init__(self,weights,components):
		"""
		Notes
		----------
		The mixture is not conjugate to anything: it must be an observed node.
		"""
		
		#Conjugacy check.
		if not isinstance(weights,(Dirichlet,nodes.Constant)):
			raise ConjugacyError, ""
		self.shape = components[0].shape
		for c in components:
			if not isinstance(c,(nodes.Gaussian,nodes.Addition,nodes.Multiplication)):
				raise ConjugacyError, ""
			if not c.shape == self.shape:
				raise ConjugacyError, "Mixture components must have the same dimensions."
		
			
		self.weight_parent = weights
		self.components = components
		
	def observe(self,data):
		assert data.shape = self.shape, ""
		self.obs_val = data
		
	def pass_up_m1_m2(self,requester):
		i = self.components.index(requester)
		
		
		