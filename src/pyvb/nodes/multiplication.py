import numpy as np
import node

class Multiplication(node.Node):
	def __init__(self,x1,x2):
		
		m1,n1 = x1.shape
		m2,n2 = x2.shape
		assert n1 == m2, "incompatible multiplication dimensions"
		assert n2 == 1, "right hand object must be a vector"
		node.Node.__init__(self, (m1,n2))
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
			return  np.dot(self.x1.pass_down_Ex().T,sumMu)
	
	
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
				return float(sumC)*self.x1.pass_down_ExTx()
			else:#lhs is a matrix.
				if isinstance(self.x1,Constant):
					return np.dot(self.x1.pass_down_Ex().T,np.dot(sumC,self.x1.pass_down_Ex()))
				else:
					raise NotImplementedError,"Objects with width (transposes, hstacks) not supported yet"
    
	def pass_down_Ex(self):
		return np.dot(self.x1.pass_down_Ex() , self.x2.pass_down_Ex())
	
	def pass_down_ExxT(self):
		if self.x1.shape[1] == 1:#rhs is scalar: this is quite easy
			return self.x1.pass_down_ExxT() * float(self.x2.pass_down_ExxT())
		elif self.x1.shape[0] == 1:#lhs is transposed vector (or hstacked scalar?)
			#print np.trace(np.dot(self.x2.pass_down_ExxT(),self.x1.pass_down_ExTx()))
			return np.trace(np.dot(self.x2.pass_down_ExxT(),self.x1.pass_down_ExTx()))
		else:
			raise NotImplementedError, "hstacks, transposes etc are not implememted yet"
