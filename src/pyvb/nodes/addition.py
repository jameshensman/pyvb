import numpy as np
import node

class Addition(Node):
	"""creates a node by adding two other nodes together (or adding a np array to a node).  
		
	Arguments
	----------
	x1 : numpy.array or node
		the first node
	x2 : numpy.array or node
		the second node

	Attributes
	----------

	Notes
	----------

	See Also
	--------

	References
	----------

	Examples
	--------

	"""
	def __init__(self,x1,x2):
		assert x1.shape == x2.shape, "Bad shapes for addition"
		Node.__init__(self, x1.shape)
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
		"""Return the 'mu' update message of the child node, modified by the co-parent
		expected value of the co-parent
		
		Arguments
		----------
		requester : node
			requester is either parent of this node
		"""
		sumMu = sum([e.pass_up_ex(self) for e in self.children])
		sumC = sum([e.pass_up_prec(self) for e in self.children])
		if requester is self.x1:
			return sumMu - np.dot(sumC,self.x2.pass_down_Ex())
		elif requester is self.x2:
			return sumMu - np.dot(sumC,self.x1.pass_down_Ex())
	
	def pass_up_prec(self, requester):
		"""return the sum of the precision matrices for the children of this
		node. - This is the 'prec' update message to the parent node.
		"""
		# TODO aint no dependence on the argument 'requester' 
		#careful - pass_up prec is common to all (Gaussian-like) nodes, and some of them need to know the requester.
		
		#get prec from children to pass upwards
		sumC = sum([e.pass_up_prec(self) for e in self.children])
		return sumC
	
	def pass_down_Ex(self):
		""" Return the sum of the expected values of the parent nodes
		
		Notes
		----------
		<A+B> = <A> + <B>
		"""
		return self.x1.pass_down_Ex()+self.x2.pass_down_Ex()
    
	def pass_down_ExxT(self):
		"""Return the expected value of the 'outer procuct' of the sum of the parent node
		
		Notes
		----------
		Is this how I use latex, Mike?
		$ <(A+B)(A+B)>^\top = <AA^\top> + <BB^\top> + <A><B>^\top + <B><A>^\top $""" 
		return self.x1.pass_down_ExxT() + self.x2.pass_down_ExxT() + np.dot(self.x1.pass_down_Ex(),self.x2.pass_down_Ex().T) + np.dot(self.x2.pass_down_Ex(), self.x1.pass_down_Ex().T)
