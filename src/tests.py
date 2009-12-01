# -*- coding: utf-8 -*-
# Copyright 2009 James Hensman
# Licensed under the Gnu General Public license, see COPYING
import numpy as np
from scipy import linalg
import pylab
from pyvb import nodes

def simple_mean_inference():
	truemean = 7.2
	trueprec = 10
	N = 20
	y = np.random.randn(N)*np.sqrt(1./trueprec) + truemean
	mu = nodes.Gaussian(1,np.array([[0]]),np.array([[1e-3]]))
	y_nodes = [nodes.Gaussian(1,mu,np.array([[trueprec]])) for i in range(N)]
	for yy,n in zip(y,y_nodes):
		n.observe(yy.reshape(1,1))
	mu.update()
	print mu.qmu, truemean
	
def scalar_addition():
	A1 = nodes.Gaussian(1,np.array([[0]]),np.array([[0.01]]))
	A2 = nodes.Gaussian(1,np.array([[0]]),np.array([[0.01]]))
	A3 = nodes.Gaussian(1,np.array([[0]]),np.array([[0.01]]))
	C = nodes.Gaussian(1,A1+A2+A3,np.array([[10]]))
    
	C.observe(np.array([[12]]))
    
	for i in range(5000):
		A1.update()
		A2.update()
		A3.update()
		print np.hstack((A1.qmu,A2.qmu,A3.qmu)).flatten()
		
def scalar_multiplication():
	A1 = nodes.Gaussian(1,np.array([[0]]),np.array([[0.001]]))
	A2 = nodes.Gaussian(1,np.array([[0]]),np.array([[0.001]]))
	C = nodes.Gaussian(1,A1*A2,np.array([[10]]))
    
	C.observe(np.array([[16]]))
    
	for i in range(5000):
		A1.update()
		A2.update()
		print np.hstack((A1.qmu,A2.qmu)).flatten()

def inner_product():
	"""tests the transpose class and the multiplication class"""
	A1 = nodes.Gaussian(3,np.zeros((3,1)),np.eye(3)*0.01)
	A2 = nodes.Gaussian(3,np.zeros((3,1)),np.eye(3)*0.01)
	C = nodes.Gaussian(1,nodes.Transpose(A1)*A2,np.eye(1)*10)
	C.observe(np.array([[3]]))
	for i in range(5000):
		A1.update()
		A2.update()
		print np.hstack((A1.qmu,A2.qmu)).flatten()
	
def vector_addition():
	A1 = nodes.Gaussian(3,np.zeros((3,1)),np.eye(3)*0.01)
	A2 = nodes.Gaussian(3,np.zeros((3,1)),np.eye(3)*0.01)
	A3 = nodes.Gaussian(3,np.zeros((3,1)),np.eye(3)*0.01)
	C = nodes.Gaussian(3,A1+A2+A3,np.eye(3)*10)
	
	C.observe(np.array([[12],[6],[3]]))
    
	for i in range(5000):
		A1.update()
		A2.update()
		A3.update()
		print np.hstack((A1.qmu,A2.qmu,A3.qmu)).flatten()
	
	print '\n',A1.qprec
    
def multiplication_of_observed():
	A1 = nodes.Gaussian(1,np.zeros((1,1)),np.eye(1)*1.)
	A2 = nodes.Gaussian(1,np.zeros((1,1)),np.eye(1)*1.)
	A1.observe(np.array([[2]]))
	A2.observe(np.array([[3]]))
	C = nodes.Gaussian(1,A1*A2,np.eye(1)*0.01)
	for i in range(5):
		C.update()
		print C.qmu
    
def multiplication_by_matrix():
	q = 2
	d = 2
	W = np.random.randn(d,q)
	A = nodes.Gaussian(q,np.zeros((q,1)),np.eye(q)*0.01)
	m = nodes.Multiplication(W,A)
	B = nodes.Gaussian(d,m,np.eye(d)*100)
	B.observe(np.random.randn(d,1))
    
	for i in range(5):
		A.update()
		print A.qmu
	
    
    
	
def simple_regression():
	N = 200
	x = np.linspace(-1,1,N).reshape(N,1)
	atrue = 0.7
	btrue = 0.3
	prec_true = 10.
	y = atrue*x + btrue + np.random.randn(N,1)*np.sqrt(1./prec_true)
	
	#vreate variable nodes
	B = nodes.Gaussian(1,np.array([[0]]),np.array([[1e-2]]))#Gaussian node with 'fixed' wide prior
	A = nodes.Gaussian(1,np.array([[0]]),np.array([[1e-2]]))
	noise = nodes.Gamma(1,1e-3,1e-3)
	
	#create a list/plate of constant nodes
	Xs = [nodes.Constant(xx.reshape(1,1)) for xx in x]
		
	#create (plate of) observation nodes
	Ys = []
	for Xnode in Xs:
		Ys.append(nodes.Gaussian(1,Xnode*A + B,noise))
	for n,yy in zip(Ys,y):
		n.observe(yy.reshape(1,1))
		
	#inference 
	for i in range(10):
		A.update()
		B.update()
		noise.update()
		print np.hstack((A.qmu,B.qmu)).flatten(),noise.pass_down_Ex(), atrue, btrue, prec_true
	
	#sample from posterior:
	Nsamples = 100
	Asamples = np.random.randn(Nsamples,1)*np.sqrt(1./A.qprec) + A.qmu[0]
	Bsamples = np.random.randn(Nsamples,1)*np.sqrt(1./B.qprec) + B.qmu[0]
	
	#plotting
	xxlin = np.linspace(-1.2,1.2,200)
	pylab.figure()
	for a,b in zip(Asamples,Bsamples):
		pylab.plot(xxlin,xxlin*a+b,'b',alpha=0.2)
	pylab.plot(x,y,'r.')
	pylab.show()
	
	
def multivariate_regression():
	N = 500
	dimx = 3 # dimy is one for the moment - not implemented hstacks yet
	xdata = np.random.rand(N,dimx)
	atrue = np.random.randn(dimx,1)
	btrue = np.random.randn(1,1)
	prec_true = 10.
	ydata = np.dot(xdata,atrue) + btrue + np.random.randn(N,1)*np.sqrt(1./prec_true)
	
	#vreate variable nodes
	A = nodes.Gaussian(dimx,np.zeros((dimx,1)),np.eye(dimx)*0.001)#Gaussian node with 'fixed' wide prior
	B = nodes.Gaussian(1,np.zeros((1,1)),np.eye(1)*0.001)#Gaussian node with 'fixed' wide prior
	noise = nodes.Gamma(1,1e-3,1e-3)
	
	#create (plate of) observation nodes
	Ys = []
	for xx in xdata:
		#m = nodes.Multiplication(xx.reshape(1,dimx),A)
		m = nodes.Multiplication(nodes.Transpose(A),xx.reshape(dimx,1))
		Ys.append(nodes.Gaussian(1,m+B,noise))
	for n,yy in zip(Ys,ydata):
		n.observe(yy.reshape(1,1))
		
	#inference 
	for i in range(100):
		A.update()
		B.update()
		noise.update()
		print np.hstack((A.qmu.T,B.qmu)).flatten(),noise.get_Ex(), atrue.flatten(), btrue, prec_true
	
		
		
def simple_PCA():
	#Set up a PCA problem. Latent dim must be one for the moment...
	N = 100
	d = 8       
	prec_true = 100.
	Z_true = np.random.randn(N,1)
	W_true = np.random.randn(d,1)
	mu_true = np.random.randn(d,1)
	X = np.dot(Z_true,W_true.T) + mu_true.T + np.random.randn(N,d)*np.sqrt(1./prec_true)
	
	#create nodes
	noise = nodes.Gamma(d,1e-3,1e-3)
	W = nodes.Gaussian(d,np.zeros((d,1)),np.eye(d)*0.001)
	Mu = nodes.Gaussian(d,np.zeros((d,1)),np.eye(d)*0.001)
	Zs = [nodes.Gaussian(1,np.zeros((1,1)),np.eye(1)) for i in range(N)]
	mults = [nodes.Multiplication(W,z) for z in Zs]
	Xs = [nodes.Gaussian(d,m+Mu,noise) for m in mults]
	[n.observe(v.reshape(d,1)) for n,v in zip(Xs,X)]
	
	#inference
	for i in range(10):
		W.update()
		[e.update() for e in Zs]
		Mu.update()
		noise.update()
		print np.hstack((np.linalg.qr(W.qmu)[0],np.linalg.qr(W_true)[0]))
	print '\n',noise.qa/noise.qb,prec_true
	
def mean_and_variance_inference():
	N = 100
	truemu = 1.23
	true_prec = 5.
	Xdata = np.random.randn(N,1)*np.sqrt(1./true_prec) + truemu
	prec = nodes.Gamma(1,1e-3,1e-3)
	mu = nodes.Gaussian(1,np.zeros((1,1)),np.array([[1e-3]]))
	nodes = [nodes.Gaussian(1,mu,prec) for n in range(N)]
	for n,x in zip(nodes,Xdata):
		n.observe(x.reshape(1,1))
	for i in range(10):
		mu.update()
		prec.update()
		print float(mu.qmu),truemu
		print float(prec.qa/prec.qb),true_prec
		
	
	
    
def linear_system_inference():
	#Infering the states of a linear dynamic system.
	# the dimension of the state is 2, the dimension of the observations is 1. There are no inputs.
	m,c,k,dt = 1,50,2000,5e-5
	A = np.array([[0,1],[-k/m,-c/m]])*dt + np.eye(2)
	Q = np.array([[0.1,0.],[.0,1.]])*np.sqrt(dt)
	Qchol = np.linalg.cholesky(Q)
	Qinv = linalg.cho_solve((Qchol,1),np.eye(2))
	#C = np.array([[0,1]])
	C = np.random.randn(1,2)
	R = np.array([[.01]])
	Rchol = np.sqrt(R)
	Rinv = 1./R
	
	z0 = np.dot(Qchol,np.random.randn(2))
	T = 500
	#simulate the system
	Z = np.zeros((T,2))
	Y = np.zeros((T,1))
	Z[0] = z0
	for ztm1,zt,yt in zip(Z[:-1],Z[1:],Y[1:]):
		zt[:] = np.dot(A,ztm1) + np.dot(Qchol,np.random.randn(2))
		yt[:] = np.dot(C,zt) + np.dot(Rchol, np.random.randn())
		
		
	#define nodes
	Anode = nodes.Constant(A)
	Cnode = nodes.Constant(C)
	Rnode = nodes.Constant(Rinv)
	Qnode = nodes.Constant(Qinv)
	Znodes = []
	Ynodes = []
	Znodes.append(nodes.Gaussian(2,np.zeros((2,1)),np.eye(2)))
	Ynodes.append(nodes.Gaussian(1,Cnode*Znodes[-1],Rnode))
	Ynodes[-1].observe(Y[0].reshape(1,1))
	Znodes[-1].update()#start of KF
	for i in range(1,T):
		Znodes.append(nodes.Gaussian(2,Anode*Znodes[-1],Qnode))
		Ynodes.append(nodes.Gaussian(1,Cnode*Znodes[-1],Rnode))
		Ynodes[-1].observe(Y[i].reshape(1,1))
		Znodes[-1].update()#KF?
		
	#update nodes (smoothing?)
	niters = 20
	for i in range(niters):
		Znodes.reverse()#this doesn't change any connections!
		for zn in Znodes:
			zn.update()#reverse pass
		Znodes.reverse()
		for zn in Znodes:
			zn.update()#forward pass
		
		
	pylab.figure()
	pylab.title(str(i)+' iters')
	pylab.plot(Y,'g',linewidth=2,label='observations')
	inferred_states = np.hstack([e.qmu for e in Znodes]).T
	pylab.plot(inferred_states,'r',label='inferred states')
	pylab.plot(Z,'b',label='true states')
	pylab.plot(np.dot(inferred_states,C.T),'m',linewidth=2,label='smoothed_obs')
	pylab.legend()
		
	
	pylab.show()
	
	
if __name__=='__main__':
	#Principal component Analysis.
	q = 2 #latent dimension
	d = 3 #observation dimension
	N = 50
	true_W = np.random.randn(d,q)
	true_Z = np.random.randn(N,q)
	true_mean = np.random.randn(d,1)
	true_prec = 50
	X_data = np.dot(true_Z,true_W.T) + true_mean.T + np.random.randn(N,d)*np.sqrt(1./true_prec)
	
	#set up the problem...
	Ws = [nodes.Gaussian(d,np.zeros((d,1)),np.eye(d)*1e-6) for  i in range(q)]
	W = nodes.hstack(Ws)
	Mu = nodes.Gaussian(d,np.zeros((d,1)),np.eye(d))
	Beta = nodes.Gamma(d,1e-3,1e-3)
	Zs = [nodes.Gaussian(q,np.zeros((q,1)),np.eye(q)) for i in range(N)]
	Xs = [nodes.Gaussian(d,W*z+Mu,Beta) for z in Zs]
	[xnode.observe(xval.reshape(d,1)) for xnode,xval in zip(Xs,X_data)]
	
	#infer!
	niters = 100
	for i in range(niters):
		[w.update() for w in Ws]
		Mu.update()
		[z.update() for z in Zs]
		Beta.update()
	
	#plot
	print np.hstack([W.pass_down_Ex(),true_W])
	
    
    
    
    
    
    
    
    