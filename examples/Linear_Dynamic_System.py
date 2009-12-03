# -*- coding: utf-8 -*-
# Copyright 2009 James Hensman and Michael Dewar
# Licensed under the Gnu General Public license, see COPYING
import numpy as np
import sys
sys.path.append('../src')
from pyvb import nodes

#def LDS(plot=True):
if __name__=='__main__':
	plot=True
	#Learning of a linear dynamic system.
	# x_{t+1} ~ N(A x_t,Q)
	# y_t  ~ N(C x_t,R)
	q = 2 #latent (state) dimension
	d = 5 #observation dimension
	T = 200 # timesteps
	niters = 50 # VB iterations
	
	true_A = np.random.randn(q,q)
	while np.max(np.abs(np.linalg.eig(true_A)[0])) > 1:# make sure we havea stable system.
		print 're-randomising A'
		true_A = np.random.randn(q,q)
	
	true_C = np.random.randn(d,q)*10
	
	#r_temp = np.random.randn(100,d)
	#true_R = np.dot(r_temp.T,r_temp)/10000
	true_R = np.diag(np.random.rand(d))*0.1
	true_R_chol = np.linalg.cholesky(true_R)
	
	#q_temp = np.random.randn(100,q)
	#true_Q = np.dot(q_temp.T,q_temp)/10000
	true_Q = np.diag(np.random.rand(q))*0.1
	true_Q_chol = np.linalg.cholesky(true_Q)
	
	#simulate the system. TODO check I've not got cholesky the wrong way around...
	true_X = np.zeros((T,q))
	Y_data = np.zeros((T,d))
	true_X[0] = np.random.randn(q)
	Y_data[0] = np.dot(true_C,true_X[0].reshape(q,1)).flatten() + np.dot(true_R_chol,np.random.randn(d,1)).flatten()
	for xlag,xnow,ynow in zip(true_X[:-1],true_X[1:],Y_data[1:]):
		xnow[:] = np.dot(true_A,xlag).flatten() + np.dot(true_Q_chol,np.random.randn(q,1)).flatten()
		ynow[:] = np.dot(true_C,xnow) + np.dot(true_R_chol,np.random.randn(d,1)).flatten()
	
	#set up the problem...
	As = [nodes.Gaussian(q,np.zeros((q,1)),np.eye(q)*1e-3) for  i in range(q)]
	A = nodes.hstack(As) #node to represent A
	Cs = [nodes.Gaussian(d,np.zeros((d,1)),np.eye(d)*1e-3) for  i in range(q)]
	C = nodes.hstack(Cs) #node to represent 
	#Q = nodes.Gamma(q,1e-3,1e-3)
	#R = nodes.Gamma(d,1e-3,1e-3)
	Q = nodes.DiagonalGamma(q,np.ones(q)*1e-3,np.ones(q)*1e-3)
	R = nodes.DiagonalGamma(d,np.ones(d)*1e-3,np.ones(d)*1e-3)
	#Q = nodes.Wishart(q,1e-3,np.eye(q)*1e-3) 
	#R = nodes.Wishart(d,1e-3,np.eye(d)*1e-3)
	
	X0 = nodes.Gaussian(q,np.zeros((q,1)),np.eye(q))
	Y0 = nodes.Gaussian(d,C*X0,R)
	Y0.observe(Y_data[0].reshape(d,1))
	Xs = [X0]
	Ys = [Y0]
	for t in range(1,T):
		Xs.append(nodes.Gaussian(q,A*Xs[-1],Q))
		Ys.append(nodes.Gaussian(d,C*Xs[-1],R))
		Ys[-1].observe(Y_data[t].reshape(d,1))
		
	#infer!
	for i in range(niters):
		[x.update() for x in Xs]
		Xs.reverse()
		[x.update() for x in Xs]
		Xs.reverse()
		[a.update() for a in As]
		[c.update() for c in Cs]
		Q.update()
		R.update()
		print niters-i
		
	#plot
	if plot:
		import pylab
		import hinton
		#plot hintons of learned (and true) matrices.
		pylab.figure()
		pylab.subplot(1,2,1)
		pylab.title('True A')
		hinton.hinton(true_A)
		pylab.subplot(1,2,2)
		pylab.title('E[A]')
		hinton.hinton(A.pass_down_Ex())
		
		pylab.figure()
		pylab.subplot(1,2,1)
		pylab.title('True C')
		hinton.hinton(true_C)
		pylab.subplot(1,2,2)
		pylab.title('E[C]')
		hinton.hinton(C.pass_down_Ex())
		
		pylab.figure()
		pylab.subplot(2,2,1)
		pylab.title('True Q')
		hinton.hinton(true_Q)
		pylab.subplot(2,2,2)
		pylab.title('E[Q]')
		hinton.hinton(np.linalg.inv(Q.pass_down_Ex()))
		pylab.subplot(2,2,3)
		pylab.title('True R')
		hinton.hinton(true_R)
		pylab.subplot(2,2,4)
		pylab.title('E[R]')
		hinton.hinton(np.linalg.inv(R.pass_down_Ex()))
		
		#plot states
		pylab.figure()
		pylab.subplot(1,2,1)
		pylab.title('True states')
		pylab.plot(true_X[:,0],true_X[:,1],'g',marker='.',label='True')
		pylab.subplot(1,2,2)
		pylab.title('Inferred states')
		Xmean = np.hstack([x.qmu for x in Xs]).T
		pylab.plot(Xmean[:,0],Xmean[:,1],'b',marker='.',label='Inferred')
		
		
		#plot signals.
		Ymean = np.hstack([y.mean_parent.pass_down_Ex() for y in Ys]).T
		Yvar = np.vstack([np.diag(np.linalg.inv(y.precision_parent.pass_down_Ex())) for y in Ys])#how very inefficient, James.
		pylab.figure()
		for i in range(d):
			pylab.subplot(d,1,i+1)
			pylab.plot(Y_data[:,i],'r',marker='.',label='observations')
			pylab.plot(Ymean[:,i],'b',label='mean smoothed val')
			pylab.legend()
			xvolume = np.hstack((np.arange(T),np.arange(T)[::-1]))
			yvolume = np.hstack((Ymean[:,i]+2*np.sqrt(Yvar[:,i]),Ymean[:,i][::-1]-2*np.sqrt(Yvar[:,i])[::-1]))
			pylab.fill(xvolume,yvolume,'b',alpha=0.5)
		
		pylab.show()
			
		
		
		
		
		
		
#if __name__=='__main__':
	#LDS()