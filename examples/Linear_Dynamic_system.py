# -*- coding: utf-8 -*-
# Copyright 2009 James Hensman and Michael Dewar
# Licensed under the Gnu General Public license, see COPYING
import numpy as np
import sys
sys.path.append('../src')
from pyvb import nodes

def LDS(plot=True):
	#Learning of a linear dynamic system.
	# x_{t+1} ~ N(A x_t,Q)
	# y_t  ~ N(C x_t,R)
	q = 2 #latent (state) dimension
	d = 3 #observation dimension
	T = 200 # timesteps
	niters = 500 # VB iterations
	
	true_A = np.random.randn(q,q)
	while np.max(np.abs(np.linalg.eig(A)[0])) >1:# make sure we havea stable system.
		true_A = np.random.randn(q,q)
	
	true_C = np.random.randn(d,q)*10
	
	r_temp = np.random.randn(1000,d)
	true_R = np.dot(r_temp.T,r_temp)/1000
	true_R_chol = np.linalg.cholesky(true_R)
	
	q_temp = np.random.randn(1000,d)
	true_Q = np.dot(q_temp.T,q_temp)/1000
	true_Q_chol = np.linalg.cholesky(true_Q)
	
	#simulate the system. TODO check I've not got cholesky the wrong way around...
	true_X = np.zeros((T,q))
	Y_data = np.zeros((T,d))
	X[0] = np.random.randn(d)
	Y_data[0] = np.dot(true_C,X[0].reshape(q,1)).flatten() + np.dot(true_R_chol,np.random.randn(d,1)).flatten()
	Xs.append(nodes.Gaussian(
	for xlag,xnow,ynow in zip(X[:-1],X[1:],Y_data[1:]):
		xnow[:] = np.dot(true_A,xlag).flatten() + np.dot(true_Q_chol,np.random.randn(q,1)).flatten()
		ynow[:] = np.dot(true_C,xnow) + np.dot(true_R_chol,np.random.randn(d,1)).flatten()
	
	#set up the problem...
	As = [nodes.Gaussian(q,np.zeros((q,1)),np.eye(q)*1e-3) for  i in range(q)]
	A = nodes.hstack(As) #node to represent A
	Cs = [nodes.Gaussian(d,np.zeros((d,1)),np.eye(d)*1e-3) for  i in range(q)]
	C = nodes.hstack(Cs) #node to represent 
	Q = nodes.Wishart(q,np.eye(q)*1e-3,1e-3) # TODO check order of args to Wishart
	R = nodes.Wishart(d,np.eye(d)*1e-3,1e-3)
	
	X0 = nodes.Gaussian(q,np.zeros((q,1)),np.eye(q))
	Y0 = nodes.Gaussian(d,C*X0,R)
	Xs = [X0]
	Ys = [Y0]
	for t in range(1,T):
		Xs.append(nodes.Gaussian(q,A*Xs[-1],Q)
		Ys.append(nodes.Gaussian(d,C*Xs[-1],R)
		Ys[-1].observe(Y_data[t].reshape(d,1))
		
	#infer!
	for i in range(niters):
		[x.update() for x in Xs]
		[a.update() for a in As]
		[c.update() for c in Cs]
		Q.update()
		R.update()
		
	#plot
	if plot:
		pass
		
		
if __name__=='__main__':
	LDS()