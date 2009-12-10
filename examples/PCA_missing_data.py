# -*- coding: utf-8 -*-
# Copyright 2009 James Hensman and Michael Dewar
# Licensed under the Gnu General Public license, see COPYING
import numpy as np
import sys
sys.path.append('../src')
from pyvb import nodes,Network

def PCA_missing_data(plot=True):
	#Principal Component Analysis, with randomly missing data
	q = 2 #latent dimension
	d = 10 #observation dimension
	N = 200
	niters = 100
	Nmissing = 100
	true_W = np.random.randn(d,q)
	true_Z = np.random.randn(N,q)
	true_mean = np.random.randn(d,1)
	true_prec = 20.
	Xdata_full = np.dot(true_Z,true_W.T) + true_mean.T 
	Xdata_observed = Xdata_full + np.random.randn(N,d)*np.sqrt(1./true_prec)
	
	#erase some data
	missing_index_i = np.argsort(np.random.randn(N))[:Nmissing]
	missing_index_j = np.random.multinomial(1,np.ones(d)/d,Nmissing).nonzero()[1]
	Xdata = Xdata_observed.copy()
	Xdata[missing_index_i,missing_index_j] = np.nan
	
	
	#set up the problem...
	Ws = [nodes.Gaussian(d,np.zeros((d,1)),np.eye(d)*1e-3) for  i in range(q)]
	W = nodes.hstack(Ws)
	Mu = nodes.Gaussian(d,np.zeros((d,1)),np.eye(d)*1e-3)
	Beta = nodes.Gamma(d,1e-3,1e-3)
	Zs = [nodes.Gaussian(q,np.zeros((q,1)),np.eye(q)) for i in range(N)]
	Xs = [nodes.Gaussian(d,W*z+Mu,Beta) for z in Zs]
	[xnode.observe(xval.reshape(d,1)) for xnode,xval in zip(Xs,Xdata)]
	
	#make a network object
	net = Network()
	net.addnode(W)
	net.find_connected(W)# automagically fetches all of the other nodes...
	
	#infer!
	net.learn(100)
		
	#plot
	if plot:
		import pylab
		import hinton
		#compare true and learned W 
		Qtrue,Rtrue = np.linalg.qr(true_W)
		Qlearn,Rlearn = np.linalg.qr(W.pass_down_Ex())
		pylab.figure();pylab.title('True W')
		hinton.hinton(Qtrue)
		pylab.figure();pylab.title('E[W]')
		hinton.hinton(Qlearn)
		
		if q==2:#plot the latent variables
			pylab.figure();pylab.title('true Z')
			pylab.scatter(true_Z[:,0],true_Z[:,1],50,true_Z[:,0])
			pylab.figure();pylab.title('learned Z')
			learned_Z = np.hstack([z.pass_down_Ex() for z in Zs]).T
			pylab.scatter(learned_Z[:,0],learned_Z[:,1],50,true_Z[:,0])
			
		#recovered X mean
		X_rec = np.hstack([x.pass_down_Ex() for x in Xs]).T
		
		#Recovered X Variance
		#slight hack here - set q variance of observed nodes to zeros (it should be random...)
		for x in Xs:
			if x.observed:
				x.qcov *=0
		var_rec = np.vstack([np.diag(x.qcov) for x in Xs]) + 1./np.diag(Beta.pass_down_Ex())
		
		#plot each recovered signal in a separate figure
		for i in range(d):
			pylab.figure();pylab.title('recovered_signal '+str(i))
			
			pylab.plot(Xdata_full[:,i],'g',marker='.',label='True') # 'true' values of missing data (without noise)
			pylab.plot(X_rec[:,i],'b',label='Recovered') # recovered mising data values
			pylab.plot(Xdata[:,i],'k',marker='o',linewidth=2,label='Observed') # with noise, and holes where we took out values
			pylab.legend()
			
			volume_x = np.hstack((np.arange(len(Xs)),np.arange(len(Xs))[::-1]))
			volume_y = np.hstack((X_rec[:,i]+2*np.sqrt(var_rec[:,i]), X_rec[:,i][::-1]-2*np.sqrt(var_rec[:,i])[::-1]))
			pylab.fill(volume_x,volume_y,'b',alpha=0.3)
			
		
		
		print '\nBeta'
		print true_prec,Beta.pass_down_Ex()[0,0]
		print '\nMu'
		print np.hstack((true_mean,Mu.pass_down_Ex()))
		pylab.show()
		
		
if __name__=='__main__':
	PCA_missing_data(False)