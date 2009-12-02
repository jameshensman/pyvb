# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.append('../src')
from pyvb import nodes

def PCA_missing_data(plot=True):
	#Principal Component Analysis, with randomly missing data
	q = 2 #latent dimension
	d = 3 #observation dimension
	N = 200
	niters = 500
	Nmissing = 100
	true_W = np.random.randn(d,q)*10
	true_Z = np.random.randn(N,q)
	true_mean = np.random.randn(d,1)
	true_prec = 100.
	Xdata_full = np.dot(true_Z,true_W.T) + true_mean.T + np.random.randn(N,d)*np.sqrt(1./true_prec)
	
	#erase some data
	missing_index_i = np.argsort(np.random.randn(N))[:Nmissing]
	missing_index_j = np.random.multinomial(1,np.ones(d)/d,Nmissing).nonzero()[1]
	Xdata = Xdata_full.copy()
	Xdata[missing_index_i,missing_index_j] = np.nan
	
	
	#set up the problem...
	Ws = [nodes.Gaussian(d,np.zeros((d,1)),np.eye(d)*1e-3) for  i in range(q)]
	W = nodes.hstack(Ws)
	Mu = nodes.Gaussian(d,np.zeros((d,1)),np.eye(d)*1e-3)
	Beta = nodes.Gamma(d,1e-3,1e-3)
	Zs = [nodes.Gaussian(q,np.zeros((q,1)),np.eye(q)) for i in range(N)]
	Xs = [nodes.Gaussian(d,W*z+Mu,Beta) for z in Zs]
	[xnode.observe(xval.reshape(d,1)) for xnode,xval in zip(Xs,Xdata)]
	
	#infer!
	for i in range(niters):
		[w.update() for w in Ws]
		Mu.update()
		[z.update() for z in Zs]
		[x.update() for x in Xs]
		Beta.update()
		print niters-i
		
	#plot
	if plot:
		import pylab
		#compare true and learned W TODO: hinton diagrams
		Qtrue,Rtrue = np.linalg.qr(true_W)
		Qlearn,Rlearn = np.linalg.qr(W.pass_down_Ex())
		pylab.figure();pylab.title('True W')
		pylab.imshow(Qtrue,interpolation='nearest')
		pylab.figure();pylab.title('E[W]')
		pylab.imshow(Qlearn,interpolation='nearest')
		
		if q==2:#plot the latent variables
			pylab.figure();pylab.title('true Z')
			pylab.scatter(true_Z[:,0],true_Z[:,1],50,true_Z[:,0])
			pylab.figure();pylab.title('learned Z')
			learned_Z = np.hstack([z.pass_down_Ex() for z in Zs]).T
			pylab.scatter(learned_Z[:,0],learned_Z[:,1],50,true_Z[:,0])
			
		#plot recovered X
		pylab.figure();pylab.title('recovered_signals')
		X_rec = np.hstack([x.pass_down_Ex() for x in Xs]).T
		pylab.plot(Xdata_full,'g',marker='.',label='True') # 'true' values of missing data
		pylab.plot(X_rec,'k',label='recovered') # recovered mising data values
		pylab.plot(Xdata,'b',marker='o',linewidth=2,label='observed') # this will have holes where we took out values
		pylab.legend()
		
		
		print '\nBeta'
		print true_prec,Beta.pass_down_Ex()[0,0]
		print '\nMu'
		print np.hstack((true_mean,Mu.pass_down_Ex()))
		pylab.show()
		
		
if __name__=='__main__':
	PCA_missing_data()