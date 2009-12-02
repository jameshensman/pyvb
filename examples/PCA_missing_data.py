# -*- coding: utf-8 -*-
from pyvb import nodes
import numpy as np
def PCA_missing_data(plot=True):
	#Principal Component Analysis, with arbitrary missing data
	q = 2 #latent dimension
	d = 7 #observation dimension
	N = 200
	niters = 200
	Nmissing = 10
	true_W = np.random.randn(d,q)*10
	true_Z = np.random.randn(N,q)
	true_mean = np.random.randn(d,1)
	true_prec = 100.
	X_data_full = np.dot(true_Z,true_W.T) + true_mean.T + np.random.randn(N,d)*np.sqrt(1./true_prec)
	
	#erase some data
	missing_index_i = np.argsort(np.random.randn(N))[:nmissing]
	missing_index_j = np.random.multinomial(1,np.ones(d),nmissing).nonzero()[1]
	Xdata = Xdata_full.copy()
	Xdata[missing_index_i,missing_index_j] = np.nan
	
	
	
	
	#set up the problem...
	Ws = [nodes.Gaussian(d,np.zeros((d,1)),np.eye(d)*1e-3) for  i in range(q)]
	W = nodes.hstack(Ws)
	Mu = nodes.Gaussian(d,np.zeros((d,1)),np.eye(d)*1e-3)
	Beta = nodes.Gamma(d,1e-3,1e-3)
	Zs = [nodes.Gaussian(q,np.zeros((q,1)),np.eye(q)) for i in range(N)]
	Xs = [nodes.Gaussian(d,W*z+Mu,Beta) for z in Zs]
	[xnode.observe(xval.reshape(d,1)) for xnode,xval in zip(Xs,X_data)]
	
	#infer!
	for i in range(niters):
		[w.update() for w in Ws]
		Mu.update()
		[z.update() for z in Zs]
		Beta.update()
		print niters-i
		
		#plot
		import pylab
		pylab.figure();pylab.title('True W')
		pylab.imshow( np.linalg.qr(W.pass_down_Ex())[0],interpolation='nearest')
		pylab.figure();pylab.title('E[W]')
		pylab.imshow( np.linalg.qr(true_W)[0],interpolation='nearest')
		pylab.figure();pylab.title('true Z')
		pylab.scatter(true_Z[:,0],true_Z[:,1],50,true_Z[:,0])
		pylab.figure();pylab.title('learned Z')
		learned_Z = np.hstack([z.qmu for z in Zs]).T
		pylab.scatter(learned_Z[:,0],learned_Z[:,1],50,true_Z[:,0])
		
		print '\nBeta'
		print true_prec,Beta.pass_down_Ex()[0,0]
		print '\nMu'
		print np.hstack((true_mean,Mu.pass_down_Ex()))
		
if __name__=='__main__':