# -*- coding: utf-8 -*-
# Copyright 2009 James Hensman and Michael Dewar
# Licensed under the Gnu General Public license, see COPYING
import pylab

def hinton(W):
	"""Draw a hinton diagram of the matrix W on the current pylab axis"""
	M,N = W.shape
	square_x,square_y = pylab.array([-.5,.5,.5,-.5]),pylab.array([-.5,-.5,.5,.5])
	pylab.fill([-.5,N-.5,N-.5,-.5],[-.5,-.5,M-.5,M-.5],'grey')
	Wmax = pylab.sqrt(pylab.square(W)).max() #pylab. has no abs!
	for m,Wrow in enumerate(W):
		for n,w in enumerate(Wrow):
			c = pylab.signbit(w) and 'k' or 'w'#love python :)
			pylab.fill(square_x*w/Wmax+n,square_y*w/Wmax+m,c,edgecolor=c)
	pylab.axis('equal')
	pylab.ylim(M-0.5,-0.5)
				
				
			
			