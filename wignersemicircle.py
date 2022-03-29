#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:41:21 2022

@author: gijsbartholomeus
"""

import numpy as np
from scipy import linalg as LA
from random import gauss
from math import sqrt
import matplotlib.pylab as plt
n = 20  
evals = []
evals = np.array(evals)
for k in range(n):                         # dimension of the random matrix
    G = np.zeros((n,n))             # initialize a two-dimensional array
    for i in range(n):
        for j in range(i,n):
            G[i, j] = gauss(0,1)    # random element
            G[j, i] = G[i, j]       # symmetrically placed element
    new = LA.eigvals(G).real        
    evals = np.append(evals, new)
bins, vals, patches = plt.hist(evals.real, bins = 50)   # draw the histogram of all eigenvalues
H = max(bins)                                           # height of the circle
R = max(abs(vals))                                      # radius of the circle
x = np.linspace(-R,R,100)                               # an x-grid to draw the semicircle
y = np.array([H*sqrt(1-(xval/R)**2) for xval in x])     # y-values for the semicircle

plt.xlim(-R*1.3,R*1.3)                                        # a bit of margin on the sides
plt.ylim(0,H*1.1)                                          # a bit of margin at the top
plt.plot(x, y, color = 'r', linewidth = 3)              # plot the semicircle
plt.xlabel(r'$\lambda$', fontsize = 18)                 
plt.ylabel(r'$n$', fontsize = 18)                       
#plt.savefig('wigner.pdf')                               

plt.show()


"Now for May"
M = G- R*np.eye(1000)
evalsM = LA.eigvals(M).real
bins, vals, patches = plt.hist(evalsM.real, bins = 50)   # draw the histogram of all eigenvalues
H = max(bins)                                           # height of the circle
R = max(abs(vals))                                      # radius of the circle
x = np.linspace(-R,R,100)                               # an x-grid to draw the semicircle
y = np.array([H*sqrt(1-(xval/R)**2) for xval in x])     # y-values for the semicircle

plt.xlim(-R*1.3,R*1.3)                                        # a bit of margin on the sides
plt.ylim(0,H*1.1)                                          # a bit of margin at the top
plt.plot(x, y, color = 'r', linewidth = 3)              # plot the semicircle
plt.xlabel(r'$\lambda - 1$', fontsize = 18)                 
plt.ylabel(r'$n$', fontsize = 18)                       
#plt.savefig('wigner.pdf')            
plt.show()
