#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 14:41:21 2022

@author: gijsbartholomeus
"""

import numpy as np
from scipy import linalg as LA
import random
from math import sqrt
import matplotlib.pylab as plt
import time


Symmetric = False
n = 50
mean = 0
gamma = 0.2
std = gamma/np.sqrt(n)  
Diagonalzero = True
evals = []
evals = np.array(evals)
if Symmetric is True:
    sym = 'sym'
else:
    sym= 'notsym'

if Symmetric is True:
    G = np.zeros((n,n))             # initialize a two-dimensional array
    for i in range(n):
        for j in range(i,n):
            G[i, j] = random.gauss(mean,std)    # random element 
            G[j, i] = G[i, j]
else:
    G = np.random.normal(mean,std, size=(n,n))
if Diagonalzero is True:
    for i in range(n):
        G[i, i] = 0  
        
Species = np.random.uniform(0.1,1,n)
D = np.diag(Species)


def Teq(x,y,Vec):
    a=0
    for i in range(n):
        Ni = Vec[i]
        z = complex(x,y)/(1-mean)
        zc = np.conjugate(complex(x,y))/(1-mean)
        if Ni>0:
            a += (Ni**2) / (( z + Ni)*(zc + Ni))
    return a

    
if Symmetric is True:
    evals = LA.eigvals(G).real        
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

else:
    "Now for May"
    M = G -np.eye(n)
    evals = LA.eigvals(M)        
    realvals = evals.real
    imvals = evals.imag
    plt.scatter(realvals,imvals,color='g')
    H = max(abs(imvals))                                           # height of the circle
    R = max(abs(realvals))                                      # radius of the circle
    #x = np.linspace(-R,R,100)                               # an x-grid to draw the semicircle
    #y = np.array([H*sqrt(1-((xval)/R)**2) for xval in x])     # y-values for the semicircle
    #negy = -y
    plt.axis('equal')
                                         # a bit of margin at the top
    #plt.plot(x, y, color = 'r', linewidth = 3)      
    #plt.plot(x, negy, color = 'r', linewidth = 3)              # plot the circle
        # plot the circle
    #plt.xlim(-R*1.1,R*1.1)                                        # a bit of margin on the sides
    plt.ylim(-H*1.1,H*1.1)   
    plt.xlabel(r'Real axis', fontsize = 18)                 
    plt.ylabel(r'Imaginary axis', fontsize = 18)    
    arrow= plt.arrow(-1, 0,gamma, 0,head_width=0.01,color = 'r',length_includes_head=True,label='My label')
    plt.legend([arrow,], ['Gamma',])                 
    #plt.savefig('wigner.pdf')            
    plt.title('Eigenvalue spectrum for matrix A')
    plt.show()
    
    "Now for Stone"
    plt.figure()
    S = np.dot(D,M)
    evals = LA.eigvals(S)        
    realvals = evals.real
    imvals = evals.imag
    plt.scatter(realvals,imvals,color='g')
    H = max(abs(imvals))                                           # height of the circle
    R = max(abs(realvals))                                      # radius of the circle
    #x = np.linspace(-R,R,100)                               # an x-grid to draw the semicircle
    #y = np.array([H*sqrt(1-((xval)/R)**2) for xval in x])     # y-values for the semicircle
    #negy = -y
    #plt.ylim(-H*1.1,H*1.1)  
    #plt.xlim(-R*1.1,R*1.1)                                        # a bit of margin on the sides
                                      # a bit of margin at the top
    #plt.plot(x, y, color = 'r', linewidth = 3)      
    #plt.plot(x, negy, color = 'r', linewidth = 3)              # plot the circle
        # plot the circle
    plt.xlabel(r'Real axis', fontsize = 18)                 
    plt.ylabel(r'Imaginary axis', fontsize = 18)   
    plt.title('Eigenvalue spectrum for matrix S = DA')
                    
    #plt.savefig('wigner.pdf')            
    plt.show()
    
    
    real = np.linspace(-1,0,1000)
    zero = np.linspace(0,0,1000)
    T=[]
    s = Species
    for i in range(len(real)):
        t=np.real(Teq(real[i],zero[i],s)) 
        T.append(t)
    plt.plot(real,T)
    plt.ylim(0,10000)
    plt.ylabel('Teq')
    plt.xlabel('Real')
    plt.axhline(y = 1/std**2, color = 'r', linestyle = 'dashed', label = "1/(gamma^2)")    
    plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper center')
    plt.show()
    
    
    real = np.linspace(-1.5,0.5, 1000)
    im = np.linspace(-1,1,1000)
    delta=10
    lijst = []
    print("start")
    start = time.time()
    for i in range(len(real)):
        for k in range(len(im)):
            bol = 1/std**2 - delta < Teq(real[i],im[k],Species) < 1/std**2+ delta
            if (bol):
                lijst.append([real[i],im[k]])
        if i % 100 ==0:
            print(i)    
    lijst= np.array(lijst)
    end = time.time()
    elapsed_time = end-start
    print("Looping took",elapsed_time,"seconds")
    
            
    plt.figure()
    plt.scatter(lijst[:,0],lijst[:,1],label='theoretical bound')
    plt.scatter(realvals,imvals,color='g', label='eigenvalues')
    plt.ylabel('Imaginary')
    plt.xlabel('Real')   
    plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper center')
    plt.title('Eigenvalue spectrum for matrix S = DA')
    plt.show()
