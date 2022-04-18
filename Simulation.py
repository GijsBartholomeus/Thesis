#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 19:14:00 2022

@author: gijsbartholomeus
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
import ipywidgets as ipw
import random

random.seed(42) # because Farshid wanted 42

Symmetric = True
n = 20
mean = 0/n
std = 0.8/np.sqrt(n)  
Diagonalzero = True
gamma = std*np.sqrt(n)/(1-mean)
print('gamma =',gamma)
"""           
We generate a matrix with random values sampled from a gaussian distribution with mean default 4/n 
and standard deeviation default 0.8/sqrt(n)
"""
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

K = np.zeros(n)            
for i in range(n):
    K[i]=1


def derivative(X,T, cap, A,delta):
   dotX = np.diag(X).dot((K-X-np.einsum('ij,j', A,X)))
   for i in range(n):
       boolean = -delta <= X[i] <= delta
       if (boolean):
           dotX[i]=0
   return np.array(dotX)

Nt = 1000
tmax = 30.
T = np.linspace(0.,tmax, Nt)
X0 = np.random.uniform(0.1,1, 20)
print(X0)
res = integrate.odeint(derivative, X0, T, args = (K,G,10**(-7)))
#a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t = np.array(res.T)[:,:]

a,b,c,d,e = res.T[0:5,]


plt.figure()
plt.grid()
plt.title("odeint method")
plt.plot(T, a, 'xb', label = '1')
plt.plot(T, b, '+r', label = "2")
plt.plot(T, c, '+g', label = "3")
plt.plot(T, d, '*m', label = "4")
plt.plot(T, e, '-c', label = "5")
plt.xlabel('Time T, [days]')
plt.ylabel('Population')
plt.legend()
plt.show()

f,g,h,i,j = np.array(res.T)[5:10,:]
plt.figure()
plt.grid()
plt.title("odeint method")
plt.plot(T, f, 'xb', label = '6')
plt.plot(T, g, '+r', label = "7")
plt.plot(T, h, '+g', label = "8")
plt.plot(T, i, '*m', label = "9")
plt.plot(T, j, '-c', label = "10")
plt.xlabel('Time T, [days]')
plt.ylabel('Population')
plt.legend()
plt.show()

k,l,m,n,o = np.array(res.T)[10:15,:]
plt.figure()
plt.grid()
plt.title("odeint method")
plt.plot(T, k, 'xb', label = '11')
plt.plot(T, l, '+r', label = "12")
plt.plot(T, m, '+g', label = "13")
plt.plot(T, n, '*m', label = "14")
plt.plot(T, o, '-c', label = "15")
plt.xlabel('Time T, [days]')
plt.ylabel('Population')
plt.legend()
plt.show()

p,q,r,s,t = res.T[15:20,:]
plt.figure()
plt.grid()
plt.title("odeint method")
plt.plot(T, p, 'xb', label = '16')
plt.plot(T, q, '+r', label = "17")
plt.plot(T, r, '+g', label = "18")
plt.plot(T, s, '*m', label = "19")
plt.plot(T, t, '-c', label = "20")
plt.xlabel('Time T, [days]')
plt.ylabel('Population')
plt.legend()
plt.show()



