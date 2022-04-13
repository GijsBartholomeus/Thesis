#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 11:55:01 2022

@author: gijsbartholomeus
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

"""
We will plot the eigenvalue spectrum for the Jacobian of a fixed points
"""  
n=20
Symmetric = False
mean = 4/n
std = 0.8/np.sqrt(n)  
Diagonalzero = True

if Symmetric is True:
    sym = 'sym'
else:
    sym= 'notsym'
gamma = std*np.sqrt(n)/(1-mean)
print(gamma)

name = str(str(sym)+"_mu"+str(mean)+"_sigma"+str(round(std,2)))
path = "/Users/gijsbartholomeus/OneDrive - Universiteit Utrecht/Thesis/Pickles/"+name

# def Teq(z):
#     a=0
#     for i in n:
#         a += (Points[j][i]**2) / (( z/(1-mean) + Points[j][i])*(np.conjugate(z)/(1-mean) + Points[j][i]))
#     return a

def Teq(x,y,array,ChosenFP):
    a=0
    for i in range(n):
        Ni =array[ChosenFP][i]
        z = complex(x,y)/(1-mean)
        zc = np.conjugate(complex(x,y))/(1-mean)
        if Ni>0:
            a += (Ni**2) / (( z + Ni)*(zc + Ni))
    return a

file_name = path+"/"+str(name)+"_Stablefeasiblefixedpoints.pkl"
open_file = open(file_name, "rb")
Points = pickle.load(open_file)
open_file.close()
print('loaded1')

file_name = path+"/"+str(name)+"_Stableevals.pkl"
open_file = open(file_name, "rb")
Lamdas = pickle.load(open_file)
open_file.close()
print('loaded1')


eigenvalues = Lamdas[-1]

x = eigenvalues.real
# extract imaginary part
y = eigenvalues.imag
  
# plot the complex numbers
plt.scatter(x, y)
plt.ylabel('Imaginary')
plt.xlabel('Real')            # plot the semicircle
plt.show()

real = np.linspace(-3,3,1000)
im = np.linspace(-10,10,1000)
delta=10**-2
lijst = []
print("start")
for i in range(len(real)):
    for k in range(len(im)):
        bol = (1 - mean)**2/std**2/5 - delta < Teq(real[i],im[k],Points,120000) < (1 - mean)**2/std**2/5 + delta
        if (bol):
            lijst.append([real[i],im[k]])
    if i % 100 ==0:
        print(i)    
lijst= np.array(lijst)
        
plt.figure()
plt.scatter(lijst[:,0],lijst[:,1])
plt.ylabel('Imaginary')
plt.xlabel('Real')   
plt.show()

"This is to test Teq"
n=10
ChosenFP=0
real = np.linspace(-1,0,1000)
zero = np.linspace(0,0,1000)
T=[]
s = np.array([np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])])
for i in range(len(real)):
    t=np.real(Teq(real[i],zero[i],s,0)) 
    T.append(t)
plt.plot(real,T)
plt.ylim(0,10000)
plt.ylabel('Teq')
plt.xlabel('Real')   
plt.show

'another way to plot teq'
m=1000
y, x = np.meshgrid(np.linspace(-5, 5, m), np.linspace(-5, 10, m))
def func(x,y):
    summand=0
    for i in range(n):
        Ni = Points[12000][i]
        if Ni > 0:
            summand += (Ni**2) / ((x+1j*y + Ni) * (x-1j*y + Ni))
    return np.real(summand)

z = func(x,y)

for i in np.arange(len(x)):
    for j in np.arange(len(y)):
        if z[j,i] > 2000 or z[j,i] < -50:
            z[j,i] = np.nan
        else:
            pass

# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
z_min, z_max = 7,0
fig, ax = plt.subplots()

c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('pcolormesh')
# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)

plt.show()




