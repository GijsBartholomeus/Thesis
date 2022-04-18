#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:22:06 2022

@author: gijsbartholomeus
"""
import numpy as np
import pickle
import random 
import scipy.linalg as la
import time
import matplotlib.pyplot as plt

"""
In this script we calculate the entropy for fixed points and look for stable 
feasible fixed points using the jacobian.
------------------------------------------------------------------------


Loading files
Make sure you run Fixedpoints first
"""

Symmetric = True
n = 20
mean = 40/n
std = 0.8/np.sqrt(n)  
Diagonalzero = True
gamma = std*np.sqrt(n)/(1-mean)
print('gamma =',gamma)

K = np.ones(n)  


if Symmetric is True:
    sym = 'sym'
else:
    sym= 'notsym'


name = str(str(sym)+"_mu"+str(mean)+"_sigma"+str(round(std,2)))
path = "/Users/gijsbartholomeus/OneDrive - Universiteit Utrecht/Thesis/Pickles/"+name


def Jac2(N):
    J = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            J[i, j] = -N[i] * G[i,j]
        J[i, i] = N[i] * (-1 + G[i,i])
        eigvals, eigvecs = la.eig(J)
    return [eigvecs,eigvals]

def Ent(N):
    T = N/N.sum()
    np.seterr(divide='ignore')
    a = -np.nansum(T*np.log(T))
    return(a)

def Nonzero(Vec):
    b = np.zeros(len(Vec))
    for i in range(len(Vec)):
        b[i] = np.count_nonzero(Vec[i])
    return np.array(b)


file_name = path+"/"+str(name)+"_Randommatrix.pkl"
open_file = open(file_name, "rb")
G = pickle.load(open_file)
open_file.close()
  
file_name = path+"/"+str(name)+"_Fixedpoints.pkl"
open_file = open(file_name, "rb")
FP = pickle.load(open_file)
open_file.close()

"""We filter for feasible solutions (N_i \geq 0)"""

FFP = []
for i in range(2**n):
    if all(j >= 0 for j in FP[i]):
        FFP.append(FP[i])
FFP = np.array(FFP)
file_name = path+"/"+str(name)+"_Feasiblefixedpoints.pkl"
open_file = open(file_name, "wb")
pickle.dump(FFP, open_file)
open_file.close()
print(len(FFP))


"""Check if it works out"""
r = random.randint(0, len(FFP))
print("time evolution for fixed point number:",r)
print(FFP[r])
for i in range(n):
    eq = ( FFP[r][i] )*( FFP[r][i] - K[i] + np.einsum('ij,j', G, FFP[r])[i] )
    print(i+1,"dn_/dt=",round(eq,15))

"""""
Finally we look for stable fixed points, i.e. max(eig(Jac(N))) <= 0 
"""""
Realevals = []
Eigvecs = []
Entropy = []
Stableevals = []
Stableeivecs = []
Stableentropy = []
Unstableevals = []
Unstableeivecs = []
Unstableentropy = []

SFFP = []
start = time.time()
for i in range(len(FFP)):
    eigvals = Jac2(FFP[i])[1]
    eigvecs = Jac2(FFP[i])[0]
    entropy = Ent(FFP[i])
    Entropy.append(entropy)
    Eigvecs.append(eigvecs)
    Realevals.append(eigvals)
    #Eigvecs.append(eigvecs)
    if all(j <= 0 for j in eigvals.real):
        SFFP.append(FFP[i])
        Stableevals.append(eigvals)
        Stableeivecs.append(eigvecs)
        Stableentropy.append(entropy)
    if i % 5000 == 0 :
        print(i)
end = time.time()
elapsed_time = end-start
print("Looping took",elapsed_time/60,"minutes")
SFFP = np.array(SFFP)

"Saving Entropy, eigenvalues and eigenvectors"
file_name = path+"/"+str(name)+"_Entropy.pkl"
open_file = open(file_name, "wb")
pickle.dump(entropy, open_file)
open_file.close()
file_name = path+"/"+str(name)+"_Eigenvectors.pkl"
open_file = open(file_name, "wb")
pickle.dump(Eigvecs, open_file)
open_file.close()
file_name = path+"/"+str(name)+"_Eigenvalues.pkl"
open_file = open(file_name, "wb")
pickle.dump(Realevals, open_file)
open_file.close()

"Saving Entropy, eigenvalues and eigenvectors for stable points only"
file_name = path+"/"+str(name)+"_Stablefeasiblefixedpoints.pkl"
open_file = open(file_name, "wb")
pickle.dump(SFFP, open_file)
open_file.close()
file_name = path+"/"+str(name)+"_Stableentropy.pkl"
open_file = open(file_name, "wb")
pickle.dump(Stableentropy, open_file)
open_file.close()
file_name = path+"/"+str(name)+"_Stableeivecs.pkl"
open_file = open(file_name, "wb")
pickle.dump(Stableeivecs, open_file)
open_file.close()
file_name = path+"/"+str(name)+"_Stableevals.pkl"
open_file = open(file_name, "wb")
pickle.dump(Stableevals, open_file)
open_file.close()


   

x = Entropy
y = Stableentropy
plt.figure()
num_bins = 100
l, bins, patches = plt.hist(x, num_bins,
                            color ='green',
                            alpha = 0.7, label='all')
l, bins, patches = plt.hist(y, num_bins,
                            color ='blue',
                            alpha = 0.7, label='stable')
#plt.plot(bins, '--', color ='black')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.ylim(0,10000)
plt.xlim(0,3)
plt.title('Entropy', fontweight ="bold")
plt.legend()
plt.show()
print('Feasible fraction of fixed points:',round(100*len(FFP)/len(FP),2),'%    ------     '
      'Stable fraction of feasible points:',round(100*len(SFFP)/len(FFP),2),'%')
print(name)

y = Nonzero(FFP)
x = Nonzero(SFFP)
plt.figure()
num_bins = 100
l, bins, patches = plt.hist(x, num_bins,
                            color ='green',
                            alpha = 0.7, label='all')
l, bins, patches = plt.hist(y, num_bins,
                            color ='blue',
                            alpha = 0.7, label='stable')
#plt.plot(bins, '--', color ='black')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.ylim(0,l.max()*1.1)
plt.xlim(0,20)
plt.title('Number of coexistent species', fontweight ="bold")
plt.legend()
plt.show()


# plt.figure()
# x = Entropy
# y = Unstableentropy
# num_bins = 100
# n, bins, patches = plt.hist(x, num_bins,
#                             color ='green',
#                             alpha = 0.7, label='all')
# n, bins, patches = plt.hist(y, num_bins,
#                             color ='blue',
#                             alpha = 0.7, label='very unstable')
# #plt.plot(bins, '--', color ='black')
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# plt.ylim(0,10000)
# plt.xlim(0,3)
# plt.title('Entropy', fontweight ="bold")
# plt.legend()
# plt.show()