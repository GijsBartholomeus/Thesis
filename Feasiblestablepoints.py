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




"""
Loading files
Make sure you run Fixedpoints first"""
n=20
K = np.ones(n)  

def Jac(N):
    J = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            J[i, j] = -N[i] * G[i,j]
        J[i, i] = K[i] - 2*N[i]  + np.einsum('ij,j', G, N)[i] 
        eigvals, eigvecs = la.eig(J)
    return [eigvecs,eigvals.real]

def Ent(N):
    n = N/N.sum()
    np.seterr(divide='ignore')
    a = -np.nansum(n*np.log(n))
    return(a)
    


file_name = "Randommatrix.pkl"
open_file = open(file_name, "rb")
G = pickle.load(open_file)
open_file.close()
  
file_name = "Fixedpoints.pkl"
open_file = open(file_name, "rb")
FP = pickle.load(open_file)
open_file.close()

"""We filter for feasible solutions (N_i \geq 0)"""

FFP = []
for i in range(2**n):
    if all(j >= 0 for j in FP[i]):
        FFP.append(FP[i])
FFP = np.array(FFP)
file_name = "Feasiblefixedpoints.pkl"
open_file = open(file_name, "wb")
pickle.dump(FFP, open_file)
open_file.close()


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

SFFP = []
start = time.time()
for i in range(len(FFP)):
    eigvals = Jac(FFP[i])[1]
    eigvecs = Jac(FFP[i])[0]
    entropy = Ent(FFP[i])
    Entropy.append(entropy)
    Eigvecs.append(eigvecs)
    Realevals.append(eigvals)
    #Eigvecs.append(eigvecs)
    if all(j >= 0 for j in eigvals):
        SFFP.append(FFP[i])
        Stableevals.append(eigvals)
        Stableeivecs.append(eigvecs)
        Stableentropy.append(entropy)
    if i % 1000 == 0 :
        print(i)
end = time.time()
elapsed_time = end-start
print("Looping took",elapsed_time,"seconds")
SFFP = np.array(SFFP)

"Saving Stable fixed points, eigenvalues and eigenvectors"
file_name = "Stablefeasiblefixedpoints.pkl"
open_file = open(file_name, "wb")
pickle.dump(SFFP, open_file)
open_file.close()
file_name = "Eigenvectors.pkl"
open_file = open(file_name, "wb")
pickle.dump(Eigvecs, open_file)
open_file.close()
file_name = "Eigenvalues.pkl"
open_file = open(file_name, "wb")
pickle.dump(Realevals, open_file)
open_file.close()




    
