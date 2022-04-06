#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 18:27:07 2022

@author: gijsbartholomeus
"""


import numpy as np
import random 
import itertools
import time
import pickle




random.seed(42) # because Farshid wanted 42


n = 20
mean = 4/n
std = 0.8/np.sqrt(n)  
"""           
We generate a matrix with random values sampled from a gaussian distribution with mean 4/n 
and standard deeviation 0.8/sqrt(n)
"""

G = np.zeros((n,n))             # initialize a two-dimensional array
for i in range(n):
    for j in range(i,n):
        G[i, j] = random.gauss(mean,std)    # random element 
        G[j, i] = G[i, j]    
for i in range(n):
    G[i, i] = 0    # set diagonal to zero

K = np.zeros(n)             # initialize a two-dimensional array
for i in range(n):
    K[i]=1
    #K[i] = random.gauss(1,0.5)
    #if K[i]<0:
        #K[i]=0 #negative capacity is not realistic

"""
We calculate fixed points for the LV-equations which are fully determined by persistent species
Therefore we loop over all permutations of possible species. Each fixed point is stored in a list
"""
lst = list(itertools.product([0, 1], repeat=20))
Fixedpoints =[]
start = time.time()
for i in range(2**n):    
    config=lst[i]
    H=np.copy(G)
    C=np.copy(K)
    I = np.identity(sum(config))
    count =0
    for j in range(n):
        if config[j] == 0:
            H = np.delete(H,j-count,0)
            H =np.delete(H,j-count,1)
            C = np.delete(C,j-count)
            count +=1
    """We will solve N = (I+A)^-1 \times K. This gives us the fixed point"""
    AI = I+np.array(H)
    inv = np.linalg.inv(AI) 
    count2 =0
    N = np.einsum('ij,j', inv, C)
    Np = np.zeros(n)    
    for j in range(n):
        if config[j] == 1:
            Np[j]= N[count2]
            count2 +=1
    Fixedpoints.append(Np)
    if i % 100000==0:
        print(i)
end = time.time()
elapsed_time = end-start
print("Looping took",elapsed_time,"seconds")


file_name = "Randommatrix.pkl"
open_file = open(file_name, "wb")
pickle.dump(G, open_file)
open_file.close()

file_name = "Fixedpoints.pkl"
open_file = open(file_name, "wb")
pickle.dump(Fixedpoints, open_file)
open_file.close()



"""We filter for feasible solutions (N_i \geq 0) in the next script Feasiblestablepoints.py"""



# lst = list(itertools.product([0, 1], repeat=20))
# Fixedpoints =[]
# start = time.time()
# for i in range(2**20):    
#     config=lst[i]
#     H=np.copy(G)
#     I = np.identity(20)
#     for j in range(20):
#         if config[j] == 0:
#             for k in range(20):
#                 H[j, k] = 0
#                 H[k, j] = 0
#                 I[j,k] = 0               
#     """We will solve N = (I+A)^-1 \times K. This gives us the fixed point"""
#     AI = I+np.array(H)
#     inv = np.linalg.inv(AI) 
#     N = np.einsum('ij,j', inv, K)
#     Fixedpoints.append(N)
#     if i % 100000==0:
#         print(i)
# end = time.time()
# elapsed_time = end-start

# print(elapsed_time)

#inv = np.linalg.inv(AI) 
#N = np.einsum('ij,j', inv, K)
