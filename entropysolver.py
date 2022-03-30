#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 22:04:19 2022

@author: gijsbartholomeus
"""

import numpy as np
from scipy import linalg as LA
from random import gauss
from math import sqrt
from sympy import symbols, solve
import matplotlib.pylab as plt
from scipy.optimize import fsolve

"Generate random symmetric matrix"
n = 20  
for k in range(n):                         # dimension of the random matrix
    G = np.zeros((n,n))             # initialize a two-dimensional array
    for i in range(n):
        for j in range(i,n):
            G[i, j] = gauss(0,1)    # random element
            G[j, i] = G[i, j] 

"Generate random species vector, and normalize species abundance"

"We will solve for N_i(K-NG_i + \sum A_ij N_j) "

def matrixsum(k,n):
    i = k-1
    K=1
    N=np.array([n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20])
    #Nj = np.array(N-pop(i))
    Gi = G[i]
    print(Gi)
    
    return(n*(n - K- np.einsum('i,i',N,Gi)))

#N = [n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20]
def equations(p):
    n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20 = p
    return (matrixsum(1,n1),
            matrixsum(2,n2),
            matrixsum(3,n3),
            matrixsum(4,n4),
            matrixsum(5,n5),
            matrixsum(6,n6),
            matrixsum(7,n7),
            matrixsum(8,n8),
            matrixsum(9,n9),
            matrixsum(10,n10),
            matrixsum(11,n11),
            matrixsum(12,n12),
            matrixsum(13,n13),
            matrixsum(14,n14),
            matrixsum(15,n15),
            matrixsum(16,n16),
            matrixsum(17,n17),
            matrixsum(18,n18),
            matrixsum(19,n19),
            matrixsum(20,n20))

n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20  =  fsolve(equations, (1, 1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1))


#A = 
#b = np.zeros(n)
#x = np.linalg.solve(A, b)
x
