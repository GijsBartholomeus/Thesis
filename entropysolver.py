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
import matplotlib.pylab as plt

"Generate random symmetric matrix"
n=20
for k in range(n):                         # dimension of the random matrix
    G = np.zeros((n,n))             # initialize a two-dimensional array
    for i in range(n):
        for j in range(i,n):
            G[i, j] = gauss(0,1)    # random element
            G[j, i] = G[i, j]   

"Generate random species vector, and normalize species abundance"

"We will solve for N_i(K-N_i) "