#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 22:15:13 2022

@author: gijsbartholomeus
"""
import numpy as np

a = np.arange(25).reshape(5,5)
b = np.arange(5)
c = np.arange(6).reshape(2,3)

np.einsum('ii', a)
np.einsum(a, [0,0])
np.trace(a)