#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 16:48:36 2022

@author: gijsbartholomeus
"""
import pickle

"""
In this script we calculate response of each entry n_i to a small perturbation (10^-6)
to a Matrix element a_ij
------------------------------------------------------------------------
Loading files
Make sure you run Fixedpoints.py and Feasiblestablepoints.py first
"""

file_name = "Randommatrix.pkl"
open_file = open(file_name, "rb")
G = pickle.load(open_file)
open_file.close()
  
 
file_name = "Stablefeasiblefixedpoints.pkl"
open_file = open(file_name, "rb")
Points = pickle.load(open_file)
open_file.close()

file_name = "Stableentropy.pkl"
open_file = open(file_name, "rb")
H = pickle.load(open_file)
open_file.close()

file_name = "Stableevals.pkl"
open_file = open(file_name, "rb")
Lamdas = pickle.load(open_file)
open_file.close()

