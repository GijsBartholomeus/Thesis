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
from random import gauss

alpha = gauss(1,1) #mortality rate due to predators
beta = gauss(-1,1)
delta = gauss(1,1)
gamma = gauss(-1,1)
epsilon = gauss(1,1)
theta = gauss(-1,1)
psi = gauss(1,1)
phi = gauss(-1,1)
omicron = gauss(1,1)
omega = gauss(-1,1)
x0 = gauss(1,1)
y0 = gauss(1,1)
z0 = gauss(1,1)
v0 = gauss(1,1)
w0 = gauss(1,1)

def derivative(X, t, alpha, beta, delta, gamma, epsilon, theta,psi,phi,omicron,omega):
    x, y, z,v,w= X
    dotx = x * (alpha + beta * y + theta* z + phi*v + omega*w)
    doty = y * (-delta + gamma * x + theta *z + phi*v + omega*w)
    dotz = z * (epsilon + gamma * x +  beta *y + phi*v + omega*w)
    dotv = z * (psi + gamma * x + beta *y + theta*z + omega*w)
    dotw = z * (omicron + gamma * x + beta *y + phi*v + theta*z)

    return np.array([dotx, doty, dotz,dotv,dotw])

Nt = 1000
tmax = 30.
t = np.linspace(0.,tmax, Nt)
X0 = [x0, y0, z0,v0,w0]
res = integrate.odeint(derivative, X0, t, args = (alpha, beta, delta, gamma, epsilon, theta,psi,phi,omicron,omega))
x, y, z,v,w = res.T

plt.figure()
plt.grid()
plt.title("odeint method")
plt.plot(t, x, 'xb', label = 'Deer')
plt.plot(t, y, '+r', label = "Wolves")
plt.plot(t, z, '+g', label = "Owls")
plt.plot(t, w, '*m', label = "cats")
plt.plot(t, v, '-c', label = "beetles")

plt.xlabel('Time t, [days]')
plt.ylabel('Population')
plt.legend()

plt.show()

