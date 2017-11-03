from __future__ import division

import numpy as np
import math
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

N = 1000
x = np.linspace(0, 10, N)
T = 2
L = T / 2 

def f(x): 
    a = np.sin((np.pi) * x)
    a = np.maximum(a, np.zeros(N))
    return a

def a(n, L):
    a, b = -L, L
    dx = (b - a) / N
    integration = np.sum(f(x) * np.cos((n * np.pi * x) / L)) * dx
    return (1 / L) * integration

def b(n, L):
    a, b = -L, L
    dx = (b - a) / N
    integration = np.sum(f(x) * np.sin((n * np.pi * x) / L)) * dx
    return (1 / L) * integration

# Fourier series 
def Sf(x, L, n = 10):
    a0 = a(0, L)
    sum = np.zeros(np.size(x))
    for i in np.arange(1, n + 1):
        sum += ((a(i, L) * np.cos((i * np.pi * x) / L)) + (b(i, L) * np.sin((i * np.pi * x) / L)))
    return (a0 / 2) + sum   

plt.plot(x, f(x)) 
plt.plot(x, Sf(x, L), '.', color = 'red') 
plt.show() 
