from __future__ import division

import numpy as np
import math
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

''' Odd Coefficents '''
def a(x, L, n):
    a, b = -L, L
    N = len(x)
    dx = (b - a) / N
    integration = np.sum(f(x) * np.cos((n * np.pi * x) / L)) * dx
    return (1 / L) * integration

''' Even Coefficents '''
def b(x, L, n):
    a, b = -L, L
    N = len(x)
    dx = (b - a) / N
    integration = np.sum(f(x) * np.sin((n * np.pi * x) / L)) * dx
    return (1 / L) * integration

''' Fourier series '''
def Sf(x, L, n = 10):
    a0 = a(x, L, 0)
    sum = np.zeros(np.size(x))
    for i in np.arange(1, n + 1):
        sum += ((a(x, L, i) * np.cos((i * np.pi * x) / L)) + (b(x, L, i) * np.sin((i * np.pi * x) / L)))
    return (a0 / 2) + sum   

if __name__ == '__main__':

	N = 1000
	x = np.linspace(0, 10, N)
	T = 2
	L = T / 2 

	''' Test Function '''
	def f(x): 
	    a = np.sin((np.pi) * x)
	    a = np.where(a > 0, 1, 0)
	    return a

	plt.plot(x, f(x)) 
	plt.plot(x, Sf(x, L, 6), color = 'red') 
	plt.show() 
