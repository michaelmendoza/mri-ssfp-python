
from __future__ import division

import numpy as np
import math
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from _lib import ssfp

x, y = ssfp.SSFP_Spectrum(TE=3.0/1000.0, TR=6.0/1000.0)
x2, y2 = ssfp.SSFP_Spectrum(TE=6.0/1000.0, TR=12.0/1000.0)
x3, y3 = ssfp.SSFP_Spectrum(TE=12.0/1000.0, TR=24.0/1000.0)
x4, y4 = ssfp.SSFP_Spectrum(TE=3.0/1000.0, TR=6.0/1000.0, dphi = math.pi)
x5, y5 = ssfp.SSFP_Spectrum(TE=6.0/1000.0, TR=12.0/1000.0, dphi = math.pi)
x6, y6 = ssfp.SSFP_Spectrum(TE=12.0/1000.0, TR=24.0/1000.0, dphi = math.pi)

#x = np.stack((x, x2, x3, x4, x5, x6))
#y = np.stack((y, y2, y3, y4, y5, y6))
#plt.plot(x[0,:],y[0,:])


fig, axes = plt.subplots(nrows=6)
axes[0].plot(x, np.abs(y))
axes[1].plot(x2, np.abs(y2))
axes[2].plot(x3, np.abs(y3))
axes[3].plot(x, np.abs(y4))
axes[4].plot(x2, np.abs(y5))
axes[5].plot(x3, np.abs(y6))
plt.show()

'''
plt.ion()
for _ in range(6):
	plt.plot(x[_,:],y[_,:])
	plt.show()
	plt.pause(0.5)
'''

'''
N = 200
L = 2.0;
x = np.linspace(0, 2*L, N)
y = np.ones(N)
n = 24;
y[n:n+50] = 0
y[n+100:n+150] = 0

x, y = ssfp.SSFP_Spectrum(BetaMax = 4 * math.pi)
plt.plot(x,y)
plt.show()


M = 20
c = np.zeros((M+1))
for n in range(0, M):
	integral = np.sum( y * np.cos(n * math.pi * x / L) ) * L / N
	c[n] = (2 / L) * integral

print c

f = c[0] / 2;
for n in range(1, M):
	f = f + c[n] * np.cos(n * math.pi * x / L)

plt.plot(x,f)
plt.show()
'''

