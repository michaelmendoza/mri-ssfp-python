
from __future__ import division

import numpy as np
import math
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import ssfp

M0 = 1.0; alpha = math.pi/3.0; phi = 0.0; dphi = 0.0; dphi2 = math.pi;  Nr = 200
T1 = 790.0/1000.0; T2 = 92.0/1000.0
T1F = 270.0/1000.0; T2F = 85.0/1000.0
T1M = 870.0/1000.0; T2M = 47.0/1000.0
Ns = 200; BetaMax = math.pi; f0 = 0.0

beta0 = 0
beta1 = 428
TR = 1 / (2.0 * (beta1 - beta0))
TE = TR / 2.0

f = beta1 * np.ones(Ns)
f[0:int(Ns/2)] = 0
f += 10 * np.random.rand(Ns)


Mc = np.zeros(Ns, 'complex')
Mc2 = np.zeros(Ns, 'complex')
for n in range(Ns): # Interate through Beta values
    Mc[n] = ssfp.SSFP_Signal(M0, alpha, phi, dphi, Nr, TR, TE, T1, T2, f[n] + f0)
    Mc2[n] = ssfp.SSFP_Signal(M0, alpha, phi, dphi2, Nr, TR, TE, T1, T2, f[n] + f0)

ssfp.plot(np.linspace(0,1, Ns), Mc)
ssfp.plot(np.linspace(0,1, Ns), Mc2)


#w = x1 .* cos(t2) + x2 .* sin(t2);
#f = x1 .* sin(t2) - x2 .* cos(t2);

f0, Mc = ssfp.SSFP_SpectrumF(f0=0, f1=428)
ssfp.plot(f0, Mc)


