
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

def SSFP_Signal(M0, alpha, phi, dphi, Nr, TR, TE, T1, T2, f0):
		cos = math.cos
		sin = math.sin
		
		beta = 2 * math.pi * f0 * TR
		E1 = math.exp(-TR/T1);
		E2 = math.exp(-TR/T2);
		theta = beta - dphi; # beta = 2*pi*f0*TR;
		Mbottom = (1 - E1 * cos(alpha)) * (1 - E2 * cos(theta)) - E2 * (E1 - cos(alpha)) * (E2 - cos(theta));
		Mx = M0 * (1 - E1) * sin(alpha) * (1 - E2 * cos(theta)) / Mbottom;
		My = M0 * (1 - E1) * E2 * sin(alpha) * sin(theta) / Mbottom;
		Mc = complex(Mx, My)
		Mc = Mc * cmath.exp(complex(0,1) * beta * (TE / TR)) * math.exp(-TE / T2)
		return Mc;

def SSFP_Spectrum(M0 = 1.0, alpha = math.pi/3.0, phi = 0.0, dphi = 0.0, Nr = 200, TR = 10.0/1000.0, TE = 5.0/1000.0, 
                  T1 = 790.0/1000.0, T2 = 92.0/1000.0, Ns = 200, BetaMax = math.pi, f0 = 0.0):

    beta = np.linspace(-BetaMax, BetaMax, Ns)
    f = beta / TR / (2 * math.pi)
    Mc = np.zeros(Ns, 'complex')

    for n in range(Ns): # Interate through Beta values
        Mc[n] = SSFP_Signal(M0, alpha, phi, dphi, Nr, TR, TE, T1, T2, f[n] + f0)

    return f, Mc

def SSFP_SpectrumF(M0 = 1.0, alpha = math.pi/3.0, phi = 0.0, dphi = 0.0, Nr = 200, TR = 10.0/1000.0, TE = 5.0/1000.0, 
                  T1 = 790.0/1000.0, T2 = 92.0/1000.0, Ns = 200, f0=0, f1=100):

    f = np.linspace(f0, f1, Ns)
    Mc = np.zeros(Ns, 'complex')

    for n in range(Ns): # Interate through Beta values
        Mc[n] = SSFP_Signal(M0, alpha, phi, dphi, Nr, TR, TE, T1, T2, f[n])

    return f, Mc



def SpectrumTest():
    T1 = 790.0/1000.0; T2 = 92.0/1000.0;
    T1F = 270.0/1000.0; T2F = 85.0/1000.0
    T1M = 870.0/1000.0; T2M = 47.0/1000.0

    f0, Mc = SSFP_Spectrum(T1=400/1000.0, T2=173.4/1000.0)
    plot(f0, Mc)

def plot(t, x):
    mag = np.absolute(x)
    phase = np.angle(x)

    plt.subplot(211)
    plt.plot(t, mag)
    plt.ylabel('Magitude')
    plt.title('SSPF Sequence')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(t, phase)
    plt.xlabel('Off-Resonance (Hz)')
    plt.ylabel('Phase')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    SpectrumTest()

