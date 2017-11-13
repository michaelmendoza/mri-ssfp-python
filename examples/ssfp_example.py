
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from lib import ssfp

def example():
	x, y = ssfp.SSFP_Spectrum(TE=3.0/1000.0, TR=6.0/1000.0)
	plt.plot(x, y)
	plt.show()

if __name__ == '__main__':
	example()