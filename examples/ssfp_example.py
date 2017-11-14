
from __future__ import division

import math
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from _lib import ssfp

def example():
	x, y = ssfp.SSFP_Spectrum(TE=3.0/1000.0, TR=6.0/1000.0, dphi=math.pi)
	ssfp.plot(x, y)

if __name__ == '__main__':
	example()