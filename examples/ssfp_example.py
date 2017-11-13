
from __future__ import division

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import ssfp

x, y = ssfp.SSFP_Spectrum(TE=3.0/1000.0, TR=6.0/1000.0)
plt.plot(x, y)
plt.show()

