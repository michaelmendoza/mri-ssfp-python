
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

""" Implementation of LAMA taken from the following paper: Quist et al. Simulataneous 
		Fat Suppression and Band Reduction with Large Angle Multiple-Acquistion Balanced SSFP, 
		Magnetic Resonance in Medicine, 2012 """

def lama(img, img2, fieldMap, fieldOffset): 
	""" Generates water and fat images from input images, and field map using LAMA """
	beta = fieldMap / 2 + fieldOffset;
	water = img1 * np.cos(beta) + img2 * np.sin(beta)
	fat = img1 * np.sin(beta) - img2 * np.cos(beta)
	return [water, fat]
	