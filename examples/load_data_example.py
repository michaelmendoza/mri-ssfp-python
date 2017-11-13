
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import scipy.io as sio

img = sio.loadmat('ssfp_te3.mat')['img']
img2 = sio.loadmat('ssfp_te6.mat')['img']
img3 = sio.loadmat('ssfp_te12.mat')['img']

fig, axes = plt.subplots(ncols=3)
axes[0].imshow(np.abs(img))
axes[1].imshow(np.abs(img2))
axes[2].imshow(np.abs(img3))
plt.show()

