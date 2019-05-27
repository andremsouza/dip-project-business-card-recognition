#%% [markdown]
# ## Imports
#%%
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import scipy.signal
#%% [markdown]
# ## Visualizing examples
#%%
# ! Remove cell after finishing initial tests
img1 = imageio.imread("./images/ex1.jpg")
img2 = imageio.imread("./images/ex2.png")
img3 = imageio.imread("./images/ex3.jpg")
plt.figure(figsize=(16, 9))
plt.subplot(131); plt.imshow(img1)
plt.subplot(132); plt.imshow(img2)
plt.subplot(133); plt.imshow(img3)
plt.show()

#%% [markdown]
# ## Implementing Harris Corner Detector
#%%
# * Converts an RGB image to gray scale, using the ITU-R 601-2 luma transform
def rgb_to_grayscale(img):
	return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

# * Calculates x and y derivatives using Sobel operator
# ? Study possibility of replacement with np.gradient
def image_derivatives(arr, x = True, y = True):
	kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
	return [scipy.signal.convolve2d(arr, kernel_x, mode='same') if x else None, \
		scipy.signal.convolve2d(arr, kernel_y, mode='same') if y else None]

# ! Remove this function after done testing
# * Calculates gradient of image array
def image_gradient(arr):
	deriv = image_derivatives(arr)
	return np.power(np.power(deriv[0], 2) + np.power(deriv[1], 2), 1/2)

# * Calculates and returns tensor setup, from image derivatives
# * Important to remember: i_x * i_y == i_y * i_x
def tensor_setup(i_x, i_y):
	return [i_x ** 2, i_x * i_y, i_y **2]

# TODO: Harris response calculation
# TODO: Find edges and corners using R

#%%

#%%
