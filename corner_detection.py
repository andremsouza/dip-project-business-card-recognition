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

def to_rgb1a(im):
    # This should be fsater than 1, as we only
    # truncate to uint8 once (?)
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  im
    return ret

# * Calculates x and y derivatives using Sobel operator
# ? Study possibility of replacement with np.gradient
def image_derivatives(arr, x = True, y = True):
	kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
	return [scipy.signal.convolve2d(arr, kernel_x, mode='same') if x else None, \
		scipy.signal.convolve2d(arr, kernel_y, mode='same') if y else None]

# * Detects corners from img input, using the Harris Corner Detector
def harris_corner_detector(img, w_size=3, k=0.05, threshold=0):
	corner_points = []
	offset = w_size // 2

	# Find derivatives and tensor setup
	if(len(img.shape) == 3):
		ret_img = np.copy(img)
		dx, dy = image_derivatives(rgb_to_grayscale(img))
	elif(len(img.shape) == 2):
		ret_img = to_rgb1a(img)
		dx, dy = image_derivatives(img)
	else: raise TypeError("Numpy array with invalid shape")
	# dx, dy = np.gradient(rgb_to_grayscale(img))
	ixx, ixy, iyy = dx**2, dx*dy, dy**2
	
	# Iterate through windows
	for i in range(offset, img.shape[0]-offset):
		for j in range(offset, img.shape[1]-offset):
			# Calculate sum over the sliding window
			sxx = np.sum(ixx[i-offset:i+offset+1, j-offset:j+offset+1])
			syy = np.sum(iyy[i-offset:i+offset+1, j-offset:j+offset+1])
			sxy = np.sum(ixy[i-offset:i+offset+1, j-offset:j+offset+1])

			# Find determinant and trace, use to get corner response -> r = det - k*(trace**2)
			r = ((sxx * syy) - (sxy**2)) - k*((sxx + syy)**2)

			# Verify if point is a corner with threshold value
			# If true, add to list of corner points and colorize point on returning image
			if (r > threshold):
				corner_points.append([i, j, r])
				ret_img[i, j] = [255, 0, 0]
	return ret_img, corner_points
#%%

#%%