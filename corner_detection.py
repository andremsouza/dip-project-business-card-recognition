#%% [markdown]
# ## Imports
#%%
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import skimage
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

# * Detects corners from img input, using the Harris Corner Detector
def harris_corner_detector(img, offset=1, k=0.095, threshold=0, k_mean=False, eps = 0.001):
	corner_points = []
	ret_img = np.copy(img)

	# Find derivatives and tensor setup
	if(len(img.shape) == 3):
		dx, dy = image_derivatives(rgb_to_grayscale(img))
	elif(len(img.shape) == 2):
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
			if(k_mean):
				k = 2*(((sxx * syy) - (sxy**2))/(sxx + syy + eps))
			r = ((sxx * syy) - (sxy**2)) - k*((sxx + syy)**2)

			# Verify if point is a corner with threshold value
			# If true, add to list of corner points and colorize point on returning image
			if (r > threshold):
				corner_points.append([i, j, r])
				if(len(ret_img.shape) == 3):
					ret_img[i, j] = [255, 0, 0]
				elif(len(ret_img.shape) == 2):
					ret_img[i, j] = 255
				else: raise TypeError("Numpy array with invalid shape")
	return ret_img, corner_points

# * Detects corner from img input, using the Shi-Tomasi Corner Detector
def shi_tomasi_corner_detector(img, offset=1, threshold=0):
	corner_points = []
	ret_img = np.copy(img)

	# Find derivatives and tensor setup
	if(len(img.shape) == 3):
		dx, dy = image_derivatives(rgb_to_grayscale(img))
	elif(len(img.shape) == 2):
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

			# Find determinant and trace, use to get corner response -> r = min(lambda1, lambda2)
			r = np.minimum(sxx, syy)

			# Verify if point is a corner with threshold value
			# If true, add to list of corner points and colorize point on returning image
			if (r > threshold):
				corner_points.append([i, j, r])
				if(len(ret_img.shape) == 3):
					ret_img[i, j] = [255, 0, 0]
				elif(len(ret_img.shape) == 2):
					ret_img[i, j] = 255
				else: raise TypeError("Numpy array with invalid shape")
	return ret_img, corner_points

# * Median filter. For each pixel of img, returns the median value of a region of k pixels around it
def median_denoise(img, k=3):
	img_final = np.copy(img)
	offset = k//2
	for i in range(offset, img.shape[0]-offset):
		for j in range(offset, img.shape[1]-offset):
			img_final[i][j] = np.median(img[i-offset:i+offset+1, j-offset:j+offset+1])
	return img_final
#%%
# Testing functions
img3_blur = skimage.filters.gaussian(img3, multichannel=True)
plt.figure(figsize=(16,16))
img3_stcd, img3_corners = shi_tomasi_corner_detector(img3_blur, offset=1, threshold=1)
plt.imshow(img3_stcd)
#%%
# Analyzing thresholding function for example images
# Listing example files
import os
example_files = ['./images/' + f for f in os.listdir('./images') if os.path.isfile(os.path.join('./images', f))]

# Running tests on an random image
img = imageio.imread(example_files[np.random.randint(0, len(example_files))])
#%%
# Testing threshold functions on image
skimage.filters.try_all_threshold(rgb_to_grayscale(img), figsize=(10, 10))
#%%
# Applying gaussian filter to remove noise
img_blur = skimage.filters.gaussian(img, sigma=1, multichannel=True)
plt.figure(figsize=(16, 16))
plt.imshow(img_blur)
#%%
# Executing HCD
img_hcd, img_hcd_c = harris_corner_detector(img_blur, offset=1, k=0.15)
plt.figure(figsize=(16, 16))
plt.imshow(img_hcd)
#%%
# Executing STCD
img_shcd, img_shcd_c = shi_tomasi_corner_detector(img_blur, offset=1, threshold=100)
plt.figure(figsize=(16, 16))
plt.imshow(img_hcd)
#%%
# * Finds corners of a img, utilizing thresholding and Harris Corner Detector
# * Also will add suport to Shi-Tomasi Corner Detector
def find_corners(img):
	# Preprocessing
	# img = skimage.transform.resize
	pass
#%%
