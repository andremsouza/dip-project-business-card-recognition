#%% [markdown]
# ## Imports
#%%
import sys
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
import skimage
import skimage.segmentation
import skimage.draw
import cv2
#%% [markdown]
# ## Implementing Harris Corner Detector
#%%
# * Converts an RGB image to gray scale, using the ITU-R 601-2 luma transform
def rgb_to_grayscale(img):
	return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

# * Calculates x and y derivatives using the Sobel operator
# ? Study possibility of replacement with np.gradient
def image_derivatives(arr, x = True, y = True):
	kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
	return [scipy.signal.convolve2d(arr, kernel_x, mode='same') if x else None, \
		scipy.signal.convolve2d(arr, kernel_y, mode='same') if y else None]

# * Detects corners from image, using Harris or Shi-Tomasi corner detectors
def corner_detector(img, offset=1, threshold=0, k=0.095, k_mean=False, eps=0.001, mode='shi-tomasi'):
	corner_points = []
	ret_img = None

	# Find derivatives and tensor setup
	if(len(img.shape) == 3):
		ret_img = np.copy(img)
		dx, dy = image_derivatives(rgb_to_grayscale(img))
	elif(len(img.shape) == 2):
		ret_img = np.zeros(img.shape)
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
			if(mode == 'harris'):
				r = ((sxx * syy) - (sxy**2)) - k*((sxx + syy)**2)
			elif(mode == 'shi-tomasi'):
				r = np.minimum(sxx, syy)
			else: raise ValueError("Invalid value for 'mode' variable")

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

# * Finds corners of a img, utilizing OpenCV functions
def find_corners(img):
	# Preprocessing
	# img = skimage.transform.resize
	pass
#%%
# Analyzing thresholding function for example images
# Listing example files
import os
example_files = ['./images/' + f for f in os.listdir('./images') if os.path.isfile(os.path.join('./images', f))]

#%%
# Running tests on an random image
file_img = example_files[np.random.randint(0, len(example_files))]
img = imageio.imread(file_img)
plt.figure(figsize=(10, 10))
plt.imshow(img)
#%%
# # Testing threshold functions on image
# skimage.filters.try_all_threshold(rgb_to_grayscale(img), figsize=(10, 10))
#%%
# Applying denoising filters
if(len(img.shape) == 3):
	img_blur = cv2.fastNlMeansDenoisingColored(img)
	img_blur = skimage.filters.sobel(rgb_to_grayscale(img_blur))
	img_blur = img_blur > skimage.filters.threshold_otsu(img_blur)
elif(len(img.shape) == 2):
	ims_blur = cv2.fastNlMeansDenoising(img)
	img_blur = skimage.filters.sobel(img_blur)
	img_blur = img_blur > skimage.filters.threshold_otsu(img_blur)
plt.figure(figsize=(10, 10))
plt.imshow(img_blur)
#%%
# Executing STCD
img_shcd, img_shcd_c = corner_detector(img_blur, offset=1, threshold=0, mode='shi-tomasi')
plt.figure(figsize=(10, 10))
plt.imshow(img_shcd)
#%%
# Getting the four best corners of the business card, after corner detection
points = np.array([[0, 0, np.inf], [0, 0, np.inf], [0, 0, np.inf], [0, 0, np.inf]])
corners = [[0, 0], [0, img_shcd.shape[1]-1], [img_shcd.shape[0]-1, 0],
					[img_shcd.shape[0]-1, img_shcd.shape[1]-1]]
for c in img_shcd_c:
	dist = np.array([
		scipy.spatial.distance.euclidean(c[:2], corners[0]),
		scipy.spatial.distance.euclidean(c[:2], corners[1]),
		scipy.spatial.distance.euclidean(c[:2], corners[2]),
		scipy.spatial.distance.euclidean(c[:2], corners[3]),
		])
	for i in range(len(dist)):
		if(dist[i] < points[i][2]):
			points[i] = [(c[0]), c[1], dist[i]]


#%%
# Testing shi-tomasi using OpenCV functions
# img = cv2.imread(file_img)
# img = cv2.fastNlMeansDenoisingColored(img)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# corners = cv2.goodFeaturesToTrack(gray, 1024, 0.01, 10)
# corners = np.int0(corners)

# for i in corners:
#     x,y = i.ravel()
#     cv2.circle(img,(x,y),3,255,-1)
# plt.figure(figsize=(10,10))
# plt.imshow(img),plt.show()