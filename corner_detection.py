#%% [markdown]
# ## Imports
#%%
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy
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
# * This function converts an RGB image to gray scale
# * Using the ITU-R 601-2 luma transform
def rgb_to_grayscale(img):
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

# TODO: Spatial derivative calculation

# TODO: Structure tensor setup
# TODO: Harris response calculation
# TODO: Find edges and corners using R

#%%

#%%
