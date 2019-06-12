# %% [markdown]
# ## Imports
import numpy as np
import cv2
from corner_detection import CornerDetector

# %%
# Running tests on an random image
# ! This segment of the code is used only for testing purposes
if __name__ == "__main__":
    import imageio
    import matplotlib.pyplot as plt
    import os
    # Listing example files
    example_files = [
        './images/' + f for f in os.listdir('./images')
        if os.path.isfile(os.path.join('./images', f))
    ]
    # Selecting random file for testing
    file_img = example_files[np.random.randint(0, len(example_files))]
    img = imageio.imread(file_img)

    # Finding corners from input image
    corner_points = CornerDetector(img).find_corners4()
    # new_corners = np.array(
    #     [[corner_points[:, 0].min(), corner_points[:, 1].min()],
    #      [corner_points[:, 0].min(), corner_points[:, 1].max()],
    #      [corner_points[:, 0].max(), corner_points[:, 1].min()],
    #      [corner_points[:, 0].max(), corner_points[:, 1].max()]])
    # new_corners = np.array([
    #     [0, 0],
    #     [0, corner_points[:, 1].max() - corner_points[:, 1].min() - 1],
    # ])
    
    matrix = cv2.getPerspectiveTransform(corner_points, new_corners)
    result = cv2.warpPerspective(
        img, matrix, (corner_points[:, 0].max() - corner_points[:, 0].min(),
                      corner_points[:, 1].max() - corner_points[:, 1].min()))
    plt.figure(figsize=(15, 15))
    plt.imshow(result)
    plt.show()
