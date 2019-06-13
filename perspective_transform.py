# %% [markdown]
# ## Imports
import numpy as np
import cv2
from corner_detection import CornerDetector
# %%


def four_point_transform(img, src, dst):
    """ Transform img, using four points in src, to four points in dst
    # TODO: Docstring
    """
    pass


def four_point_transform_cv2(image, pts):
    """ Transform img according to the 4 points in pts

    Note:
        It is assumed that the points in pts are in the following order:
        [top-left, top-right, bottom-left, bottom-right]
    # TODO: Docstring
    """
    rect = pts.astype(np.float32)
    (tl, tr, bl, br) = pts

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [0, maxHeight - 1],
        [maxWidth - 1, maxHeight - 1],
    ], dtype=np.float32)

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


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
    # file_img = './images/806123698_321554.jpg'  # Good file for testing
    img = imageio.imread(file_img)

    # Finding corners from input image
    corner_points = CornerDetector(img).find_corners4().astype(np.float32)
    corner_points[:, [0, 1]] = corner_points[:, [1, 0]]
    img2 = cv2.imread(file_img)
    result = four_point_transform_cv2(img2, corner_points)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()
    # cv2.imwrite("./out.png", result)
    # TODO: Test my implementation
