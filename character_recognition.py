# %% [markdown]
# ## Imports
import numpy as np
from PIL import Image
import pytesseract
# %% [markdown]
# ## Functionalities


def rgb_to_grayscale(img):
    """ Converts an RGB image to gray scale.
        Using the ITU-R 601-2 luma transform

        Args:
            img (array-like): array representation of a RGB image.
        Returns:
            numpy.ndarray: Array representation of img, converted to grayscale.
        """
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])


def test_text(img):
    ret_img = np.zeros(img.shape, dtype=np.uint8)
    img_gray = rgb_to_grayscale(img)
    ret_img = img_gray
    return ret_img


# %% [markdown]
# ## Tests
if __name__ == "__main__":
    import imageio
    from corner_detection import CornerDetector
    from perspective_transform import PerspectiveTransform
    import matplotlib.pyplot as plt
    import os

    # Listing example files
    example_files = [
        './images/' + f for f in os.listdir('./images')
        if os.path.isfile(os.path.join('./images', f))
    ]
    # Selecting random file for testing
    file_img = example_files[np.random.randint(0, len(example_files))]
    file_img = './images/806123698_321554.jpg'  # Good file for testing
    img = imageio.imread(file_img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()

    # Finding corners from input image
    corner_points = CornerDetector(
        img).find_corners4().astype(np.float32)
    corner_points[:, [0, 1]] = corner_points[:, [1, 0]]

    # Computing the perspective transform
    # Comparing OpenCV's method with self-made implementation
    img_p = PerspectiveTransform(img, corner_points).four_point_transform()
    plt.figure(figsize=(10, 10))
    plt.imshow(img_p)
    plt.show()

    img_pil = Image.fromarray(img_p)
    text = pytesseract.image_to_string(img_pil)
    print(text)
