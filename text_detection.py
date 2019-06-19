# %% [markdown]
# ## Imports
import numpy as np
from PIL import Image
import pytesseract
import cv2
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


def detect_text(img, strucuring_el_size=(17, 3)):
    boundRect = []
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_sobel = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, None, 3, 1, 0,
                          cv2.BORDER_DEFAULT)
    _, img_threshold = cv2.threshold(img_sobel, 0, 255,
                                     cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, strucuring_el_size)
    img_threshold = cv2.morphologyEx(cv2.UMat(img_threshold), cv2.MORPH_CLOSE,
                                     element)
    contours, hier = cv2.findContours(img_threshold, 0, 1)
    contoursPoly = []
    for c in contours:
        contoursPoly.append(cv2.approxPolyDP(c, 3, True))
        rect = cv2.boundingRect(contoursPoly[-1])
        x, y, w, h = rect
        if (w > h):
            boundRect.append(rect)
    return boundRect


def find_text_pytesseract(img):
    img_pil = Image.fromarray(img)
    text = pytesseract.image_to_string(img_pil)
    return text


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
    # file_img = './images/806123698_321554.jpg'  # Good file for testing
    img = imageio.imread(file_img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()

    # Finding corners from input image
    corner_points = CornerDetector(img).find_corners4().astype(np.float32)
    corner_points[:, [0, 1]] = corner_points[:, [1, 0]]

    # Computing the perspective transform
    img_p = PerspectiveTransform(img, corner_points).four_point_transform()

    # * Finding text areas
    img_cv = cv2.cvtColor(img_p, cv2.COLOR_RGB2BGR)
    letter_bboxes = detect_text(img_cv, (30, 10))

    for box in letter_bboxes:
        x, y, w, h = box
        cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 0, 255), 3, 8, 0)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite('./out.png', img_cv)

    img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    for box in letter_bboxes:
        x, y, w, h = box
        print(find_text_pytesseract(img_p[y:y + h + 1, x:x + w + 1]))
