# %% [markdown]
# ## Imports
import numpy as np
from PIL import Image
import pytesseract
import cv2
# %% [markdown]
# ## Functionalities


class TextDetector:
    """Detect text on a image, using OpenCV's functions.

    Args:
        img (array-like): Source image for detection.
            The image array should be compatible with OpenCV's methods.
            May be a grayscale or a BGR image (as by OpenCV's standards).
        structuring_el_size (tuple): A tuple of two integers, representing
            the width and height of the structuring element used to detect
            text.
    Attributes:
        __img (array-like): Source image for detection.
            The image array should be compatible with OpenCV's methods.
            May be a grayscale or BGR image (as by OpenCV's standards).
        __bound_rects (list): List of bounding rectangles of areas of the image
            where text was detected.
    """

    def __init__(self, img, structuring_el_size=(17, 3)):
        self.__img = img
        self.__bound_rects = self.__detect_text(structuring_el_size)

    def __detect_text(self, strucuring_el_size=(17, 3)):
        """Detect regions of image that may contain text, using OpenCV's
            functions and morphological image processing methods, such as
            structuring elements and closing operation over the binary
            thresholding of the input image.

            After the closing operation, we utilize OpenCV's fincContours to
            detect lines of text, and filter then only for rectangular areas
            that have their widths larger than their heights.

        Args:
            structuring_el_size (tuple): A tuple of two integers, representing
                the width and height of the structuring element used to detect
                text. The default value is (17, 3).
        Returns:
            list: A list of bounding rectangle tuples, representing the
                bounding boxes around each potential text area of the image.
                The structure of each element is as follows: (x, y, w, h)
                    x: Starting line in the image
                    y: Starting column in the image
                    w: Width of the rectangle
                    h: Height of the rectangle
        """
        # Empty list of bounding rectangles for return.
        boundRect = []

        # Converting image to grayscale.
        if (len(self.__img.shape) == 3):
            img_gray = cv2.cvtColor(self.__img, cv2.COLOR_BGR2GRAY)
        elif (len(self.__img.shape) == 2):
            img_gray = self.__img
        else:
            raise TypeError("Invalid shape for image array.")

        # Using the Sobel filter to find the X derivative of the image,
        # with kernel_size = 3.
        # Other values are the default of the function.
        img_sobel = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, None, 3, 1, 0,
                              cv2.BORDER_DEFAULT)

        # Using Otsu's method for thresholding in the filtered image
        # The result is a binary conversion of the image, highlighting
        # transitions in the horizontal direction.
        _, img_threshold = cv2.threshold(img_sobel, 0, 255,
                                         cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        # Using a rectangle of size structuring_el_size as a sctructuring
        # element, and perform closinf operation over the thresholded image.
        element = cv2.getStructuringElement(cv2.MORPH_RECT, strucuring_el_size)
        img_threshold = cv2.morphologyEx(cv2.UMat(img_threshold),
                                         cv2.MORPH_CLOSE, element)

        # Utilize the Suzuki algorithm to find contours in the image, after
        # the closing operations, using OpenCV's implementation.
        contours, _ = cv2.findContours(img_threshold, 0, 1)

        # For each contour, calculate its bounding rectangle and verify
        # width/height proportions, discarding contours that have
        # width/height > 1 (boxes should be on lines/areas of text).
        contoursPoly = []
        for c in contours:
            contoursPoly.append(cv2.approxPolyDP(c, 3, True))
            rect = cv2.boundingRect(contoursPoly[-1])
            x, y, w, h = rect
            if (w > h):
                boundRect.append(rect)
        return boundRect

    def recognize_text(self):
        """Recognize text from the bounding boxes found in self.__detect_text,
            using each bounding box as input to the Tesseract neural network.
            This method uses pytesseract to recognize the characters in each
            area.

        Note:
            To properly use the Tesseract, we utilize the Pillow library to
            convert self.__img to a Pillow Image
        Returns:
            strs (list): list of strings recognized by the Tesseract.
            bboxes (list): list of bounding boxes recognized in the previous
                method.
            img_bboxes (np.ndarray): Array representation of the input image,
                with red (white) rectangles drawn over each bounding box.
        """
        strs = []
        img_bboxes = self.__img.copy()

        # For each bounding box
        for box in self.__bound_rects:
            x, y, w, h = box
            # Draw rectangle
            if (len(self.__img.shape) == 3):
                cv2.rectangle(img_bboxes, (x, y), (x + w, y + h), (0, 0, 255),
                              3, 8, 0)
            elif (len(self.__img.shape) == 2):
                cv2.rectangle(img_bboxes, (x, y), (x + w, y + h), 255, 3, 8, 0)
            else:
                raise TypeError("Invalid shape for image array.")
            # Find text
            img_pil = Image.fromarray(self.__img[y:y + h + 1, x:x + w + 1])
            strs.append(pytesseract.image_to_string(img_pil))

        return strs, self.__bound_rects, img_bboxes


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

    # Finding text areas
    img_cv = cv2.cvtColor(img_p, cv2.COLOR_RGB2BGR)
    # Testing with different structuring element sizes
    sizes = [(17, 3), (30, 10), (5, 5), (9, 3)]
    for size in sizes:
        strs, bound_rects, img_bboxes = TextDetector(img_cv,
                                                     size).recognize_text()
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img_bboxes, cv2.COLOR_BGR2RGB))
        plt.show()
        print(size)
        print(*strs, sep='\n')
