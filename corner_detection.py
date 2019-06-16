# %% [markdown]
# ## Imports
# %%
import numpy as np
import scipy
import skimage
import cv2


# %%
class CornerDetector:
    """Corner detector for an image.

    Args:
        img (array-like): matrix representation of input image.
            May be a grayscale or RGB image.
    Attributes:
        img (numpy.ndarray): numpy array of image input image representation.
    """

    def __init__(self, img):
        self.__img = np.array(img)

    def rgb_to_grayscale(self, img):
        """ Converts an RGB image to gray scale.
        Using the ITU-R 601-2 luma transform

        Args:
            img (array-like): array representation of a RGB image.
        Returns:
            numpy.ndarray: Array representation of img, converted to grayscale.
        """
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

    def image_derivatives(self, arr, x=True, y=True):
        """ Calculates x and y derivatives using the Sobel operator,
        with convolution using Scipy.

        Args:
            arr (array-like): An array representation of a grayscale image.
            x (bool): True to calculate the X-derivative, else False
            y (bool): True to calculate the Y-derivative, else False.
        Returns:
            numpy.ndarray: X-derivative of arr if x = True, else None.
            numpy.ndarray: Y-derivative of arr if y = True, else None.
        """
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        deriv_x, deriv_y = None, None
        if (x):
            deriv_x = scipy.signal.convolve2d(arr, kernel_x, mode='same')
        if (y):
            deriv_y = scipy.signal.convolve2d(arr, kernel_y, mode='same')
        return deriv_x, deriv_y

    def __preprocess(self):
        """
        Pre-processing the image, with denoising using openCV2,
        and thresholding with the Sobel filter and threshold_otsu
        from Scikit-image.

        Returns:
            numpy.ndarray: Pre-processed image for self.corner_detector method.
        """
        img_p = None
        if (len(self.__img.shape) == 3):
            img_p = cv2.fastNlMeansDenoisingColored(self.__img)
            img_p = skimage.filters.sobel(self.rgb_to_grayscale(img_p))
            img_p = img_p > skimage.filters.threshold_otsu(img_p)
        elif (len(self.__img.shape) == 2):
            img_p = cv2.fastNlMeansDenoising(self.__img)
            img_p = skimage.filters.sobel(img_p)
            img_p = img_p > skimage.filters.threshold_otsu(img_p)
        return img_p

    def corner_detector(self,
                        offset=1,
                        threshold=0,
                        k=0.06,
                        k_mean=False,
                        eps=0.001,
                        mode='shi-tomasi'):
        """ Corner detection method.
        Uses Harris Corner Detector or Shi-Tomasi Corner Detector.

        Note:
            This method calls the self.__preprocess method before applying the
            Harris/Shi-Tomasi corner detector on the resulting image.
        Args:
            offset (int): Offset to center of analyzed regions around a pixel.
                Equals the integer division of the size of the region by two.
            threshold (float): Threshold of corner response measure.
                The higher the limit, the fewer points will be returned.
            k (float): Harris detector parameter
                Should be around 0.04 to 0.06.
            k_mean (bool): Determines if k should be automatically computed.
            eps (float): Small value (around 0.001) for k computation.
                Only relevant if k_mean = True.
            mode (str): 'harris' or 'shi-tomasi'.
                Selector between Harris and Shi-Tomasi Corner Detectors.
        Returns:
            numpy.ndarray: Input image, with marked regions identified
                as corners.
            numpy.ndarray: List of points identified as corners.
                Structure: [x, y, E], where x and y are the coordinates,
                    and E is the corner response measure of the point.
        """
        corner_points = []
        ret_img = np.copy(self.__img)

        # Preprocessing image with thresholding
        img_p = self.__preprocess()

        # Find derivatives and tensor setup
        # Create image for return, illustrating corner points
        if (len(img_p.shape) == 3):
            dx, dy = self.image_derivatives(self.rgb_to_grayscale(img_p))
        elif (len(img_p.shape) == 2):
            dx, dy = self.image_derivatives(img_p)
        else:
            raise TypeError("Numpy array with invalid shape")
        ixx, ixy, iyy = dx**2, dx * dy, dy**2

        # Iterate through windows
        for i in range(offset, self.__img.shape[0] - offset):
            for j in range(offset, self.__img.shape[1] - offset):
                # Calculate sum over the sliding window
                sxx = np.sum(ixx[i - offset:i + offset + 1, j - offset:j +
                                 offset + 1])
                syy = np.sum(iyy[i - offset:i + offset + 1, j - offset:j +
                                 offset + 1])
                sxy = np.sum(ixy[i - offset:i + offset + 1, j - offset:j +
                                 offset + 1])

                # Find determinant and trace,
                # use to get corner response -> r = det - k*(trace**2)
                det = ((sxx * syy) - (sxy**2))
                trace = sxx + syy
                if (k_mean):
                    k = 2 * (det / (trace + eps))
                if (mode == 'harris'):
                    r = det - k * (trace**2)
                elif (mode == 'shi-tomasi'):
                    r = np.minimum(sxx, syy)
                else:
                    raise ValueError("Invalid value for 'mode' variable")

                # Verify if point is a corner with threshold value
                # If true, add to list of corner points and colorize point
                # on returning image
                if (r > threshold):
                    corner_points.append([i, j, r])
                    if (len(ret_img.shape) == 3):
                        ret_img[i, j] = [255, 0, 0]
                    elif (len(ret_img.shape) == 2):
                        ret_img[i, j] = 255
                    else:
                        raise TypeError("Numpy array with invalid shape")
        return ret_img, np.array(corner_points)

    def find_corners4(self,
                      offset=1,
                      threshold=0,
                      k=0.06,
                      k_mean=False,
                      eps=0.001,
                      mode='shi-tomasi'):
        """
        Find the corner points nearest to the corners of the input image,
            using self.corner_detector.

        Args:
            offset (int): Offset to center of analyzed regions around a pixel.
                Equals the integer division of the size of the region by two.
            threshold (float): Threshold of corner response measure.
                The higher the limit, the fewer points will be returned.
            k (float): Harris detector parameter
                Should be around 0.04 to 0.06.
            k_mean (bool): Determines if k should be automatically computed.
            eps (float): Small value (around 0.001) for k computation.
                Only relevant if k_mean = True.
            mode (str): 'harris' or 'shi-tomasi'.
                Selector between Harris and Shi-Tomasi Corner Detectors.
        Returns:
            numpy.ndarray: Array of coordinates of the four identified corners
                of the object.

        """
        img_cd, img_cd_c = self.corner_detector(offset, threshold, k, k_mean,
                                                eps, mode)

        # Getting the four best corners of the business card, after corner
        # detection
        points = np.array([[0, 0, np.inf], [0, 0, np.inf], [0, 0, np.inf],
                           [0, 0, np.inf]])
        corners = [[0, 0], [0, img_cd.shape[1] - 1], [img_cd.shape[0] - 1, 0],
                   [img_cd.shape[0] - 1, img_cd.shape[1] - 1]]
        for c in img_cd_c:
            # Getting distances from c to the corners of the image
            dist = np.array([
                scipy.spatial.distance.euclidean(c[:2], corners[0]),
                scipy.spatial.distance.euclidean(c[:2], corners[1]),
                scipy.spatial.distance.euclidean(c[:2], corners[2]),
                scipy.spatial.distance.euclidean(c[:2], corners[3]),
            ])
            # Limiting each element from points to a quadrant of the image
            if (dist[0] < points[0][2] and c[0] < img_cd.shape[0] // 2 and
                    c[1] < img_cd.shape[1] // 2):
                points[0] = [(c[0]), c[1], dist[0]]
            if (dist[1] < points[1][2] and c[0] < img_cd.shape[0] // 2 and
                    c[1] > img_cd.shape[1] // 2):
                points[1] = [(c[0]), c[1], dist[1]]
            if (dist[2] < points[2][2] and c[0] > img_cd.shape[0] // 2 and
                    c[1] < img_cd.shape[1] // 2):
                points[2] = [(c[0]), c[1], dist[2]]
            if (dist[3] < points[3][2] and c[0] > img_cd.shape[0] // 2 and
                    c[1] > img_cd.shape[1] // 2):
                points[3] = [(c[0]), c[1], dist[3]]
        return points[:, :2]


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
    plt.figure(figsize=(20, 20))
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(CornerDetector(img).corner_detector()[0])
    print(CornerDetector(img).find_corners4())
