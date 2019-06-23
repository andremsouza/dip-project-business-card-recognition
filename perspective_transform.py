# %% [markdown]
# ## Imports
import numpy as np
import cv2
# %%


class PerspectiveTransform:
    """Four-point perspective transformation for an digital image.

    Note:
        It is assumed that the points in pts are in the following order:
        [top-left, top-right, bottom-left, bottom-right]
        Also, the structure of each point should be as follows:
            (x, y), where x represents the column and y represents the line
                of the point in the image.
    Args:
        img (array-like): Source image for transformation.
            May be a grayscale or RGB image.
        pts (array-like): Four points of the source image, as corners of
            the transformation.
    Attributes:
        __img (array-like): Source image for transformation.
            May be a grayscale or RGB image.
        __pts (array-like): Four points of the source image, as corners of
            the transformation.
        __dst_pts (array-like): Four points of the destination image
        __dst_shape (array-like): Shape of the destination image
        __matrix (array-like): matrix to transform the coordinates from source
            image to the output image.
    """

    def __init__(self, img, pts):
        self.__img = np.array(img)
        self.__pts = np.array(pts)
        self.__dst_pts, self.__dst_shape = self.__calc_dst()
        self.__matrix = self.__transform_matrix()

    def __projective_mapping(self, pts):
        """ Compute projective mapping of the four points in pts, by solving
            a linear system
            Used to compute the transform matrix for four-point transform.

        Note:
            Pay close attention to the order of the x,y coordinates
        Args:
            pts (array-like): Four points of the source image, as specified
                in the class documentation.
        Returns:
            numpy.ndarray: Projective mapping of the four points.
        """
        # Solve system of linear equations
        a = np.array([[pts[0, 1], pts[1, 1], pts[2, 1]],
                      [pts[0, 0], pts[1, 0], pts[2, 0]], [1, 1, 1]],
                     dtype=np.double)
        b = np.array([[pts[3, 1]], [pts[3, 0]], [1]], dtype=np.double)
        x = np.linalg.solve(a, b)

        return a * x.T

    def __calc_dst(self):
        """ Calculates the destination points of the four-point transform.
        Used by the transform to get the output image shape.

        Returns:
            numpy.ndarray: Four coordinates of the resulting image.
            numpy.ndarray: Shape of the resulting images
        """
        # Calculating shape and points of the resulting image
        rect = self.__pts.astype(np.float32)
        (tl, tr, bl, br) = rect

        # compute the width of the new image
        widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
        widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image
        heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
        heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
        maxHeight = max(int(heightA), int(heightB))

        # Calculate destination points of the transform
        dst = np.array([[0, 0], [0, maxWidth - 1], [maxHeight - 1, 0],
                        [maxHeight - 1, maxWidth - 1]],
                       dtype=np.float32)
        new_shape = (maxHeight, maxWidth)

        return dst, new_shape

    def __transform_matrix(self):
        """ Compute transformation matrix for four-point transform.

        Returns:
            numpy.ndarray: Matrix for transformation from the coordinates
                of the source image, to the coordinates of the output image
        """
        # Solve system of linear equations to compute projective mappings
        A = self.__projective_mapping(self.__pts)
        B = self.__projective_mapping(self.__dst_pts)

        # Inverting matrix A
        A_inv = np.linalg.inv(A)

        # Computing the transform matrix and returning
        return B @ A_inv

    def __warp(self):
        """Given the image, the transform matrix and the shape of the result,
            warp the source image to generate the result image of the
            four-point transform.

        Note:
            Pay close attention to the order of the x and y coordinates in each
            operation.
        Returns:
            numpy.ndarray: Warped image
        """
        # Declaring new image
        ret_img = np.zeros(
            (self.__dst_shape[0], self.__dst_shape[1], self.__img.shape[2]),
            dtype=np.uint8)

        # Transforming coordinates from source image to new image
        for x in range(self.__img.shape[0]):
            for y in range(self.__img.shape[1]):
                new_pos = self.__matrix @ np.array([[x], [y], [1.0]])
                new_pos = np.round((new_pos / new_pos[2])[:2],
                                   decimals=0).astype(int)

                # if new_pos is in the new image, copy from the source image
                if (new_pos[1] > 0 and new_pos[1] < self.__dst_shape[0] and
                        new_pos[0] > 0 and
                        new_pos[0] < self.__dst_shape[1]):
                    ret_img[new_pos[1], new_pos[0]] = self.__img[x, y]

        # Denoising pixels that are black, due to float conversion
        # for each black pixel img(x, y) = (0, 0, 0), convert this value to the
        # median of the 8-neighborhood
        for x in range(1, ret_img.shape[0] - 1):
            for y in range(1, ret_img.shape[1] - 1):
                if ((ret_img.shape[2] == 1 and ret_img[x, y] == 0) or
                        (ret_img.shape[2] == 3 and
                            (ret_img[x, y] == [0, 0, 0]).all())):
                    ret_img[x, y] = np.median(ret_img[x-1:x+2, y-1:y+2])

        return ret_img

    def four_point_transform(self):
        """ Transform img, using four points in pts.
        The area of the source img between the 4 points pts will be
        transformed to a new rectangular image, obtaining a "bird's eye view".

        Returns:
            numpy.ndarray: Resulting warped image
        """
        # Return warped image
        return self.__warp()

    def four_point_transform_cv2(self, img, pts):
        """ Transform img according to four selected points, into a rectangle
            to obtain a "bird's eye view".
        Utilizes openCV2 functions to obtain the result.

        Note:
            It is assumed that the points in pts are in the following order:
            [top-left, top-right, bottom-left, bottom-right]
            This method should be used only for output comparison.
        Args:
            img (array-like): Array rerpesentation of a digital image
            pts (array-like): Four coordinates of img
        Returns:
            numpy.ndarray: Resulting warped image
        """
        rect = pts.astype(np.float32)
        (tl, tr, bl, br) = rect
        # Compute shape of the destination image
        widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
        widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
        heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
        maxHeight = max(int(heightA), int(heightB))

        # construct the set of destination points to obtain
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [0, maxHeight - 1],
            [maxWidth - 1, maxHeight - 1],
        ],
                       dtype=np.float32)

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        # return the warped image
        return warped


# %%
# Running tests on an random image
# ! This segment of the code is used only for testing purposes
if __name__ == "__main__":
    import corner_detection
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
    file_img = './images/806123698_321554.jpg'  # Good file for testing
    img_s = imageio.imread(file_img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_s)
    plt.show()

    # Finding corners from input image
    corner_points = corner_detection.CornerDetector(
        img_s).find_corners4().astype(np.float32)
    corner_points[:, [0, 1]] = corner_points[:, [1, 0]]

    # Computing the perspective transform
    # Comparing OpenCV's method with self-made implementation
    img2 = cv2.imread(file_img)
    img_p_cv2 = PerspectiveTransform(img_s,
                                     corner_points).four_point_transform_cv2(
                                         img2, corner_points)
    img_p = PerspectiveTransform(img_s, corner_points).four_point_transform()
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img_p_cv2, cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    plt.imshow(img_p)
    plt.show()
