import numpy as np
import cv2

# Calculate Source and Destination points 
def calc_warp_points():
    """
    :return: Source and Destination pointts 
    """
    src = np.float32 ([
        [220, 651],
        [350, 577],
        [828, 577],
        [921, 651]
    ])

    dst = np.float32 ([
            [220, 651],
            [220, 577],
            [921, 577],
            [921, 651]
        ])
    return src, dst


# Calculate Transform 
def calc_transform(src_, dst_):
    """
    Calculate Perspective and Inverse Perspective Transform Matrices 
    :param src_: Source points
    :param dst_: Destination Points
    :return: Perspective Matrix and Inverse Perspective Transform Matrix
    """
    M_ = cv2.getPerspectiveTransform(src_, dst_)
    Minv_ = cv2.getPerspectiveTransform(dst_, src_)
    return M_, Minv_


# Get perspective transform 
def perspective_transform(M_, img_):
    """

    :param M_: Perspective Matrix 
    :param img_ : Input Image
    :return: Transformed Image 
    """
    img_size = (img_.shape[1],img_.shape[0])
    transformed = cv2.warpPerspective(
        img_,
        M_, img_size,
        flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)
    return transformed


# Inverse Perspective Transform 
def inv_perspective_transform(Minv_, img_):
    """

    :param M_: Inverse Perspective Transform Matrix
    :param img_: Input Image
    :return: Transformed Image
    """
    img_size = (img_.shape[1], img_.shape[0])
    transformed = cv2.warpPerspective(
        img_,
        Minv_, img_size,
        flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)
    return transformed

src, dst = calc_warp_points()
M_, Minv_ = calc_transform(src, dst)

print(src, dst, M_, Minv_)

img_ = cv2.imread("eh.jpg")

transformed = perspective_transform(M_, img_)
transformed2 = inv_perspective_transform(Minv_, img_)

cv2.imshow("ransformed", transformed)
cv2.imshow("ransformed2", transformed2)
cv2.imshow("img_", img_)

cv2.waitKey(0)
cv2.destroyAllWindows()