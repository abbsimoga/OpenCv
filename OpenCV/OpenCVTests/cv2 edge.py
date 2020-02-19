import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("beezos.png", cv2.IMREAD_GRAYSCALE)
lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
lap = np.uint8(np.absolute(lap))
sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
kenny = cv2.Canny(img, 100, 200)

sobleCombined = cv2.bitwise_or(sobelX, sobelY)

titles = ["beezos", "laplazos", "soblex", "sobley", "combine", "kenny"]
images = [img, lap, sobelX, sobelY, sobleCombined, kenny]

for amount, i in enumerate(titles):
    plt.subplot(2, 3, amount+1), plt.imshow(images[amount], "gray")
    plt.title(titles[amount])
    plt.xticks([]), plt.yticks([])

plt.show()