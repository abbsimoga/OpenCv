import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("jfk.png", cv2.IMREAD_GRAYSCALE)
kenny = cv2.Canny(img, 100, 200)

titles = ["jfk", "kennyfied"]
images = [img, kenny]

for amount, i in enumerate(titles):
    plt.subplot(1, 2, amount+1), plt.imshow(images[amount], "gray")
    plt.title(titles[amount])
    plt.xticks([]), plt.yticks([])

plt.show()