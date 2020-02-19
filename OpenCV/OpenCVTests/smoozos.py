import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("beezos.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel = np.ones((5,5), np.float32)/25
dst = cv2.filter2D(img, -1, kernel)
blur = cv2.blur(img, (5, 5))
gblur = cv2.GaussianBlur(img, (5, 5), 0)
median = cv2.medianBlur(img, 5)
bilateralFilter = cv2.bilateralFilter(img, 9, 75, 75)

titles = ["beezos", "smoozos", "blurzos", "bhlurh", "medos", "bi-zos"]
images = [img, dst, blur, gblur, median, bilateralFilter]

for amount, i in enumerate(titles):
    plt.subplot(2, 3, amount+1), plt.imshow(images[amount], "gray")
    plt.title(titles[amount])
    plt.xticks([]), plt.yticks([])

plt.show()