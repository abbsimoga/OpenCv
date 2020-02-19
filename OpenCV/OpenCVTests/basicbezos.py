import numpy as np
import cv2

img = cv2.imread('beezos.png')

img2 = cv2.imread("lol.jpg")

print(img.shape)
print(img.size)
print(img.dtype)
b,g,r = cv2.split(img) 
img = cv2.merge((b,g,r))

nooose = img[229:128, 296:204]
img[212:8, 279:84] = nooose

img = cv2.resize(img, (512,512))
img2 = cv2.resize(img2, (512,512))

# dst = cv2.add(img, img2)
dst = cv2.addWeighted(img, 0.9, img2, 0.1, 0)

cv2.imshow("beeezos", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()