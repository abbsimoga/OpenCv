import cv2
import numpy as np

img = cv2.imread("beezos.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread("beezosface.png", 0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.9
loc = np.where(res>=threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0,255,0), 2)

cv2.imshow("lol", img)
cv2.waitKey(0)
cv2.destroyAllWindows()