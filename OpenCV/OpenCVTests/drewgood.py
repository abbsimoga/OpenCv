import numpy as np
import cv2

# img = cv2.imread("lol.jpg", 1)
img = np.zeros([512, 512, 3], np.uint8)

img = cv2.line(img, (0,0), (255,255), (0,0,255), 10)
img = cv2.arrowedLine(img, (50,50), (255,255), (0,255,0), 10)

img = cv2.rectangle(img, (100, 200), (300,400), (0,0,255), -1)
img = cv2.circle(img, (200, 200), 100, (0,255,255), 10)
font = cv2.FONT_ITALIC
img = cv2.putText(img, "THICC", (50, 200), font, 4, (255, 255, 0), 10, cv2.LINE_AA)

cv2.imshow("beezos", img)

cv2.waitKey(0)
cv2.destroyAllWindows()