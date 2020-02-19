import cv2
import numpy as np 

def nothing(x):
    print(x)

cv2.namedWindow("image")

cv2.createTrackbar("CP", "image", 10, 400, nothing)

switch = "colow/gray"
cv2.createTrackbar(switch, "image", 0, 1, nothing)

while True:
    img = cv2.imread("beezos.png")
    pos = cv2.getTrackbarPos("CP", "image")

    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img, str(pos),(50,150),font,5,(0,0,255),10)
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break
    
    s = cv2.getTrackbarPos(switch, "image")

    if s == 0:
        pass
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.imshow("image",img)

cv2.destroyAllWindows()