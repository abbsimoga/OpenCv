import cv2
import numpy as np
from PIL import ImageGrab

def nothing(x):
    pass

def edgeDetection(frame, blurSize, lowT, highT):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurredFrame = cv2.GaussianBlur(grayFrame, (blurSize, blurSize), 0)
    edgeDetectedFrame = cv2.Canny(blurredFrame, lowT, highT)
    return edgeDetectedFrame

cv2.namedWindow("trash")
cv2.createTrackbar("lowT", "trash", 0, 255, nothing)
cv2.createTrackbar("highT", "trash", 0, 255, nothing)

while True:
    img = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
    img_np = np.array(img)
    videoFrame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    lowT = cv2.getTrackbarPos("lowT","trash")
    highT = cv2.getTrackbarPos("highT","trash")
    
    edgeDetectedFrame = edgeDetection(videoFrame, 3, lowT, highT)
    cv2.imshow('edges', edgeDetectedFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()