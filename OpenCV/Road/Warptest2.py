import numpy as np
import cv2
from PIL import ImageGrab

width = 1920
height = 1080

def nothing(x):
    pass

cv2.namedWindow('src')
cv2.namedWindow('dst')

cv2.createTrackbar('src1','src',220,width,nothing)
cv2.createTrackbar('src2','src',651,width,nothing)
cv2.createTrackbar('src3','src',350,width,nothing)
cv2.createTrackbar('src4','src',577,width,nothing)
cv2.createTrackbar('src5','src',828,width,nothing)
cv2.createTrackbar('src6','src',577,width,nothing)
cv2.createTrackbar('src7','src',921,width,nothing)
cv2.createTrackbar('src8','src',651,width,nothing)

cv2.createTrackbar('dst1','dst',220,width,nothing)
cv2.createTrackbar('dst2','dst',651,width,nothing)
cv2.createTrackbar('dst3','dst',220,width,nothing)
cv2.createTrackbar('dst4','dst',577,width,nothing)
cv2.createTrackbar('dst5','dst',921,width,nothing)
cv2.createTrackbar('dst6','dst',577,width,nothing)
cv2.createTrackbar('dst7','dst',921,width,nothing)
cv2.createTrackbar('dst8','dst',651,width,nothing)


while True:
    img = ImageGrab.grab(bbox=(0, 0, width, height))
    img_np = np.array(img)
    videoFrame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)


    src1 = [int(cv2.getTrackbarPos('src1','src')), int(cv2.getTrackbarPos('src2','src'))]
    src2 = [int(cv2.getTrackbarPos('src3','src')), int(cv2.getTrackbarPos('src4','src'))]
    src3 = [int(cv2.getTrackbarPos('src5','src')), int(cv2.getTrackbarPos('src6','src'))]
    src4 = [int(cv2.getTrackbarPos('src7','src')), int(cv2.getTrackbarPos('src8','src'))]

    dst1 = [int(cv2.getTrackbarPos('dst1','dst')), int(cv2.getTrackbarPos('dst2','dst'))]
    dst2 = [int(cv2.getTrackbarPos('dst3','dst')), int(cv2.getTrackbarPos('dst4','dst'))]
    dst3 = [int(cv2.getTrackbarPos('dst5','dst')), int(cv2.getTrackbarPos('dst6','dst'))]
    dst4 = [int(cv2.getTrackbarPos('dst7','dst')), int(cv2.getTrackbarPos('dst8','dst'))]

    src_ = np.float32 ([src1, src2, src3, src4])
    dst_ = np.float32 ([dst1, dst2, dst3, dst4])

    M_ = cv2.getPerspectiveTransform(src_, dst_)

    transformed = cv2.warpPerspective(videoFrame, M_, (width,height), flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)

    cv2.imshow('original', videoFrame)
    cv2.imshow('transformed', transformed)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cv2.destroyAllWindows()