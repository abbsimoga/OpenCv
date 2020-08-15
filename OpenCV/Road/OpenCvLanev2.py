import cv2 # Steering angle with two lines. Then merge the two lines into middel of road and steer by that line. Highlight Exactly the road ie able to handle turns and make line through that and steer by that line # gör sväng höger sväng vänster car pov till en funktion som är middle of lines
import numpy as np
from PIL import ImageGrab
import math

def edgeDetection(frame, blurSize, lowT, highT):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurredFrame = cv2.GaussianBlur(grayFrame, (blurSize, blurSize), 0)
    edgeDetectedFrame = cv2.Canny(blurredFrame, lowT, highT)
    return edgeDetectedFrame

def regionOfInterest(frame, cord1, cord2, cord3, cord4):
    FOV = np.array([[cord1, cord2, cord3, cord4]])
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, FOV, 255)
    reducedFOVFrame = cv2.bitwise_and(frame, mask)
    return reducedFOVFrame

def makeCoordinates(frame, lines, priorCords):
    lineImage = np.zeros_like(frame)
    cords = []
    for line, priorLine in zip(lines, priorCords):
        if np.isnan(line).any():
            cords.append(priorLine)
            lineImage = drawRoadLines(lineImage, priorLine[:4], 2, 255, 255, 0)
        else:
            slope, intercept = line
            y1 = frame.shape[0]
            y2 = int(y1*(3.25/5))
            x1 = int((y1-intercept)/slope)
            x2 = int((y2-intercept)/slope)
            y3 = int(y1*(4/5))
            x3 = int((y3-intercept)/slope)
            cordsAverage = np.average(np.array([np.array([x1,y1,x2,y2,x3,y3]), priorLine]), axis=0).astype(int)
            cords.append(cordsAverage) 
            lineImage = drawRoadLines(lineImage, cordsAverage[:4], 2, 0, 255, 0)
    
    resultingFrame = cv2.addWeighted(frame, 1, lineImage, 1, 1)
    return cords, resultingFrame

def stabilizeCoordinates(newCords, oldCords):
    array = np.array([newCords, oldCords])
    return np.average(array, axis=0).astype(int)

def averageByK(lines):
    leftLines = []
    rightLines = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            leftLines.append((slope, intercept))
        else:
            rightLines.append((slope, intercept))
    leftLinesAverage = np.average(leftLines, axis=0)
    rightLinesAverage = np.average(rightLines, axis=0)
    return np.array([leftLinesAverage, rightLinesAverage])

def drawRoadLines(frame, cords, thicc, r, g, b):
    cv2.line(frame, (cords[0], cords[1]), (cords[2], cords[3]), (b, g, r), thicc)
    return frame

def drawLines(frame, cords, thicc, r, g, b):
    lineImage = np.zeros_like(frame)
    cv2.line(lineImage,  (cords[0], cords[1]), (cords[2], cords[3]), (b, g, r), thicc)
    return lineImage

def highlightRoad(frame, lines):
    cords = np.array([[lines[0][0],lines[0][1]], [lines[0][2],lines[0][3]], [lines[1][2],lines[1][3]], [lines[1][0], lines[1][1]]])
    roadImage = np.zeros_like(frame)
    cv2.fillPoly(roadImage, pts = [cords], color=(255, 255, 0))
    return roadImage

def clickEvent(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        roadCords.append((x,y))

def steeringAngle(frame, lines):
    height, width, _ = frame.shape
    
    left_x2 = lines[0][0]
    right_x2 = lines[1][0]

    mid = int(width / 2)
    x_offset = (left_x2 + right_x2) / 2 - mid
    
    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan2(x_offset, y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90

    print("x_offset", x_offset, "angle_to_mid_radian", angle_to_mid_radian, "angle_to_mid_deg", angle_to_mid_deg, "steering_angle", steering_angle)

    return steering_angle

def steeringAnglev2(lines):
    x1,y1,x2,y2 = lines[:4]
    Angle = 180/math.pi * math.atan((y2-y1)/(x2-x1))
    return Angle

roadCords = []

def nothing(x):
    pass

cv2.namedWindow("trash")
cv2.createTrackbar("lowT", "trash", 220, 255, nothing)
cv2.createTrackbar("highT", "trash", 255, 255, nothing)

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

while True:
    img = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
    img_np = np.array(img)
    videoFrame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    cv2.imshow('raw', videoFrame)
    cv2.setMouseCallback("raw", clickEvent)
    if cv2.waitKey(1) & 0xFF == ord('q') or len(roadCords) == 4:
        break

cv2.destroyAllWindows()

if len(roadCords) != 4:
    roadCords = [(1025, 650), (900, 650), (300,950), (1600,950)]

cords = [np.array([roadCords[0][0],roadCords[0][1],roadCords[1][0],roadCords[1][1],(roadCords[0][0]+roadCords[1][0])/2,(roadCords[0][1]+roadCords[1][1])/2]), np.array([roadCords[2][0],roadCords[2][1],roadCords[3][0],roadCords[3][1],(roadCords[2][0]+roadCords[3][0])/2,(roadCords[2][1]+roadCords[3][1])/2])] # roadcords

LANEFOLLOW_VERSION = True

birdsEyeCords = np.float32([[roadCords[1][0], roadCords[1][1]], [roadCords[2][0], roadCords[2][1]], [roadCords[0][0], roadCords[0][1]], [roadCords[3][0], roadCords[3][1]]])

M = cv2.getPerspectiveTransform(birdsEyeCords, np.float32([[0, 0], [1920, 0], [0, 1080], [1920, 1080]]))

def clickEv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global LANEFOLLOW_VERSION
        LANEFOLLOW_VERSION = not LANEFOLLOW_VERSION

while True:
    img = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
    img_np = np.array(img)
    videoFrame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    birdsEye = cv2.warpPerspective(videoFrame, M, (1920,1080), flags=cv2.WARP_FILL_OUTLIERS + cv2.INTER_CUBIC)

    edgeDetectedFrame = edgeDetection(videoFrame, 3, lowT, highT)
    reducedFOVFrame = regionOfInterest(edgeDetectedFrame, *roadCords)

    lines = cv2.HoughLinesP(reducedFOVFrame, 2, np.pi/180, 100, np.array([]), minLineLength=15, maxLineGap=275)

    if lines is None:
        continue
    line = averageByK(lines)
    
    cords, resultingFrame = makeCoordinates(videoFrame, line, cords)
    highlightedRoad = highlightRoad(videoFrame, cords)
    resultingFrame = cv2.addWeighted(resultingFrame, 1, highlightedRoad, 0.1, 1)

    font = cv2.FONT_ITALIC
    
    if LANEFOLLOW_VERSION:
        highlightedCarPOV = drawLines(resultingFrame, [int(videoFrame.shape[1]/2), int(videoFrame.shape[0]), int(videoFrame.shape[1]/2), int(videoFrame.shape[0]*4/5)], 2, 255, 255, 0)    
        resultingFrame = cv2.addWeighted(resultingFrame, 1, highlightedCarPOV, 1, 1)

        lineImage = np.zeros_like(videoFrame)
        for x1, y1, x2, y2, x3, y3 in cords:
            cv2.line(lineImage, (x3, y3+20), (x3, y3-20), (0, 255, 0), 2)
        resultingFrame = cv2.addWeighted(resultingFrame, 1, lineImage, 1, 1)

        lineImage = np.zeros_like(videoFrame)
        cv2.line(lineImage, (int((cords[0][4]+cords[1][4])/2), cords[0][5]+20), (int((cords[0][4]+cords[1][4])/2), cords[0][5]-20), (0, 255, 0), 2)
        resultingFrame = cv2.addWeighted(resultingFrame, 1, lineImage, 1, 1)

        lineImage = np.zeros_like(videoFrame)
        cv2.line(lineImage, (int((cords[0][4]+cords[1][4])/2), cords[0][5]), (int(videoFrame.shape[1]/2), cords[0][5]), (255, 255, 255), 2)
        resultingFrame = cv2.addWeighted(resultingFrame, 1, lineImage, 1, 1)

        if int((cords[0][4]+cords[1][4])/2) - int(videoFrame.shape[1]/2) < -15:
            textSize = cv2.getTextSize("Turn Left", font, 1, 2)[0][0]
            resultingFrame = cv2.putText(resultingFrame, "Turn Left", (int(videoFrame.shape[1]/2)-int(textSize/2), videoFrame.shape[0]-100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        elif int((cords[0][4]+cords[1][4])/2) - int(videoFrame.shape[1]/2) > 15:
            textSize = cv2.getTextSize("Turn Right", font, 1, 2)[0][0]
            resultingFrame = cv2.putText(resultingFrame, "Turn Right", (int(videoFrame.shape[1]/2)-int(textSize/2), videoFrame.shape[0]-100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        newSteering = steeringAngle(resultingFrame, cords)
        textSize = cv2.getTextSize(str(newSteering), font, 1, 2)[0][0]
        resultingFrame = cv2.putText(resultingFrame, str(newSteering), (int(videoFrame.shape[1]/2)-int(textSize/2), 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # newSteering = steeringAnglev2(cords)
        # textSize = cv2.getTextSize(str(newSteering), font, 1, 2)[0][0]
        # resultingFrame = cv2.putText(resultingFrame, str(newSteering), (int(videoFrame.shape[1]/2)-int(textSize/2), 200), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    
    stackedFramesDriver = cv2.cvtColor(np.concatenate((cv2.cvtColor(np.concatenate((cv2.pyrDown(edgeDetectedFrame), cv2.pyrDown(reducedFOVFrame)), axis=1), cv2.COLOR_BGR2RGB), cv2.cvtColor(np.concatenate((cv2.pyrDown(videoFrame), cv2.pyrDown(resultingFrame)), axis=1), cv2.COLOR_BGR2RGB)), axis=0), cv2.COLOR_BGR2RGB)
    # stackedFramesDriver = cv2.cvtColor(np.concatenate((cv2.cvtColor(np.concatenate((cv2.pyrDown(edgeDetectedFrame), cv2.pyrDown(reducedFOVFrame)), axis=1), cv2.COLOR_BGR2RGB), cv2.cvtColor(np.concatenate((cv2.pyrDown(birdsEye), cv2.pyrDown(resultingFrame)), axis=1), cv2.COLOR_BGR2RGB)), axis=0), cv2.COLOR_BGR2RGB)

    cv2.imshow('birdsEye', birdsEye)
    cv2.imshow('stackedFramesDriver', stackedFramesDriver)



    cv2.setMouseCallback("result", clickEv)
    cv2.setMouseCallback("stackedFramesDriver", clickEv)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

    # cv2.imshow('stackedFramesBirdsEye', stackedFramesBirdsEye)
    # cv2.imshow('stackedFramesCar', stackedFramesCar)

cv2.destroyAllWindows()