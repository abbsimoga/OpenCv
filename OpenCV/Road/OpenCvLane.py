import cv2 # Steering angle with two lines. Then merge the two lines into middel of road and steer by that line. Highlight Exactly the road ie able to handle turns and make line through that and steer by that line

# gör sväng höger sväng vänster car pov till en funktion som är middle of lines

import numpy as np
from PIL import ImageGrab
import math
# from numba import jit, cuda 

# @jit(target = "cuda")

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
            lineImage = drawRoadLines(lineImage, priorLine, 2, 0, 255, 255)
            print("yellow", np.array([x1,y1,x2,y2]))
        else:
            slope, intercept = line
            y1 = frame.shape[0]
            y2 = int(y1*(3.25/5))
            x1 = int((y1-intercept)/slope)
            x2 = int((y2-intercept)/slope)
            y3 = int(y1*(4/5))
            x3 = int((y3-intercept)/slope)
            cords.append(np.array([x1,y1,x2,y2,x3,y3])) 
            lineImage = drawRoadLines(lineImage, np.array([x1,y1,x2,y2]), 2, 0, 255, 0)
            print("greene", np.array([x1,y1,x2,y2]))
    
    redCords = np.average(np.array([cords, priorCords]), axis=0).astype(int)
    for redCord in redCords:
        print("red", redCord)
        lineImage = drawRoadLines(lineImage, np.array([redCord[0],redCord[1],redCord[2],redCord[3]]), 2, 255, 0, 0)
    
    resultingFrame = cv2.addWeighted(frame, 1, lineImage, 1, 1)
    return cords, resultingFrame



# def stabilizeCoordinates(newCords, oldCords):
#     cordAverage = []
#     for newCord, oldCord in zip(newCords, oldCords):
#         for loneCordNew, loneCordOld in zip(newCord, oldCord):
#             deviation = loneCordNew - loneCordOld
#             if abs(deviation) > max_deviation:
#                 array.append(int(loneCordOld + max_deviation * deviation / abs(deviation)))
#             else:
#                 array.append(loneCordNew)
#         cordAverage.append(np.array([x1,y1,x2,y2,x3,y3]))
#     return cordAverage

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
    cords = np.array( [ [lines[0][0],lines[0][1]], [lines[0][2],lines[0][3]], [lines[1][2],lines[1][3]], [lines[1][0], lines[1][1]] ] )
    roadImage = np.zeros_like(frame)
    cv2.fillPoly(roadImage, pts = [cords], color=(0, 255, 255))
    return roadImage

def clickEvent(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        roadCords.append((x,y))

def steeringAngle(frame, lines):
    height, width, _ = frame.shape
    
    left_x2 = lines[0][2]
    right_x2 = lines[1][2]

    mid = int(width / 2 * 1)
    x_offset = (left_x2 + right_x2) / 2 - mid
    
    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90

    return steering_angle


def stabilizeSteering(curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lines=5, max_angle_deviation_one_lane=1):
    """
    Using last steering angle to stabilize the steering angle
    This can be improved to use last N angles, etc
    if new angle is too different from current angle, only turn by max_angle_deviation degrees
    """
    if num_of_lane_lines == 2 :
        # if both lane lines detected, then we can deviate more
        max_angle_deviation = max_angle_deviation_two_lines
    else :
        # if only one lane detected, don't deviate too much
        max_angle_deviation = max_angle_deviation_one_lane
    
    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle
                                        + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle
    logging.info('Proposed angle: %s, stabilized angle: %s' % (new_steering_angle, stabilized_steering_angle))
    return stabilized_steering_angle

roadCords = []

while True:
    img = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
    img_np = np.array(img)
    videoFrame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    cv2.imshow('raw', videoFrame)
    cv2.setMouseCallback("raw", clickEvent)
    if cv2.waitKey(1) & 0xFF == ord('q') or len(roadCords) == 4:
        break

cv2.destroyAllWindows()
cords = [np.array([0,0,1,1,2,2]), np.array([0,0,1,1,2,2])]

if len(roadCords) != 4:
    roadCords = [(1025, 650), (900, 650), (300,950), (1600,950)]

print(roadCords)

previousCords = roadCords

LANEFOLLOW_VERSION = 2

# def nothing(x):
#     pass

# cv2.namedWindow("Version")
# switch = "VERSION"
# cv2.createTrackbar(switch, "Version", 1, 2, nothing)

while True:
    try:
        # LANEFOLLOW_VERSION = cv2.getTrackbarPos(switch, "Version")

        img = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
        img_np = np.array(img)
        videoFrame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        edgeDetectedFrame = edgeDetection(videoFrame, 3, 25, 75)
        reducedFOVFrame = regionOfInterest(edgeDetectedFrame, *roadCords)
    
        lines = cv2.HoughLinesP(reducedFOVFrame, 2, np.pi/180, 100, np.array([]), minLineLength=15, maxLineGap=275)

        line = averageByK(lines)
        # currentCords, resultingFrame = makeCoordinates(videoFrame, line, cords) # previous cords, current cords, cords average
        cords, resultingFrame = makeCoordinates(videoFrame, line, cords) # previous cords, current cords, cords average

        # cords = stabilizeCoordinates(currentCords, previousCords)
        # highlightedAverage = drawLines(videoFrame, [cords[0], cords[1], cords[2], cords[3]], 2, 255, 0, 0)
        # resultingFrame = cv2.addWeighted(resultingFrame, 1, highlightedAverage, 1, 1)

        highlightedRoad = highlightRoad(videoFrame, cords)
        resultingFrame = cv2.addWeighted(resultingFrame, 1, highlightedRoad, 0.1, 1)

        font = cv2.FONT_ITALIC
        
        if LANEFOLLOW_VERSION == 1:
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
        elif LANEFOLLOW_VERSION == 2:
            newSteering = steeringAngle(resultingFrame, cords)
            textSize = cv2.getTextSize(str(newSteering), font, 1, 2)[0][0]
            resultingFrame = cv2.putText(resultingFrame, str(newSteering), (int(videoFrame.shape[1]/2)-int(textSize/2), 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            pass

        stackedFrames = np.concatenate((cv2.cvtColor(np.concatenate((cv2.pyrDown(edgeDetectedFrame), cv2.pyrDown(reducedFOVFrame)), axis=1), cv2.COLOR_BGR2RGB), cv2.cvtColor(np.concatenate((cv2.pyrDown(videoFrame), cv2.pyrDown(resultingFrame)), axis=1), cv2.COLOR_BGR2RGB)), axis=0)

        cv2.imshow('result', resultingFrame)
        cv2.imshow('stackedFrames', stackedFrames)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        # previousCords = currentCords
    except:
        # previousCords = currentCords
        continue

cv2.destroyAllWindows()