import cv2

img = cv2.imread("beezos.png")

layer = img.copy()
gp = [layer]

for i in range(6):
    layer = cv2.pyrDown(layer)
    gp.append(layer)
    # cv2.imshow(str(i), layer)

layer = gp[-1]
cv2.imshow("upper level", layer)
lp = [layer]

for i in range(5, 0, -1):
    guassian_extended = cv2.pyrUp(gp[i])
    laplacian = cv2.subtract(gp[i-1], guassian_extended)
    cv2.imshow(str(i), laplacian)
    
# lr1 = cv2.pyrDown(img)
# lr2 = cv2.pyrDown(lr1)
# hr1 = cv2.pyrUp(lr2)

cv2.imshow("beeezos", img)
# cv2.imshow("lowreezos", lr1)
# cv2.imshow("lowerreezos", lr2)
# cv2.imshow("heezos", hr1)
cv2.waitKey(0)
cv2.destroyAllWindows()