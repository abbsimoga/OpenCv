import cv2

imgur = cv2.imread("lol.jpg", -1) 
cv2.imshow("beeezos", imgur)
kay = cv2.waitKey(0) & 0xFF

if kay == 27:
    cv2.destroyAllWindows()
elif kay == ord("s"):
    cv2.imwrite("beezos.png", imgur)
    cv2.destroyAllWindows()