import cv2

image = cv2.imread('1.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for cont in contours:
    # 外接圆
    (x, y), radius = cv2.minEnclosingCircle(cont)
    cv2.circle(image,(int(x),int(y)),int(radius), (0, 0, 255), 10)
    cv2.imshow('test', image)
    cv2.waitKey()
