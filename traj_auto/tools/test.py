# import cv2
#
# image = cv2.imread('1.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# _, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# contours, _ = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# for cont in contours:
#     # 外接圆
#     (x, y), radius = cv2.minEnclosingCircle(cont)
#     cv2.circle(image,(int(x),int(y)),int(radius), (0, 0, 255), 10)
#     cv2.imshow('test', image)
#     cv2.waitKey()


import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt

#x = np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/8)
#y = np.sin(x)

ctr =np.array( [
                (3 , 1), (2.5, 4), (0, 1),
                (-2.5, 4),
                # (-3, 0), (-2.5, -4), (0, -1), (2.5, -4), (3, -1)
                ])

x=ctr[:,0]
y=ctr[:,1]

#x=np.append(x,x[0])
#y=np.append(y,y[0])

nums_ = 2*ctr.shape[0]
tck, u = interpolate.splprep([x, y], k=3,s=0)
u=np.linspace(0,1,num=nums_,endpoint=True)
out = interpolate.splev(u,tck)

plt.figure()
plt.plot(x, y, 'ro', out[0], out[1], 'b')
plt.legend(['Points', '插值B样条', '真实数据'],loc='best')
plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
plt.title('B样条插值')
plt.show()

