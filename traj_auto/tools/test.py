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


import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像
image = cv2.imread('/home/hongqingde/Downloads/DJI_0110.JPG')  # 替换为你的图像路径
# 将 BGR 转换为 RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. 计算直方图
# 使用 cv2.calcHist() 计算每个通道的直方图
channels = ('r', 'g', 'b')
hist_data = {}
for i, color in enumerate(channels):
    hist_data[color] = cv2.calcHist([image], [i], None, [256], [0, 256])

# 3. 绘制直方图
plt.figure(figsize=(10, 5))
for color, hist in hist_data.items():
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.title('Color Histogram')
plt.xlabel('Pixel Values')
plt.ylabel('Frequency')
plt.legend(channels)
plt.show()
