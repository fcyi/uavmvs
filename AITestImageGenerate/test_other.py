import cv2
import numpy as np

import matplotlib.pyplot as plt


def draw_threshold():
    data = [[94, 0.13, -1],
            [80, 0.275, 1],
            [86, 0.325, 1],
            [98, 0.1837, 1],
            [105, 0.2095, -1],
            [101, 0.2376, 1],
            [102, 0.4706, 0],
            [116, 0.4569, 1],
            [116, 0.4224, 1],
            [94, 0.24, 1],
            [96, 0.385, 0],
            [82, 0.4390, 0],
            [106, 0.2358, 1],
            [117, 0.3247, 1],
            [104, 0.1538, 1],
            [87, 0.1724, 1],
            [89, 0.1011, -1],
            [67, 0.1045, -1],
            [74, 0.2703, 1],
            [103, 0.4078, 0],
            [56, 0.1964, 0],
            [80, 0.1, -1],
            [91, 0.1099, -1],
            [107, 0.2710, 1],
            [109, 0.4587, 0],
            [112, 0.375, 0],
            [94, 0.1702, -1],
            [73, 0.0959, -1],
            [81, 0.0864, 1],
            [80, 0.1, -1],
            [52, 0.1154, -1],
            [61, 0.1639, -1],
            [75, 0.2, 1],
            [89, 0.2584, 1],
            [122, 0.5328, 1],
            [83, 0.2651, 1]]
    print(len(data))

    r_x = []
    r_y = []

    g_x = []
    g_y = []

    b_x = []
    b_y = []

    for dat in data:
        if dat[2] == 0:
            b_x.append(dat[0] / 4914)
            b_y.append(dat[1])
        elif dat[2] == 1:
            g_x.append(dat[0] / 4914)
            g_y.append(dat[1])
        else:
            r_x.append(dat[0] / 4914)
            r_y.append(dat[1])

    # 绘制散点图
    # 创建Figure对象和Axes对象
    fig, ax = plt.subplots()
    ax.scatter(r_x, r_y, color='red', label='bad')
    ax.scatter(g_x, g_y, color='green', label='norm')
    ax.scatter(b_x, b_y, color='blue', label='blue')
    plt.xlabel("gm_percent")
    plt.ylabel("in_percent")

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()
    fig.savefig('threshold_save.jpg')


def compute_descriptor_distance(imgPath0, imgPath1):
    img0 = cv2.imread(imgPath0)
    img1 = cv2.imread(imgPath1)

    imgGray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    imgGray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    detect0 = cv2.xfeatures2d_SIFT.create()
    descri0 = cv2.xfeatures2d_SIFT.create()

    detect1 = cv2.xfeatures2d_SIFT.create()
    descri1 = cv2.xfeatures2d_SIFT.create()

    kpts0 = detect0.detect(imgGray0, None)
    kpts0, des0 = descri0.compute(imgGray0, kpts0)

    kpts1 = detect1.detect(imgGray1, None)
    kpts1, des1 = descri1.compute(imgGray1, kpts1)

    print("=======================================")


if __name__ == "__main__":
    draw_threshold()

    compute_descriptor_distance(imgPath0="/home/hongqingde/Downloads/AugumentImageMarker/AIMarker_rename/Pic_0000_085.jpg",
                                imgPath1="/home/hongqingde/workspace/keypointMatch/test_img_dir/Pic0002/Pic0002_0032.jpg")
