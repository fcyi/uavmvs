import copy

import numpy as np
import matplotlib.pyplot as plt
import read_write_model as rwm


def UniSampling(number):
    return np.linspace(0, 360, number, endpoint=True)


def LinerSampling(start, stop, number):
    return np.linspace(start, stop, number, endpoint=True)


def get_accumList(accumStart, accumEnd, step, refineStepRatio):
    """
    增量列表计算，带拐角细化
    :param accumStart: 其实位置
    :param accumEnd: 终止位置
    :param step: 步长
    :param refineStepRatio:拐角处细化比率
    :return:
    """
    rstep = step * refineStepRatio
    accumV = []
    residualArc = 0
    # 直线部分
    while 1:
        if accumStart > accumEnd:
            break
        accumV.append(accumStart)
        accumStart += step

    # 拐角部分细化
    if accumStart > accumEnd:
        if len(accumV) > 0:
            accumStart -= step
            if (accumStart + rstep) > accumEnd:
                residualArc = rstep - (accumEnd - accumStart)
            else:
                while 1:
                    accumStart += rstep
                    if accumStart > accumEnd:
                        break
                    accumV.append(accumStart)
                residualArc = rstep - (accumEnd - accumStart + rstep)
        else:
            residualArc = accumStart - accumEnd
    return accumV, residualArc


def Pic(data):
    x, y = [], []
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(data[i][1])
    return x, y


def polygon_draw(vertexses, trajLine=()):
    for vertexs in vertexses:
        ver_ = copy.deepcopy(vertexs)
        ver_.append(ver_[0])
        verx_, very_ = Pic(ver_)
        plt.plot(verx_, very_)

    if len(trajLine) > 0:
        trajX_, trajY_ = Pic(trajLine)
        plt.plot(trajX_, trajY_)

    plt.show()


def draw_2dPos(trajList):
    x_coords = []
    y_coords = []

    for elem in trajList:
        x_coords.append(elem[0])
        y_coords.append(elem[1])

    # 绘制散点图
    plt.scatter(x_coords, y_coords)

    # 显示图形
    plt.axis('equal')
    plt.show()


def images_bin_write(binPath, posList, dstPath=""):
    data = rwm.read_images_binary(binPath)
    dataSize = len(data)
    dataIter = iter(data)
    firstKey = next(dataIter)
    maxKeyId = firstKey
    # 遍历剩余的键值对
    while True:
        try:
            key = next(dataIter)
            if maxKeyId < key:
                maxKeyId = key
        except StopIteration:
            break
    maxKeyId += 1
    posLen = len(posList)
    for i in range(posLen):
        image1 = data[firstKey]
        image1 = image1._replace(id=maxKeyId + i, qvec=np.array(posList[i][0:4]), tvec=np.array(posList[i][4:7]))
        data[maxKeyId + i] = image1

    if dstPath == "":
        dstPath = binPath
    rwm.write_images_binary(data, dstPath)


def get_step_base_rep_ratio(flyHeight, frame, focal=0, ratio=0.8):
    # 基于指定的重叠率、飞行高度、相机画幅来获取相机移动的步长
    # 飞行高度的单位为m，飞行高度以及画幅的单位为mm
    # 相机画幅与实际的感光元件的尺寸有关，一般长画幅为35mm（与旁向重叠率相关），短画幅为24mm（与航向重叠率相关）
    # 如果focal焦距为0的话，则使用默认值值26毫米
    focal = 26 if focal == 0 else focal
    # 单位换成米
    focal /= 1000
    frame /= 1000
    # 设呈现的真实距离（拍摄到的距离）为x
    x = frame * flyHeight / focal
    # 重叠部分的距离
    d = ratio * x
    # 非重叠部分的距离 （单位米）
    d = x - d
    return d





