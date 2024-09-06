import copy
import math

import numpy as np
import matplotlib.pyplot as plt
import read_write_model as rwm

import geo_tools as gtls


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


def get_step_base_rep_ratio(flyHeight, frame, focal=0, ratio=0.8, pp=1.):
    """
    * height    无人机高度，单位米(m)
    * frame     画幅（航向--短画幅、旁向--长画幅）传感器的长宽
    * focal     焦距
    * ratio     重叠率
    * return double   非重叠部分的真实距离
    """
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
    # 重叠率的计算，pp，若实际重叠率ration的步进长度为基准重叠率ratio的步进长度的pp倍，该如何计算ration
    ration = 1. - pp*(1. - ratio)
    # 重叠部分的距离
    d = ration * x
    # 非重叠部分的距离 （单位米）
    d = x - d
    return d


def calcFlightSpeed(courseInterval, timeInterval):
    """
     计算飞行速度
     courseInterval    航向间距，在具体拍摄时设为图像采集位置之间的间隔距离
     timeInterval      时间间隔，在具体拍摄时设为图像采集的间隔时间
     return double           飞行拍摄速度
    """
    if timeInterval == 0:
        return 0
    return courseInterval / timeInterval


def calculate_pixel_area(fov, resolution, distance):
    """
    根据输入的水平、垂直方向的视场角、图像分辨率，以及拍摄距离，推断出每个像素所对应的成像区域在水平方向和垂直方向上的距离
    """
    fov_w, fov_h = fov[:]
    resolution_w, resolution_h = resolution[:]

    # 计算水平和垂直方向的IFOV
    ifov_h = fov_h / resolution_h
    ifov_w = fov_w / resolution_w

    # 计算每个像素在目标物体上的宽度和高度，单位与distance一致
    pixel_width = 2 * distance * math.tan(math.radians(ifov_w) / 2)
    pixel_height = 2 * distance * math.tan(math.radians(ifov_h) / 2)

    # # 计算每个像素的实际面积
    # pixel_area = pixel_width * pixel_height

    return pixel_width, pixel_height


def calculate_crop_area(fov, resolution, flyHeight,
                        regionVertexs):
    """
    成像区域外接矩形的重心与图像中心一致的情况下，从图像中将成像区域的外接矩形区域给截取出来，返回外接矩形在拍摄图像中的左上角点和右下角点
    """
    recPts, _ = gtls.create_poly_bounds(regionVertexs, type='sim rectangle')
    recW = recPts[1][0]-recPts[3][0]+1
    recH = recPts[1][1]-recPts[3][1]+1
    pixRangeW, pixRangeH = calculate_pixel_area(fov, resolution, flyHeight)

    pixNumHalfW = math.ceil(recW / pixRangeW) // 2
    pixNumHalfH = math.ceil(recH / pixRangeH) // 2

    imgCen = [resolution[0]//2, resolution[1]//2]

    leftUpPt = [imgCen[0]-pixNumHalfW, imgCen[1]-pixNumHalfH]
    rightBottomPt = [imgCen[0]+pixNumHalfW, imgCen[1]+pixNumHalfH]

    xBound, yBound = [leftUpPt[0], rightBottomPt[0]], [leftUpPt[1], rightBottomPt[1]]

    return xBound, yBound


def calculate_flyHeight(fov, resolution, regionVertexs):
    """
    成像区域外接矩形的重心与图像中心一致的情况下，保证成像区域完整地出现在图像中的飞行高度
    """
    recPts, _ = gtls.create_poly_bounds(regionVertexs, type='sim rectangle')
    recW = recPts[1][0] - recPts[3][0]+1
    recH = recPts[1][1] - recPts[3][1]+1
    resolution_w, resolution_h = resolution[:]
    pixRangeW, pixRangeH = recW / resolution_w, recH / resolution_h

    fov_w, fov_h = fov[:]
    ifov_h = fov_h / resolution_h
    ifov_w = fov_w / resolution_w

    flyHeightH = 2 * math.tan(math.radians(ifov_h) / 2) / pixRangeH
    flyHeightW = 2 * math.tan(math.radians(ifov_w) / 2) / pixRangeW

    return 10.0 * max(flyHeightH, flyHeightW)


def calculate_intrinsic_params(resoluation_, fov_):
    """
    resoluation_: 宽度和高度分辨率（或画幅宽度和高度）
    fov_: 视场角（单位为度，此处假设水平视场角和垂直视场角一致）
    return: [cx_, cy_, fx_, fy_], 相机中心，水平和垂直方向的焦距（中心、焦距的单位由resoluation决定）
    """
    width_, height_ = resoluation_[:]
    cx_ = width_ / 2.
    cy_ = height_ / 2.




    pass
