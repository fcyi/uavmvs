import copy
import math
import os.path

import sys 

sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import read_write_model as rwm

import geo_tools as gtls

import collections

ExtrinsicParams = collections.namedtuple(
    "Image", ["id", "params", "camId", "name"]  # params = [qw, qx, qy, qz, tx, ty, tz]
)


# 在有序且无重复元素的列表s_中查找p_出现或者第一个大于p_的位置，后续可以改为进一步2分加速
def get_equal_based_sortedSquences(p_, s_, iS_, iE_):
    isFound_ = False
    iST_ = iS_
    for idx_ in range(iS_, iE_):
        if s_[idx_] >= p_:
            iST_ = idx_
            if s_[idx_] == p_:
                isFound_ = True
            break

    return isFound_, iST_


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


def get_step_base_rep_ratio(flyHeight_, frame_, fovD_, pitchD_, focal_=0, ratio_=0.8, HLimit=100, pp_=1.):
    """
    * height_    无人机高度，单位米(m)
    * frame_    画幅（航向--短画幅、旁向--长画幅）传感器的长宽
    * focal_     焦距
    * ratio_     重叠率
    * HLimit     无人机最大飞行高度，根据要求的地物最小分辨率、无人机的最大相对地面的飞行高度亦或是无人机所携带相机的最大有效观测范围决定
    * return double   非重叠部分的真实距离
    """
    # 基于指定的重叠率、飞行高度、相机画幅来获取相机移动的步长
    # 飞行高度的单位为m，飞行高度以及画幅的单位为mm
    # 相机画幅与实际的感光元件的尺寸有关，一般长画幅为35mm（与旁向重叠率相关），短画幅为24mm（与航向重叠率相关）
    # 如果focal焦距为0的话，则使用默认值值26毫米
    focal_ = 26 if focal_ == 0 else focal_
    # 单位换成米
    focal_ /= 1000
    frame_ /= 1000
    # 设相机能拍摄、呈现的真实距离（拍摄到的距离）为x，相机能够拍摄到的真实距离需要根据相机的俯仰角以及视场角进行调整修正
    assert (pitchD_ <= 0) and (fovD_ > 0), "在俯瞰视角获取场合，相机朝上没有意义。当然相机的视场角也可能设置错误"

    pitchDTmp_ = -pitchD_  # 后续计算都是使用俯仰角的角度值的绝对值
    if (2*pitchDTmp_) > fovD_:
        recorrectRatio_ = math.sin(math.pi - (fovD_ / 2.)) / math.sin((fovD_ / 2.) + pitchDTmp_)
        x_ = (frame_ * flyHeight_ / focal_) * recorrectRatio_
    elif (fovD_ >= (2*pitchDTmp_)) and (pitchDTmp_ > 0):
        x_ = flyHeight_ * (math.tan(math.pi/2. - pitchDTmp_) - math.tan(math.pi/2. - fovD_/2. - pitchDTmp_))
    else:
        x_ = HLimit - flyHeight_*math.tan(math.pi/2. - fovD_/2.)
    # 重叠率的计算，pp，若实际重叠率ration的步进长度为基准重叠率ratio的步进长度的pp倍，该如何计算ration
    ration_ = 1. - pp_*(1. - ratio_)
    # 重叠部分的距离
    d_ = ration_ * x_
    # 非重叠部分的距离 （单位米）
    d_ = x_ - d_
    return d_


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


def get_intrinsic_params_for_airsim(resoluation_, fov_):
    """
    在airsim中水平方向的fov与垂直方向的fov是一致的，在设置文件settings.json中调整的fov其实也就是这个参数
    resoluation_: 图像的宽度和高度（单位为像素）
    fov_: 视场角（单位为度）
    return: [fx_, fy_, cx_, cy_, width_, height_], 相机中心，水平和垂直方向的焦距（中心、焦距的单位由resoluation决定）
    """
    width_, height_ = resoluation_[:]
    cx_ = width_ / 2.
    cy_ = height_ / 2.

    fx_ = width_ / 2. / math.tan(math.radians(fov_ / 2.))
    fy_ = width_ / 2. / math.tan(math.radians(fov_ / 2.))  # 注意，此处不是height_ / 2. / math.tan(math.radians(fov_ / 2.))

    return [fx_, fy_, cx_, cy_, width_, height_]


def get_extrinsic_params_from_airsim(inputPath_, isInv=False):
    assert os.path.exists(inputPath_)
    extrinsicParams_ = dict()

    with open(inputPath_, 'r') as fid_:
        fcot_ = 1
        camId_ = 1
        for line_ in fid_:
            if len(line_) <= 0 or line_[0] == '#':
                continue
            line_ = line_.strip()
            elements_ = line_.split()
            if isInv:
                paramsTmp_ = np.array(tuple(map(float, [elements_[7], *elements_[4:7], *elements_[1:4]])))
                paramsTmp_ = paramsTmp_.tolist()
                RTmp_ = rwm.qvec2rotmat(paramsTmp_[:4])
                TTmp_ = np.eye(4, 4)
                TTmp_[:3, :3] = RTmp_
                TTmp_[:3, 3] = np.array(paramsTmp_[4:])
                TInvTmp_ = np.linalg.inv(TTmp_)
                RInvTmp_ = TInvTmp_[:3, :3]
                tInvTmp_ = TInvTmp_[:3, 3]
                qvecInv_ = rwm.rotmat2qvec(RInvTmp_)
                params_ = qvecInv_.tolist() + tInvTmp_.tolist()
            else:
                params_ = np.array(tuple(map(float, [elements_[7], *elements_[4:7], *elements_[1:4]])))
            fileName = elements_[0] + '.png'
            extrinsicParams_[fcot_] = ExtrinsicParams(id=fcot_, params=params_, camId=camId_, name=fileName)
            fcot_ += 1
    return extrinsicParams_


def write_intrinsic_params(intrinsicMatrixs_, outputDir_, fileName_="cameras.txt"):
    intrinsicMatrixsNum_ = len(intrinsicMatrixs_)

    if not os.path.exists(outputDir_):
        os.makedirs(outputDir_, exist_ok=True)

    outputPath_ = os.path.join(outputDir_, fileName_)

    cameras_ = dict()
    for inMIdx_, inM_ in enumerate(intrinsicMatrixs_):
        cameras_[inMIdx_+1] = rwm.Camera(id=inMIdx_+1, model='PINHOLE', width=inM_[4], height=inM_[5], params=inM_[:4])

    rwm.write_cameras_text(cameras_, outputPath_)


def write_extrinsic_params(extrinsicParams_, outputDir_, fileName_="images.txt"):
    extrinsicParamsNum_ = len(extrinsicParams_)

    if not os.path.exists(outputDir_):
        os.makedirs(outputDir_, exist_ok=True)

    outputPath_ = os.path.join(outputDir_, fileName_)

    HEADER_ = (
            "# Image list with two lines of data per image:\n"
            + "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
            + "# POINTS2D[] as (X, Y, POINT3D_ID)\n"
            + "# Number of images: {}, mean observations per image: 2\n".format(extrinsicParamsNum_)
    )

    with open(outputPath_, "w") as fid_:
        fid_.write(HEADER_)
        for _, exParam in extrinsicParams_.items():
            to_write = [exParam.id, *exParam.params, exParam.camId, exParam.name]
            line_ = " ".join([str(elem_) for elem_ in to_write])
            fid_.write(line_ + "\n\n")


if __name__ == '__main__':

    intrinsicParams = get_intrinsic_params_for_airsim([1920, 1080], 90)
    extrinsicParams = get_extrinsic_params_from_airsim(
        "/home/hongqingde/devdata/trans/urban/block2_sim/train/gt_as_col_pose.txt",
        True
    )

    outputDir = "/home/hongqingde/devdata/workspace_gitmp/input/testData/created/sparse"

    write_intrinsic_params(intrinsicMatrixs_=[intrinsicParams], outputDir_=outputDir)
    write_extrinsic_params(extrinsicParams_=extrinsicParams, outputDir_=outputDir)

    pass

