import math
import numpy as np
import sys
sys.path.append('../tools')
import utils_aban as utla
import tools.utils as utls
import tools.base_tools as btls
import tools.geo_tools as gtls
from read_write_model import *


def test_fukan(width, height, controlPoint, flyHeight,
               stepRatio, rstepRatio,
               XRationList, YRationList,
               yawRestricts=[-1, -1, -1],
               workDir="", fileName="auto_traj_fukan"):
    """
        使用方式，首先根据成像区域中主要内容分布，确定好井字路线的数目，之后，再根据数据量设置采集间隔。
        需要注意的是，由于路线时圆角类型，所以若运行期间报错，还需要调整圆角对应的半径，或者是采集间隔，
        这是为了保证圆角上包含至少3个采集点。
        确保行方向和列方向的井字路线数目都为奇数，且都大于1，这是为了方便井字路径可以回到原点
    :param width:
    :param height:
    :param step:
    :param rstep:
    :param XRationList: 横向井字路线（与宽平行）的间隔相对于高的占比
    :param YRationList: 纵向井字路线（与高平行）的间隔相对于宽的占比
    :return:
    """

    # 圆角矩形的参数
    # bian_ = 8
    # width = 2*bian_
    # height = bian_
    # center = [0, 0]
    # flyHeight = 1.5
    # step = bian_ / 10
    # rstep = step / 4
    # step = 0.2
    # rstep = 0.1

    # 确保行方向和列方向的井字路线数目都为奇数，且都大于1
    # radius = bian_ / 4
    # mradius = radius / 4  # 井字路线中圆角半径

    # xBias = radius
    # yBias = radius
    #
    # xJNum = 5
    # yJNum = 3

    # 获取轨迹节点
    bian_ = width if width <= height else height

    step = bian_*stepRatio
    rstep = step*rstepRatio

    xSegList = [height * xelem for xelem in XRationList]
    ySegList = [width * yelem for yelem in YRationList]
    xSegMin, ySegMin = height, width
    for xSeg in xSegList:
        if xSegMin > xSeg:
            xSegMin = xSeg
    for ySeg in ySegList:
        if ySegMin > ySeg:
            ySegMin = ySeg
    radiuss = [bian_/4, xSegMin/4, ySegMin/4]

    roadNR, roadNX, roadNY = utla.get_road_node_nu(width, height, xSegList, ySegList, radiuss)

    # 获取具体轨迹二维坐标
    traj = []
    posL = []

    regionPoint = controlPoint+[0]
    if len(roadNR) > 0:
        circular_at_R = [[0, 0], [1, 0], [2, 0], [3, 0]]
        trajt_ = utla.get_traj_by_node(roadNR, circular_at_R, step, rstep, radiuss[0], 0, yawRestrict=yawRestricts[0])
        traj += trajt_
        for elem in trajt_:
            elem[0] = elem[0] - width / 2
            elem[1] = elem[1] - height / 2
        heightListt_ = [flyHeight for _ in trajt_]
        # posLt_ = utl.get_pos_by_traj(trajt_, heightListt_, regionPoint, radiuss[0])
        posLt_ = utls.get_pos_by_traj_sim(trajt_, heightListt_, regionPoint)
        posL += posLt_

    if len(roadNX) > 0:
        circular_at_X = [[0, 0], [1, 0], [3, 1], [2, 1]]
        trajt_ = utla.get_traj_by_node(roadNX, circular_at_X, step, rstep, radiuss[1], 1, yawRestrict=yawRestricts[1])
        traj += trajt_
        for elem in trajt_:
            elem[0] = elem[0] - width / 2
            elem[1] = elem[1] - height / 2
        heightListt_ = [flyHeight for _ in trajt_]
        # posLt_ = utl.get_pos_by_traj(trajt_, heightListt_, regionPoint, radiuss[1])
        posLt_ = utls.get_pos_by_traj_sim(trajt_, heightListt_, regionPoint)
        posL += posLt_

    #
    if len(roadNY) > 0:
        circular_at_Y = [[0, 1], [3, 1], [1, 0], [2, 0]]
        trajt_ = utla.get_traj_by_node(roadNY, circular_at_Y, step, rstep, radiuss[2], 2, yawRestrict=yawRestricts[2])
        traj += trajt_
        for elem in trajt_:
            elem[0] = elem[0] - width / 2
            elem[1] = elem[1] - height / 2
        heightListt_ = [flyHeight for _ in trajt_]
        # posLt_ = utl.get_pos_by_traj(trajt_, heightListt_, regionPoint, radiuss[2])
        posLt_ = utls.get_pos_by_traj_sim(trajt_, heightListt_, regionPoint)
        posL += posLt_

    # utl.draw_2dPos(traj)
    print(len(posL))
    # if len(workDir) > 0:
    #     utl.tum_txt_write(posL, workDir, fileName)
    btls.images_bin_write("/home/hongqingde/workspace_git/test/images.bin", posL,
                          "/home/hongqingde/workspace_git/test/cdata_sparse/images.bin")


if __name__ == '__main__':
    height = 20
    width = 40
    controlPoint = [0, 0]
    flyHeight = 10
    stepRatio = 0.1
    rstepRatio = 0.25
    xRationList = [1/8, 3/8, 3/8, 1/8]
    yRationList = [1/8, 3/8, 3/8, 1/8]
    # xRationList = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    # yRationList = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    test_fukan(width, height, controlPoint, flyHeight, stepRatio, rstepRatio, xRationList, yRationList,
               yawRestricts=[-1, -1, -1],
               workDir="/home/hongqingde/devdata/trans")

    # utl.tum_txt_test("/home/hongqingde/devdata/trans/auto_traj_fukan.txt",
    #                  "/home/hongqingde/workspace_git/test/images.bin",
    #                  0.1,
    #                  dstPath="/home/hongqingde/workspace_git/test/cdata_sparse/images.bin")

