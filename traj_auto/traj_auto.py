import math
import numpy as np
import sys
sys.path.append('tools')
import tools.utils as utl
from read_write_model import *


def test_calculate_circle_center(trajList, radius):
    trajT = []
    roadI = [[], []]  # 用于包含轨迹中每一段路径在轨迹中的起始位置以及该段路径所包含的采集位置数目（0-直线型，1-弧线型）
    trajLen = len(trajList)

    rIdx = 0 if trajList[0][2] == 0 else 1
    roadI[rIdx].append([0, 1])

    for i in range(1, trajLen):
        rIdx = 0 if trajList[i][2] == 0 else 1
        if trajList[i][2] == trajList[i - 1][2]:
            roadI[rIdx][-1][1] += 1
        else:
            roadI[rIdx].append([i, 1])

    cott = [0, 0]
    leng = [len(roadI[0]), len(roadI[1])]
    while (cott[0] < leng[0]) or (cott[1] < leng[1]):
        idx0 = roadI[0][cott[0]][0] if cott[0] != leng[0] else trajLen
        idx1 = roadI[1][cott[1]][0] if cott[1] != leng[1] else trajLen

        rIdx = 0 if idx0 < idx1 else 1
        iIdx = idx0 if idx0 < idx1 else idx1
        posNum = roadI[rIdx][cott[rIdx]][1]
        if rIdx == 0:
            assert posNum > 1, "直线型路线上至少需要有一个采集位置"
            for idxp in range(iIdx, iIdx + posNum, 1):
                trajT.append(trajList[idxp])
        else:
            pts = [trajList[idxp][:2] for idxp in range(iIdx, iIdx + posNum, 1)]
            cen = utl.calculate_circle_center_fast(pts, radius)
            ptsn = [[elem[0] - cen[0], elem[1] - cen[1]] for elem in pts]

            for idxp_ in range(posNum):
                idxp = idxp_+iIdx
                trajT.append(ptsn[idxp_]+trajList[idxp][2:])

        cott[rIdx] += 1

    return trajT


def test_fukan(width, height, controlPoint, flyHeight,
               stepRatio, rstepRatio,
               XRationList, YRationList,
               yawRestricts=[-1, -1, -1],
               relativeV=(0, 0, 0, 0, 0, 0),
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

    roadNR, roadNX, roadNY = utl.get_road_node_nu(width, height, xSegList, ySegList, radiuss)

    # 获取具体轨迹二维坐标
    traj = []
    posL = []

    regionPoint = controlPoint+[0]
    if len(roadNR) > 0:
        circular_at_R = [[0, 0], [1, 0], [2, 0], [3, 0]]
        trajt_ = utl.get_traj_by_node(roadNR, circular_at_R, step, rstep, radiuss[0], 0, yawRestrict=yawRestricts[0])
        traj += trajt_
        for elem in trajt_:
            elem[0] = elem[0] - width / 2
            elem[1] = elem[1] - height / 2
        heightListt_ = [flyHeight for _ in trajt_]
        # posLt_ = utl.get_pos_by_traj(trajt_, heightListt_, regionPoint, radiuss[0], relativeV)
        posLt_ = utl.get_pos_by_traj_sim(trajt_, heightListt_, regionPoint, radiuss[0], relativeV)
        posL += posLt_

    if len(roadNX) > 0:
        circular_at_X = [[0, 0], [1, 0], [3, 1], [2, 1]]
        trajt_ = utl.get_traj_by_node(roadNX, circular_at_X, step, rstep, radiuss[1], 1, yawRestrict=yawRestricts[1])
        traj += trajt_
        for elem in trajt_:
            elem[0] = elem[0] - width / 2
            elem[1] = elem[1] - height / 2
        heightListt_ = [flyHeight for _ in trajt_]
        # posLt_ = utl.get_pos_by_traj(trajt_, heightListt_, regionPoint, radiuss[1], relativeV)
        posLt_ = utl.get_pos_by_traj_sim(trajt_, heightListt_, regionPoint, radiuss[2], relativeV)
        posL += posLt_

    #
    if len(roadNY) > 0:
        circular_at_Y = [[0, 1], [3, 1], [1, 0], [2, 0]]
        trajt_ = utl.get_traj_by_node(roadNY, circular_at_Y, step, rstep, radiuss[2], 2, yawRestrict=yawRestricts[2])
        traj += trajt_
        for elem in trajt_:
            elem[0] = elem[0] - width / 2
            elem[1] = elem[1] - height / 2
        heightListt_ = [flyHeight for _ in trajt_]
        # posLt_ = utl.get_pos_by_traj(trajt_, heightListt_, regionPoint, radiuss[2], relativeV)
        posLt_ = utl.get_pos_by_traj_sim(trajt_, heightListt_, regionPoint, radiuss[2], relativeV)
        posL += posLt_

    # utl.draw_2dPos(traj)
    print(len(posL))
    # if len(workDir) > 0:
    #     utl.tum_txt_write(posL, workDir, fileName)
    utl.images_bin_write("/home/hongqingde/workspace_git/test/images.bin", posL,
                         "/home/hongqingde/workspace_git/test/cdata_sparse/images.bin")


def test_loop(height, radius, regionPoint, loopNum, posNum,
              baseHeight=5,
              yawNumRadPerPos=0,
              isConnect=False,
              isSnake=False,
              relativeV=(0, 0, 0, 0, 0, 0),
              workDir="", fileName="auto_traj_loop"):

    assert (len(radius) == len(posNum)) and (len(radius) == loopNum), "圈数应与radius的长度以及posNum的长度一致"
    # controlPoint = [-0.009859, 4.297023, 0.11011]
    # 此处假设距离单位与模拟环境一致，为了对楼顶进行重建，无人机要高于楼顶
    baseHeight = baseHeight
    heightList = [baseHeight]
    if (loopNum > 2) and isSnake:
        heightSeg = (height - baseHeight) / (loopNum - 2)
        heightList.append(baseHeight)
        heightList += [baseHeight + (i + 1) * heightSeg for i in range(loopNum - 2)]
    else:
        if loopNum > 1:
            heightSeg = (height-baseHeight) / (loopNum-1)
            heightList += [baseHeight + (i+1)*heightSeg for i in range(loopNum-1)]
        else:
            if not isSnake:
                heightList[0] = height

    if isSnake:
        heightList.append(height)

    posList = []

    for i in range(loopNum):
        heightS = heightList[i]
        heightE = heightList[i+1] if isSnake else heightList[i]
        posList_ = utl.get_loop_pos(radius[i], heightS, heightE, posNum[i], regionPoint,
                                    yawNumRadPerPos=yawNumRadPerPos,
                                    relativeV=relativeV)
        posList += posList_

        if (isConnect) and (not isSnake) and (i+1 != loopNum):
            nextSeg = 2*np.pi*radius[i] / posNum[i]
            # samNum = (heightList[i+1] - heightList[i])/nextSeg
            hSamTmp = heightList[i]+nextSeg
            hSamList = []
            while hSamTmp < heightList[i+1]:
                hSamList.append(hSamTmp)
                hSamTmp += nextSeg

            if len(hSamList) > 0:
                connectTmp = posList_[0]
                qvA = np.array(connectTmp[:4])
                Rw2c = qvec2rotmat(qvA)
                tTmp = utl.get_Tc2w_1(connectTmp[4:], Rw2c)
                for hSam in hSamList:
                    tTmp_ = tTmp.copy()
                    tTmp_[1] = tTmp_[1]-heightList[i]+hSam
                    tTmpA = utl.get_Tw2c_1(tTmp_, Rw2c)
                    posList.append(qvA.tolist() + tTmpA.tolist())

    print(len(posList))
    # if len(workDir) > 0:
    #     # utl.tum_txt_write(posList, workDir, fileName)
    #     utl.tum_txt_write_colmap(posList, workDir, fileName)
    print(posList)
    posListTest = utl.tran_to_blender(posList, isCamera=True)
    print(posList)
    utl.images_bin_write("/home/hongqingde/workspace_git/test/images.bin",
                         posListTest,
                         "/home/hongqingde/workspace_git/test/cdata_sparse/images.bin")


if __name__ == '__main__':
    # height = 20
    # width = 40
    # controlPoint = [0, 0]
    # flyHeight = 10
    # stepRatio = 0.1
    # rstepRatio = 0.25
    # xRationList = [1/8, 3/8, 3/8, 1/8]
    # yRationList = [1/8, 3/8, 3/8, 1/8]
    # # xRationList = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    # # yRationList = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
    # test_fukan(width, height, controlPoint, flyHeight, stepRatio, rstepRatio, xRationList, yRationList,
    #            yawRestricts=[-1, -1, -1],
    #            relativeV=(0, 0, 0, 0, 0, 0),
    #            workDir="/home/hongqingde/devdata/trans")
    #
    test_loop(0, [3], [0, 0, 0], 1, [5], 0,
              # yawNumRadPerPos=(-90, -45, 45, 90),
              yawNumRadPerPos=0,
              isSnake=False,
              isConnect=False,
              relativeV=(0, 0, 0, 0, 0, 0),
              workDir="/home/hongqingde/workspace_git/traj_gen")
    #
    # utl.tum_txt_test("/home/hongqingde/devdata/trans/auto_traj_fukan.txt",
    #                  "/home/hongqingde/workspace_git/test/images.bin",
    #                  0.1,
    #                  dstPath="/home/hongqingde/workspace_git/test/cdata_sparse/images.bin")
