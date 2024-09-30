import copy
import math
import numpy as np
import sys
sys.path.append('tools')
import tools.utils as utl
import tools.base_tools as btls
import tools.geo_tools as gtls
from read_write_model import *


def test_line(vertexs, flyHeight, roadWidth,
              headOverlap=0.8, rratio=1.,
              pitchD=-60,
              isCoord=False,
              frameW=35, frameH=24, focal=26,
              yawType=-1,
              baseIdx=100000,
              workDir="", fileName="auto_traj"):
    viewDis = (roadWidth//2)+1
    sstep = btls.get_step_base_rep_ratio(viewDis, frameH, focal, headOverlap) if isCoord else btls.get_step_base_rep_ratio(viewDis, frameW, focal, headOverlap)

    print(sstep)
    print(vertexs)

    trajLists = utl.get_traj_by_node_sim(vertexs, sstep, rratio, yawType, isClose=True)

    trajListLen = len(trajLists)
    heightList = [flyHeight for _ in range(trajListLen)]
    regionPoint = [0, 0, 0]
    posList = utl.get_pos_by_traj_sim(trajLists, heightList, regionPoint, pitchD)

    if len(workDir) > 0:
        utl.tum_txt_write(posList, workDir, fileName)


def test_fukan_dji(vertexs,
                   flyHeight, sideOverlap=0.7,  headOverlap=0.8, rratio=1.,
                   trajType='cattle',
                   pitchD=-60,
                   yawType=-2,
                   isCoord=False,
                   frameW=35, frameH=24, focal=26,
                   workDir="", fileName="auto_traj_fukan_dji"):
    """
        大疆版本的俯瞰场景数据采集实现
        vertexs: 重建区域所对应的多边形的顶点
        flyHeight: 飞行高度
        sideOverlap: 旁向重叠率
        headOverlap: 航向重叠率
        trajType: 航线类型
        isCoord: 采集期间，相机镜头长画幅与短画幅和行进路线之间的关系
        frameW: 相机列方向画幅（单位为mm）
        frameH: 相机行方向画幅（单位为mm）
        focal: 相机焦距（单位为mm）
        yawType: 偏航角类型，
            1-相机朝向与飞机行进方向一致，
            2-相机镜头朝着行进方向逆时针90度的方向，
            0-相机镜头朝着行进方向顺时针90度的方向，
            -2-大疆的行进方向，
            -1-5向飞行，带着单个相机的飞机飞五次来模拟五个镜头，
            -3-带偏移的5向飞行
            -4-5向飞行2
    """

    vertexsNorm, verGrav = gtls.polygon_norm(vertexs)

    if isCoord:
        sideStep = btls.get_step_base_rep_ratio(flyHeight, frameW, focal, sideOverlap)
        headStep = btls.get_step_base_rep_ratio(flyHeight, frameH, focal, headOverlap)
    else:
        sideStep = btls.get_step_base_rep_ratio(flyHeight, frameH, focal, sideOverlap)
        headStep = btls.get_step_base_rep_ratio(flyHeight, frameW, focal, headOverlap)

    print(sideStep, headStep)
    print(vertexs)

    # extendD = min(-headStep, -sideStep)
    extendD = 0

    if trajType == 'cattle':
        trajNodes, _ = gtls.dji_poly_traj_v1(vertexsNorm, sideStep, 0, extendD)
    elif trajType == 'well':
        trajNodes, _ = gtls.well_poly_traj_v1(vertexsNorm, sideStep, 0, extendD)
    else:
        raise Exception

    # btls.polygon_draw([vertexsNorm], trajNodes)
    yawTypeTmp_ = yawType if trajType != 'well' else 1

    print(trajNodes)

    trajLists = utl.get_traj_by_node_sim(trajNodes, headStep, rratio, yawTypeTmp_)

    trajListLen = len(trajLists)
    heightList = [flyHeight for _ in range(trajListLen)]
    regionPoint = verGrav + [0]
    posList = utl.get_pos_by_traj_sim(trajLists, heightList, regionPoint, pitchD)

    # if len(workDir) > 0:
    #     utl.tum_txt_write(posList, workDir, fileName)
    btls.images_bin_write("/home/hongqingde/workspace_git/test/images.bin", posList,
                          "/home/hongqingde/workspace_git/test/cdata_sparse/images.bin")


def test_loop(height, vertexs,
              radiusSup, posNum, pitchDs,
              baseHeight=5,
              yawNumRadPerPos=0,
              isConnect=False, isSnake=False,
              workDir="", fileName="auto_traj_loop"):

    loopNum = len(radiusSup)
    assert len(vertexs) > 0, "重建区域边界点请给定"
    assert loopNum == len(posNum), "圈数应与采集数目的长度的数目一致"

    vertexsCp = []
    if len(vertexs[0]) >= 3:
        vertexsCp = [ver[:2] for ver in vertexs]
    elif len(vertexs[0]) == 2:
        vertexsCp = [ver for ver in vertexs]
    else:
        raise ValueError("输入的区域边界点集格式有问题")

    convexHull = gtls.graham_scan(vertexsCp)
    radiusBase, circen = gtls.create_poly_bounds(convexHull, 'min circle')
    regionPoint = [circen[0], circen[1], 0]
    radius = [radiusBase+rsup for rsup in radiusSup]

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
                                    pitchD=pitchDs[i],
                                    yawNumRadPerPos=yawNumRadPerPos)
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
                tTmp = utl.get_Tw2c_1(connectTmp[4:], Rw2c, isTranspose=True)
                for hSam in hSamList:
                    tTmp_ = tTmp.copy()
                    tTmp_[1] = tTmp_[1]-heightList[i]+hSam
                    tTmpA = utl.get_Tw2c_1(tTmp_, Rw2c)
                    posList.append(qvA.tolist() + tTmpA.tolist())

    print(len(posList))
    if len(workDir) > 0:
        utl.tum_txt_write(posList, workDir, fileName)
    # posListTest = utl.tran_to_blender(posList, isCamera=True)
    # btls.images_bin_write("/home/hongqingde/workspace_git/test/images.bin",
    #                       posList,
    #                       "/home/hongqingde/workspace_git/test/cdata_sparse/images.bin")


def test_valid_region(vertexs, flyHeight, pitchDRange,
                      isBound,
                      overlap=0.8,
                      isCoord=False,
                      frameW=35, frameH=24, focal=26,
                      workDir="", fileName="auto_traj_fukan_dji"
                      ):
    """
        区域级的测试数据采集
        vertexs: 重建区域所对应的多边形的顶点
        flyHeight: 飞行高度
        pitchDRange: 俯仰角相关信息，采集训练数据时所用的俯仰角，采集测试数据时所用的俯仰角的随机变化范围
    """
    vertexsNorm, verGrav = gtls.polygon_norm(vertexs)

    if isCoord:
        sideStep = btls.get_step_base_rep_ratio(flyHeight, frameW, focal, overlap)
        headStep = btls.get_step_base_rep_ratio(flyHeight, frameH, focal, overlap)
    else:
        sideStep = btls.get_step_base_rep_ratio(flyHeight, frameH, focal, overlap)
        headStep = btls.get_step_base_rep_ratio(flyHeight, frameW, focal, overlap)

    print(sideStep, headStep)
    print(vertexs)

    extendD = min(headStep, sideStep) if not isBound else max(-headStep, -sideStep)

    trajNodes, vertexsE = gtls.dji_poly_traj_v1(vertexsNorm, sideStep, 0, extendD, isFixLine=False)

    nodesTest = trajNodes if not isBound else vertexsE
    if isBound:
        nodesTest.append(vertexsE[0])
    trajLists = utl.get_traj_by_node_sim(nodesTest, headStep, 1., 1)

    trajListLen = len(trajLists)
    heightList = [flyHeight for _ in range(trajListLen)]
    regionPoint = verGrav + [0]
    posList = utl.get_pos_by_traj_for_region_test(trajLists, heightList, regionPoint, pitchDRange, isBound=isBound)

    # if len(workDir) > 0:
    #     utl.tum_txt_write(posList, workDir, fileName)
    btls.images_bin_write("/home/hongqingde/workspace_git/test/images.bin", posList,
                          "/home/hongqingde/workspace_git/test/cdata_sparse/images.bin")


def test_polygon(height, vertexs, radiusSup,
                 posNum, pitchDs,
                 rratio=1.,
                 baseHeight=5,
                 isConnect=False,
                 isFocus=False,
                 workDir="", fileName="auto_traj"):

    loopNum = len(radiusSup)
    assert len(vertexs) >= 3, "重建区域边界点请给定"
    assert loopNum == len(posNum), "圈数应与采集数目的长度的数目一致"

    vertexsNorm, verGrav = gtls.polygon_norm(vertexs)

    # 此处假设距离单位与模拟环境一致，为了对楼顶进行重建，无人机要高于楼顶
    baseHeight = baseHeight
    heightList = [baseHeight]

    if loopNum > 1:
        heightSeg = (height-baseHeight) / (loopNum-1)
        heightList += [baseHeight + (i+1)*heightSeg for i in range(loopNum-1)]
    else:
        heightList[0] = height

    posList = []
    regionPoint = [verGrav[0], verGrav[1], 0]

    for i in range(loopNum):
        heightS = heightList[i]
        heightE = heightList[i]
        vertexsNormd = gtls.expand_polygon_d(vertexsNorm, radiusSup[i], expand_point=False)
        posList_, stepS = utl.get_polygon_pos(vertexsNormd, heightS, heightE, posNum[i], regionPoint, rratio, pitchD=pitchDs[i], isFocus=isFocus)
        posList += posList_

        if (isConnect) and (i+1 != loopNum):
            nextSeg = stepS
            hSamTmp = heightList[i]+nextSeg
            hSamList = []
            while hSamTmp < heightList[i+1]:
                hSamList.append(hSamTmp)
                hSamTmp += nextSeg

            if len(hSamList) > 0:
                connectTmp = posList_[0]
                qvA = np.array(connectTmp[:4])
                Rw2c = qvec2rotmat(qvA)
                tTmp = utl.get_Tw2c_1(connectTmp[4:], Rw2c, isTranspose=True)
                for hSam in hSamList:
                    tTmp_ = tTmp.copy()
                    tTmp_[1] = tTmp_[1]-heightList[i]+hSam
                    tTmpA = utl.get_Tw2c_1(tTmp_, Rw2c)
                    posList.append(qvA.tolist() + tTmpA.tolist())

    print(len(posList))
    # if len(workDir) > 0:
    #     utl.tum_txt_write(posList, workDir, fileName)
    # posListTest = utl.tran_to_blender(posList, isCamera=True)
    btls.images_bin_write("/home/hongqingde/workspace_git/test/images.bin",
                          posList,
                          "/home/hongqingde/workspace_git/test/cdata_sparse/images.bin")


if __name__ == '__main__':
    # data = gtls.tu_polygon_gen(5, 30)

    # test_loop(20, data, [1], [50], [-30],
    #           # yawNumRadPerPos=(-90, -45, 45, 90),
    #           yawNumRadPerPos=0,
    #           isSnake=False,
    #           isConnect=True,
    #           workDir="/home/hongqingde/workspace_git/traj_gen")

    # utl.tum_txt_test("/home/hongqingde/devdata/trans/auto_traj_fukan.txt",
    #                  "/home/hongqingde/workspace_git/test/images.bin",
    #                  0.1,
    #                  dstPath="/home/hongqingde/workspace_git/test/cdata_sparse/images.bin")

    data = [[-127, 0], [0, 0], [0, 127], [-127, 127]]
    test_fukan_dji(data,
                   40, sideOverlap=0.7, headOverlap=0.8, rratio=0.5,
                   trajType='well',
                   pitchD=-45,
                   yawType=-4,
                   isCoord=False,
                   frameW=35, frameH=24, focal=26,
                   workDir="", fileName="auto_traj_fukan_dji")

    # test_valid_region(data, 40, pitchDRange=[-60, -44, -45])

    # test_polygon(44, data, [1, 5, 9], [10, 20, 30], [-30, -30, -30], rratio=1, isConnect=True, isFocus=False)
