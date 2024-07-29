import math
import sys

import numpy as np

sys.path.append("../")
import tools.geo_tools as gtls
import tools.utils as utls
import tools.pos_parse as posPar


def get_road_node_sub(swidth, sheight, mradius, seg, idx, type=0, dire=0):
    # dire = 0：顺时针
    # type == 0: 从左到右、1：从右到左、2：右边从上到下、3：左边从上到下
    # dire == 1：逆时针
    # type == 0: 从右到左、1：从右到左、2：左边从上到下、3：右边从上到下

    assert seg > 2*mradius, "间隔过小"
    subPoints = []
    if type == 0 or type == 1:
        subPoints.append([mradius, sheight-idx*seg])
        subPoints.append([swidth-mradius, sheight-idx*seg])
        if (dire == 0 and type == 1) or (dire == 1 and type == 0):
            tmp = subPoints[0]
            subPoints[0] = subPoints[1]
            subPoints[1] = tmp
    else:
        subPoints.append([0, sheight-mradius-idx*seg])
        subPoints.append([0, sheight-(idx+1)*seg+mradius])
        if (dire == 0 and type == 2) or (dire == 1 and type == 3):
            subPoints[0][0] += swidth
            subPoints[1][0] += swidth
    return subPoints


def get_road_node(width, height, xBias, yBias, xJNum, yJNum, radius, mRadius):
    regionW = width - 2*xBias
    regionH = height - 2*yBias

    xJSeg = regionW / (xJNum-1)
    yJSeg = regionH / (yJNum-1)

    # 外围
    roadNodesR = [
        [radius, height], [width-radius, height],
        [width, height-radius], [width, radius],
        [width - radius, 0], [radius, 0],
        [0, radius], [0, height - radius]
    ]

    # 井字横路
    roadNodesX = []
    for i in range(yJNum):
        type = i % 2
        roadNodesX += get_road_node_sub(regionW, regionH, mRadius, yJSeg, i, type, 0)
        if i < yJNum-1:
            roadNodesX += get_road_node_sub(regionW, regionH, mRadius, yJSeg, i, type+2, 0)

    nodeNumX = len(roadNodesX)
    roadNodesX[0][0] = 0
    roadNodesX[nodeNumX-1][0] = regionW
    for i in range(nodeNumX):
        roadNodesX[i][0] += xBias
        roadNodesX[i][1] += yBias

    # 井字纵路
    roadNodesY = []
    for i in range(xJNum):
        type = 0  # 奇数条路径
        if i % 2 == 1:
            type = 1  # 偶数条路径
        roadNodesY += get_road_node_sub(regionH, regionW, mRadius, xJSeg, i, type, 1)
        if i < xJNum - 1:
            roadNodesY += get_road_node_sub(regionH, regionW, mRadius, xJSeg, i, type + 2, 1)

    nodeNumY = len(roadNodesY)
    roadNodesY[0][0] = regionH
    roadNodesY[nodeNumY - 1][0] = 0

    for i in range(nodeNumY):
        roadNodesY[i][0] -= regionH / 2
        roadNodesY[i][1] -= regionW / 2

    roadNodesY = gtls.rotate_coordinates(roadNodesY, 0, 0, 90)
    for i in range(nodeNumY):
        roadNodesY[i][0] += width / 2
        roadNodesY[i][1] += height / 2

    return roadNodesR, roadNodesX, roadNodesY


def get_road_node_sub_nu(swidth, sheight, mradius, segList, idx, type=0, dire=0):
    # dire = 0：顺时针
    # type == 0: 从左到右、1：从右到左、2：右边从上到下、3：左边从上到下
    # dire == 1：逆时针
    # type == 0: 从右到左、1：从右到左、2：左边从上到下、3：右边从上到下
    # assert seg > 2*mradius, "间隔过小"
    ACCumSeg = 0
    for i in range(1, idx+1):
        ACCumSeg += segList[i]
    subPoints = []
    if type == 0 or type == 1:
        subPoints.append([mradius, sheight-ACCumSeg])
        subPoints.append([swidth-mradius, sheight-ACCumSeg])
        if (dire == 0 and type == 1) or (dire == 1 and type == 0):
            tmp = subPoints[0]
            subPoints[0] = subPoints[1]
            subPoints[1] = tmp
    else:
        subPoints.append([0, sheight-mradius-ACCumSeg])
        subPoints.append([0, sheight-ACCumSeg-segList[idx+1]+mradius])
        if (dire == 0 and type == 2) or (dire == 1 and type == 3):
            subPoints[0][0] += swidth
            subPoints[1][0] += swidth
    return subPoints


def get_road_node_nu(width, height, xSegList, ySegList, radiuss):
    xJNum, yJNum = len(xSegList)-1, len(ySegList)-1
    xBias = ySegList[-1]
    yBias = xSegList[0]
    regionW = width-ySegList[0]-ySegList[-1]
    regionH = height-xSegList[0]-xSegList[-1]

    radius, xRadius, yRadius = radiuss[:]
    # 外围
    roadNodesR = [
        [radius, height], [width-radius, height],
        [width, height-radius], [width, radius],
        [width - radius, 0], [radius, 0],
        [0, radius], [0, height - radius]
    ]

    # 井字横路
    roadNodesX = []
    for i in range(xJNum):
        type = i % 2
        roadNodesX += get_road_node_sub_nu(regionW, regionH, xRadius, xSegList, i, type, 0)
        if i < xJNum-1:
            roadNodesX += get_road_node_sub_nu(regionW, regionH, xRadius, xSegList, i, type+2, 0)
    # print(xJNum, len(roadNodesX))

    nodeNumX = len(roadNodesX)
    if nodeNumX > 0:
        roadNodesX[0][0] = 0
        roadNodesX[nodeNumX-1][0] = regionW
    for i in range(nodeNumX):
        roadNodesX[i][0] += xBias
        roadNodesX[i][1] += yBias

    # 井字纵路
    roadNodesY = []
    for i in range(yJNum):
        type = 0  # 奇数条路径
        if i % 2 == 1:
            type = 1  # 偶数条路径
        roadNodesY += get_road_node_sub_nu(regionH, regionW, yRadius, ySegList, i, type, 1)
        if i < yJNum - 1:
            roadNodesY += get_road_node_sub_nu(regionH, regionW, yRadius, ySegList, i, type + 2, 1)

    nodeNumY = len(roadNodesY)
    if nodeNumY > 0:
        roadNodesY[0][0] = regionH
        roadNodesY[nodeNumY - 1][0] = 0
    for i in range(nodeNumY):
        roadNodesY[i][0] -= regionH / 2
        roadNodesY[i][1] -= regionW / 2
    if nodeNumY > 0:
        roadNodesY = gtls.rotate_coordinates(roadNodesY, 0, 0, 90)

    for i in range(nodeNumY):
        roadNodesY[i][0] += width / 2
        roadNodesY[i][1] += height / 2

    return roadNodesR, roadNodesX, roadNodesY


def noncircular_traj_tmp(ps, pe, res, step, rstep):
    traj = []
    residualArc = 0
    type = 0 if ps[0] == pe[0] else 1  # 纵向还是横向
    dire = 0 if (ps[0] < pe[0]) or (ps[1] < pe[1]) else 1  # 前进还是后退

    accumV = []
    accumStart = res
    accumEnd = abs(pe[1]-ps[1]) if type == 0 else abs(pe[0]-ps[0])

    # 直线部分
    while 1:
        if accumStart > accumEnd:
            break
        accumV.append(accumStart)
        accumStart += step

    # 直线拐角部分细化
    if accumStart > accumEnd:
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

    accumLen = len(accumV)
    if dire == 1:
        for i in range(accumLen):
            accumV[i] = -accumV[i]

    if type == 0:
        for i in range(accumLen):
            traj.append([ps[0], ps[1]+accumV[i]])
    else:
        for i in range(accumLen):
            traj.append([ps[0]+accumV[i], ps[1]])

    return traj, residualArc


def circular_traj(startP, startArc, radius, step, rstep, type=0, dire=0):
    # type表示圆四等分后的四个象限
    # dire表示行进方向是顺时针还是逆时针
    # 拐角部分
    # type == 0: 第一象限、1：第四象限、2：第三象限、3：第二象限
    # dire == 0: 顺时针、1：逆时针
    startX, startY = startP[0:2]
    traj = []
    residualLen = 0
    arcLim = math.pi * radius / 2
    accumArc = startArc
    while 1:
        alpha = gtls.get_angle(accumArc, radius)  if accumArc > 1e-4 else 0 # 弧度制
        if accumArc > arcLim:
            break
        tmpx = radius * math.cos(alpha)
        tmpy = radius * math.sin(alpha)
        traj.append([tmpx, tmpy])
        accumArc += rstep

    if accumArc > arcLim:
        residualLen = step - (arcLim - accumArc + rstep)

    trajLen = len(traj)
    if dire == 0:
        if type == 0:
            for i in range(trajLen):
                tmpx = traj[i][0]
                traj[i][0] = startX + traj[i][1]
                traj[i][1] = startY + tmpx - radius
        elif type == 1:
            for i in range(trajLen):
                traj[i][0] = startX + traj[i][0] - radius
                traj[i][1] = startY - traj[i][1]
        elif type == 2:
            for i in range(trajLen):
                tmpx = traj[i][0]
                traj[i][0] = startX - traj[i][1]
                traj[i][1] = startY + (radius - tmpx)
        elif type == 3:
            for i in range(trajLen):
                traj[i][0] = startX + (radius-traj[i][0])
                traj[i][1] = startY + traj[i][1]
    else:
        if type == 0:
            for i in range(trajLen):
                traj[i][0] = startX - (radius-traj[i][0])
                traj[i][1] = startY + traj[i][1]
        elif type == 1:
            for i in range(trajLen):
                tmpx = traj[i][0]
                traj[i][0] = startX + traj[i][1]
                traj[i][1] = startY + (radius-tmpx)
        elif type == 2:
            for i in range(trajLen):
                traj[i][0] = startX + (radius-traj[i][0])
                traj[i][1] = startY - traj[i][1]
        elif type == 3:
            for i in range(trajLen):
                tmpx = traj[i][0]
                traj[i][0] = startX - traj[i][1]
                traj[i][1] = startY - (radius-tmpx)

    return traj, residualLen


def get_traj_by_node(nodeList, circular_at, step, rstep, radius, type=0, yawRestrict=-1):
    # type == 0: roadRound、1：roadX、2：roadY
    traj_ = []
    rest_ = 0
    cot = 0
    assert  yawRestrict in (-1, 0, 1), '偏航角定义存在问题'
    if yawRestrict == -1:
        yawd = 0 if type == 0 else 1  # 相机镜头朝着行进方向顺时针90度的方向，亦或是沿着行进方向
    else:
        yawd = yawRestrict
    lineds = []  # 每一段直线或弧线所包含的采集位置数目
    nodeLen = len(nodeList)
    for i in range(0, nodeLen, 2):
        trajt_, rest_ = noncircular_traj_tmp(nodeList[i], nodeList[i + 1], rest_, step, rstep)
        traj_ += trajt_
        lineds.append(len(trajt_))
        if (type == 0) or (i < (nodeLen - 2)):
            trajtc_, rest_ = circular_traj(nodeList[i + 1], rest_, radius,
                                           step, rstep,
                                           circular_at[cot][0], circular_at[cot][1])
            traj_ += trajtc_
            cot += 1
            cot = cot % 4
            lineds.append(len(trajtc_))

    # 获取位置的路线信息、镜头方向信息、路线行进是否为逆时针（主要是针对于弧型路线，求相机朝向时很必要）
    cot = 0
    # 0-行进方式为顺时针，1-行进方向为逆时针
    # suniL[0]/suniL[1] - 外围的直线和弧线
    # suniL[2]/suniL[3] - 横向井路的直线和弧线
    # suniL[4]/suniL[5] - 纵向井路的直线和弧线
    suniL = [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 1, 1],
             [1, 1, 1, 1],
             [1, 1, 0, 0]]
    # sunit = 0 if type != 2 else 1
    cotL = 0
    cotA = 0
    for i in range(len(lineds)):  # 对每一条路线进行遍历
        posNum = lineds[i]
        posD = i % 2  # 1-采集位置处于弧形路线，0-采集位置处于直线路线
        suniLT = suniL[type*2 + posD]
        if posD == 0:
            sunit = suniLT[cotL % 4]
            cotL += 1
        else:
            sunit = suniLT[cotA % 4]
            cotA += 1

        for pi in range(posNum):
            traj_[cot] += [posD, yawd, sunit]
            cot += 1

    return traj_


def get_pos_by_traj(trajList, heightList, regionPoint, radius):
    # 根据轨迹的二维坐标、所在路线形状以及相机朝向，确定相机位姿
    posList = []
    org_vec = np.array([0, 0, 1])
    sta_ = np.radians(30)
    rollD = np.pi

    roadI = [[], []]  # 用于包含轨迹中每一段路径在轨迹中的起始位置以及该段路径所包含的采集位置数目（0-直线型，1-弧线型）
    trajLen = len(trajList)

    rIdx = 0 if trajList[0][2] == 0 else 1
    roadI[rIdx].append([0, 1])

    for i in range(1, trajLen):
        rIdx = 0 if trajList[i][2] == 0 else 1
        if trajList[i][2] == trajList[i-1][2]:
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
            tangen_vec = [-trajList[iIdx+1][0]+trajList[iIdx][0], -trajList[iIdx+1][1]+trajList[iIdx][1]]
            if trajList[iIdx][3] == 0:  # 行进方向的法向
                tangen_vec = gtls.rotate_coordinate(tangen_vec, 0, 0, -90)
            else:
                tangen_vec = gtls.rotate_coordinate(tangen_vec, 0, 0, -180)

            # 获取从世界坐标系到相机坐标系的变换矩阵（不理解为啥此处求完逆之后还要再一次求逆，第一次求逆有何作用）
            for idxp in range(iIdx, iIdx+posNum, 1):
                t_ = [trajList[idxp][0], 0, trajList[idxp][1]]
                sta = -sta_
                transM_X = utls.get_pitch_rot(sta)
                transM_Z = utls.get_roll_rot(rollD)
                transM_ZX = np.dot(transM_Z, transM_X)
                R_w2c = utls.get_Rw2c(tangen_vec, org_vec, transM_ZX)
                qv = posPar.rotmat2qvec(R_w2c).tolist()
                t = [t_[0]+regionPoint[0], -t_[1]+heightList[idxp]+regionPoint[2], t_[2]+regionPoint[1]]
                tn = utls.get_Tw2c_1(t, R_w2c)
                tv = tn.tolist()
                # 将变换矩阵转为四元数+平移向量的形式，并存放到images.bin里面以进行查看生成的轨迹效果
                posList.append(qv+tv)
        else:
            pts = [trajList[idxp][:2] for idxp in range(iIdx, iIdx+posNum, 1)]
            cen = gtls.calculate_circle_center_fast(pts, radius)
            ptsn = [[-elem[0] + cen[0], -elem[1] + cen[1]] for elem in pts]
            if trajList[iIdx][3] == 1:
                rotat = 90 if trajList[iIdx][4] == 1 else -90
                ptsn = gtls.rotate_coordinates(ptsn, 0, 0, rotat)

            for idxp_ in range(posNum):
                idxp = idxp_+iIdx
                t_ = [trajList[idxp][0], 0, trajList[idxp][1]]
                sta = -sta_
                transM_X = utls.get_pitch_rot(sta)
                transM_Z = utls.get_roll_rot(rollD)
                transM_ZX = np.dot(transM_Z, transM_X)
                R_w2c = utls.get_Rw2c(ptsn[idxp_], org_vec, transM_ZX)
                qv = posPar.rotmat2qvec(R_w2c).tolist()
                t = [t_[0]+regionPoint[0], -t_[1]+heightList[idxp]+regionPoint[2], t_[2]+regionPoint[1]]
                tn = utls.get_Tw2c_1(t, R_w2c)
                tv = tn.tolist()
                posList.append(qv + tv)

        cott[rIdx] += 1

    return posList