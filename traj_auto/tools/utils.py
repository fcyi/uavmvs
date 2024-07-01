import cv2
import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt
import sys

sys.path.append("./")

import pos_parse as posPar
import read_write_model as rwm


def get_center_bias(centerx, centery, cols, rows):
    return [centerx-cols//2, centery-rows//2]


def get_angle(arc_length, radius):
    angle = (arc_length / radius)
    return angle


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
        alpha = get_angle(accumArc, radius)  if accumArc > 1e-4 else 0 # 弧度制
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


def rotate_coordinate(coordinate, a, b, alpha):
    alpha_rad = math.radians(alpha)  # 将角度转换为弧度
    cos_alpha = math.cos(alpha_rad)
    sin_alpha = math.sin(alpha_rad)

    x, y = coordinate  # 拆分坐标元组
    rx = (x - a) * cos_alpha + (y - b) * sin_alpha
    ry = -(x - a) * sin_alpha + (y - b) * cos_alpha
    rotated_x = rx + a
    rotated_y = ry + b
    rotated_coord = [rotated_x, rotated_y]  # 组合旋转后的坐标
    return rotated_coord


def inverse_rotate_coordinate(coordinate, a, b, alpha):
    # 将二维点绕着(a, b)逆时针旋转alpha度
    # 确保传入的转角信息为角度
    alpha_rad = math.radians(alpha)
    cos_alpha = math.cos(alpha_rad)
    sin_alpha = math.sin(alpha_rad)
    x, y = coordinate
    x_prime = x - a
    y_prime = y - b
    x_new = x_prime * cos_alpha - y_prime * sin_alpha
    y_new = x_prime * sin_alpha + y_prime * cos_alpha

    x_rotated = x_new + a
    y_rotated = y_new + b
    rotated_coordinate = [x_rotated, y_rotated]

    return rotated_coordinate


def rotate_coordinates(coordinates, a, b, alpha):
    # 将二维点绕着(a, b)顺时针旋转alpha度
    # 确保传入的转角信息为角度
    rotated_coordinates = []
    alpha_rad = math.radians(alpha)  # 将角度转换为弧度
    cos_alpha = math.cos(alpha_rad)
    sin_alpha = math.sin(alpha_rad)
    for coord in coordinates:
        x, y = coord  # 拆分坐标元组
        rx = (x - a) * cos_alpha + (y - b) * sin_alpha
        ry = -(x - a) * sin_alpha + (y - b) * cos_alpha
        rotated_x = rx + a
        rotated_y = ry + b
        rotated_coord = [rotated_x, rotated_y]  # 组合旋转后的坐标
        rotated_coordinates.append(rotated_coord)
    return rotated_coordinates


def inverse_rotate_coordinates(coordinates, a, b, alpha):
    # 将二维点绕着(a, b)逆时针旋转alpha度
    # 确保传入的转角信息为角度
    alpha_rad = math.radians(alpha)
    rotated_coordinates = []
    cos_alpha = math.cos(alpha_rad)
    sin_alpha = math.sin(alpha_rad)
    for (x, y) in coordinates:
        x_prime = x - a
        y_prime = y - b
        x_new = x_prime * cos_alpha - y_prime * sin_alpha
        y_new = x_prime * sin_alpha + y_prime * cos_alpha

        x_rotated = x_new + a
        y_rotated = y_new + b
        rotated_coordinates.append([x_rotated, y_rotated])

    return rotated_coordinates


def calculate_circle_center(pts, radius, max_iterations=1000, tolerance=1e-6):
    points = np.array(pts)
    center = np.mean(points, axis=0)  # 初始近似圆心为所有点的平均位置

    for _ in range(max_iterations):
        distances = np.linalg.norm(points - center, axis=1)  # 计算每个点到近似圆心的距离
        difference = distances - radius  # 距离与给定半径之间的差异

        if np.all(np.abs(difference) < tolerance):  # 如果差异小于容忍度，则认为已经找到圆心
            break

        delta = difference / distances  # 计算每个点对于圆心的位移比例
        center += np.mean(delta[:, np.newaxis] * (points - center), axis=0)  # 更新圆心位置

    return center.tolist()


def calculate_circle_intersection(p1, p2, r1, r2):
    x1, y1 = p1[:]
    x2, y2 = p2[:]
    distance = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

    if (distance <= 1e-6) or (distance > (r1 + r2)):
        # 两个圆相离或相合，可能有0个交点或无数个交点
        return None
    elif (distance == r1 + r2) or (distance == math.fabs(r1-r2)):
        # 两个圆相切（外切或内切），有一个交点
        x_intersect = x1 + r1 * (x2 - x1) / distance
        y_intersect = y1 + r1 * (y2 - y1) / distance
        return [[x_intersect, y_intersect]]
    else:
        # distance > (r1 + r2)
        if distance < math.fabs(r1-r2):
            return None
        elif distance > math.fabs(r1-r2):
            # 两个圆相交，有两个交点
            l1 = (r1 ** 2 + distance ** 2 - r2 ** 2) / (2 * distance)
            l2 = math.sqrt(r1**2 - l1**2)
            px_ = x1 + l1*(x2-x1)/distance
            py_ = y1 + l1 * (y2 - y1) / distance
            px1 = px_-l2*(y2-y1)/distance
            py1 = py_+l2*(x2-x1)/distance
            px2 = px_ + l2 * (y2 - y1) / distance
            py2 = py_ - l2 * (x2 - x1) / distance
            return [[px1, py1], [px2, py2]]


def calculate_circle_center_fast(pts, radius):
    # 基于几何方法计算一系列点所在的圆
    # 输入的点最好根据一定的规则排好序，不排序也行，排序后续可以进一步优化
    ptsLen = len(pts)
    assert ptsLen >=3, "输入的点少于3个，圆心求解存在歧义"

    pt1 = pts[0]
    pt2 = pts[ptsLen-1]
    pth = pts[ptsLen//2]
    centerL = calculate_circle_intersection(pt1, pt2, radius, radius)
    assert centerL is not None, "输入的在圆上的点有问题"

    centerLen = len(centerL)
    assert centerLen < 3, "两圆交点算法有问题"

    if centerLen == 1:
        return centerL[0]
    else:
        c0 = centerL[0]
        c1 = centerL[1]
        distance0 = math.sqrt((c0[0]-pth[0])*(c0[0]-pth[0]) + (c0[1]-pth[1])*(c0[1]-pth[1]))
        distance1 = math.sqrt((c1[0]-pth[0])*(c1[0]-pth[0]) + (c1[1]-pth[1])*(c1[1]-pth[1]))
        if distance0 < distance1:
            return c1
        else:
            return c0


# 获取偏航角对应的旋转矩阵（也就是绕着第2个坐标分量进行旋转，对应的旋转矩阵）
def get_yaw_rot(org_vec, targt_vec):
    # org_vec为参考向量，targt_vec为坐标与原点所连成的归一化后的单位向量
    # 相当于采集位置计算关于建筑中心的偏航角，首先偏航的转轴是垂直于坐标向量以及参考向量所构造的平面，其次由于targt_vec与org_vec皆为单位向量，
    # 若偏航角为the，cos(the) = c，sin(the) = s
    c = np.dot(org_vec, targt_vec)
    n_vector = np.cross(org_vec, targt_vec)  # s*n = n_vector，n为旋转向量
    R_w2c = posPar.rotvec2rotmat(n_vector, c)
    return R_w2c


# 获取俯仰角对应的旋转矩阵（也就是绕着第1个坐标分量进行旋转，对应的旋转矩阵）
def get_pitch_rot(sta):
    # 绕着第一个坐标基顺时针旋转sta度，sta即为俯仰角
    transM_X = np.array([[1, 0, 0],
                         [0, np.cos(sta), np.sin(sta)],
                         [0, -np.sin(sta), np.cos(sta)]])
    return transM_X


# 获取翻滚角对应的旋转矩阵（也就是绕着第3个坐标分量进行旋转，对应的旋转矩阵）
def get_roll_rot(theta):
    # 绕着第三个坐标基顺时针旋转180度，也就是说翻滚角为180度
    transM_Z = np.array([[np.cos(theta), np.sin(theta), 0],
                         [-np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])
    return transM_Z


def get_rotate_vec(src_vec, theta, axis):
    # 将3维坐标向量沿着第axis个坐标向量顺时针旋转theta度
    res_vec = [0, 0, 0]
    rad = np.radians(theta)
    crad = np.cos(rad)
    srad = np.sin(rad)
    if axis == 0:
        res_vec[0] = src_vec[0]
        res_vec[1] = crad*src_vec[1]-srad*src_vec[2]
        res_vec[2] = srad*src_vec[1]+crad*src_vec[2]
    elif axis == 1:
        res_vec[0] = crad * src_vec[0] + srad * src_vec[2]
        res_vec[1] = src_vec[1]
        res_vec[2] = -srad * src_vec[0] + crad * src_vec[2]
    elif axis == 2:
        res_vec[0] = crad*src_vec[0]-srad*src_vec[1]
        res_vec[1] = srad*src_vec[0]+crad*src_vec[1]
        res_vec[2] = src_vec[2]
    return res_vec


def get_rotate_vec_fast(src_vec, thetas):
    rad0, rad1, rad2 = np.radians(thetas[0]), np.radians(thetas[1]), np.radians(thetas[2])
    c0, c1, c2 = np.cos(rad0), np.cos(rad1), np.cos(rad2)
    s0, s1, s2 = np.sin(rad0), np.sin(rad1), np.sin(rad2)
    res_vec = [0, 0, 0]
    res_vec[0] = c1*c2*src_vec[0] - c1*s2*src_vec[1] + s1*src_vec[2]
    res_vec[1] = (s0*s1*c2+c0*s2)*src_vec[0] + (c0*c2-s0*s1*s2)*src_vec[1] - s0*c1*src_vec[2]
    res_vec[2] = (s0*s2-c0*s1*c2)*src_vec[0] + (c0*s1*s2+s0*c2)*src_vec[1] + c0*c1*src_vec[2]
    return res_vec


def get_Rw2c(tangen_vec, org_vec, transM_ZX, controlAngle=(0, 0, 0)):
    # transM_ZX: 翻滚角对应的旋转矩阵与俯仰角对应的旋转矩阵的矩阵乘法
    # 此处之所以使用[x, z, y]形式的坐标，是为了使用左手系，之所以令z分量为0，是为了让坐标向量与参考向量
    # 计算切向方向，此处假设坐标系原点为[0, 0, 0]
    # 计算切向方向时，不需要考虑高度这些采集位置都处于0高度的位置
    # 即使旋转在平移之前，所以绕着z轴的旋转也会影响到偏航角
    org_vec_c = get_rotate_vec(org_vec, -controlAngle[2], 1)
    tangen3_vec = np.array([tangen_vec[0], 0, tangen_vec[1]])
    tangen3_norm = np.linalg.norm(tangen3_vec)
    tangen3_vec = tangen3_vec / tangen3_norm
    R_w2c = get_yaw_rot(org_vec_c, tangen3_vec)
    R_w2c = np.dot(R_w2c, np.array([[-1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, 1]]))
    R_w2c = np.dot(transM_ZX, R_w2c)
    return R_w2c


def get_Tw2c(t, R):
    # 这种方法有数值溢出的问题
    # t: 相机到世界坐标系的平移向量
    # R: 旋转矩阵的逆矩阵
    t = np.reshape(t, (3, 1))
    Rt = np.hstack((R, t))
    Rt = np.vstack((Rt, np.array([0, 0, 0, 1]).T))
    Rt = np.linalg.inv(Rt)
    return Rt[:3, 3]


def get_Tw2c_1(t, R):
    # t: 相机到世界坐标系的平移向量
    # R: 旋转矩阵
    t = np.reshape(t, (3, 1))
    return (-np.dot(R, t)).reshape((3,))


def get_Tc2w_1(t, R):
    t = np.reshape(t, (3, 1))
    return (-np.dot(R.transpose(), t)).reshape((3,))


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

    roadNodesY = rotate_coordinates(roadNodesY, 0, 0, 90)
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
        roadNodesY = rotate_coordinates(roadNodesY, 0, 0, 90)

    for i in range(nodeNumY):
        roadNodesY[i][0] += width / 2
        roadNodesY[i][1] += height / 2

    return roadNodesR, roadNodesX, roadNodesY


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


def get_pos_by_traj_sim(trajList, heightList, regionPoint, radius, relativeV=(0, 0, 0, 0, 0, 0)):
    # 根据轨迹的二维坐标、所在路线形状以及相机朝向，确定相机位姿
    posList = []
    org_vec = np.array([0, 0, 1])
    sta_ = np.radians(30)
    rollD = np.pi

    trajLen = len(trajList)
    assert trajLen > 1, "路线上的采集位置数目少于2，航向难以确定"

    controlPoint = [regionPoint[i] + relativeV[i + 3] for i in range(3)]
    controlAngle = [relativeV[i] for i in range(3)]

    for tidx in range(trajLen):
        if tidx == 0:
            tangen_vec = [-trajList[tidx+1][0] + trajList[tidx][0], -trajList[tidx+1][1] + trajList[tidx][1]]
        elif tidx == trajLen-1:
            tangen_vec = [-trajList[tidx][0] + trajList[tidx-1][0], -trajList[tidx][1] + trajList[tidx-1][1]]
        else:
            tangen_vec = [(-trajList[tidx + 1][0] + trajList[tidx-1][0])/2.,
                          (-trajList[tidx + 1][1] + trajList[tidx-1][1])/2.]

        if trajList[tidx][3] == 0:  # 行进方向的法向
            tangen_vec = inverse_rotate_coordinate(tangen_vec, 0, 0, 90)
        else:
            tangen_vec = inverse_rotate_coordinate(tangen_vec, 0, 0, 180)

        t_ = [trajList[tidx][0], 0, trajList[tidx][1]]
        t_ = get_rotate_vec(t_, controlAngle[0], 0)
        t_ = get_rotate_vec(t_, controlAngle[2], 1)
        t_ = get_rotate_vec(t_, controlAngle[1], 2)
        star = np.arctan(t_[1] / np.sqrt(t_[0] ** 2 + t_[2] ** 2))
        sta = sta_ + star
        sta = -sta
        transM_X = get_pitch_rot(sta)
        transM_Z = get_roll_rot(rollD)
        transM_ZX = np.dot(transM_Z, transM_X)
        R_w2c = get_Rw2c(tangen_vec, org_vec, transM_ZX, tuple(controlAngle))
        qv = posPar.rotmat2qvec(R_w2c).tolist()

        t = [t_[0] + controlPoint[0], -t_[1] + heightList[tidx] + controlPoint[2], t_[2] + controlPoint[1]]
        # tn = get_Tw2c(t, R)
        tn = get_Tw2c_1(t, R_w2c)
        tv = tn.tolist()
        # 将变换矩阵转为四元数+平移向量的形式，并存放到images.bin里面以进行查看生成的轨迹效果
        posList.append(qv + tv)

    return posList


def get_pos_by_traj(trajList, heightList, regionPoint, radius, relativeV=(0, 0, 0, 0, 0, 0)):
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

    controlPoint = [regionPoint[i] + relativeV[i + 3] for i in range(3)]
    controlAngle = [relativeV[i] for i in range(3)]

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
                tangen_vec = inverse_rotate_coordinate(tangen_vec, 0, 0, 90)
            else:
                tangen_vec = inverse_rotate_coordinate(tangen_vec, 0, 0, 180)

            # 获取从世界坐标系到相机坐标系的变换矩阵（不理解为啥此处求完逆之后还要再一次求逆，第一次求逆有何作用）
            for idxp in range(iIdx, iIdx+posNum, 1):
                t_ = [trajList[idxp][0], 0, trajList[idxp][1]]
                t_ = get_rotate_vec(t_, controlAngle[0], 0)
                t_ = get_rotate_vec(t_, controlAngle[2], 1)
                t_ = get_rotate_vec(t_, controlAngle[1], 2)
                star = np.arctan(t_[1] / np.sqrt(t_[0] ** 2 + t_[2] ** 2))
                sta = sta_ + star
                sta = -sta
                transM_X = get_pitch_rot(sta)
                transM_Z = get_roll_rot(rollD)
                transM_ZX = np.dot(transM_Z, transM_X)
                R_w2c = get_Rw2c(tangen_vec, org_vec, transM_ZX, tuple(controlAngle))
                qv = posPar.rotmat2qvec(R_w2c).tolist()
                # R = np.linalg.inv(R_w2c)
                t = [t_[0]+controlPoint[0], -t_[1]+heightList[idxp]+controlPoint[2], t_[2]+controlPoint[1]]
                # tn = get_Tw2c(t, R)
                tn = get_Tw2c_1(t, R_w2c)
                tv = tn.tolist()
                # 将变换矩阵转为四元数+平移向量的形式，并存放到images.bin里面以进行查看生成的轨迹效果
                posList.append(qv+tv)
        else:
            pts = [trajList[idxp][:2] for idxp in range(iIdx, iIdx+posNum, 1)]
            cen = calculate_circle_center_fast(pts, radius)
            ptsn = [[-elem[0] + cen[0], -elem[1] + cen[1]] for elem in pts]
            if trajList[iIdx][3] == 1:
                rotat = 90 if trajList[iIdx][4] == 1 else -90
                ptsn = rotate_coordinates(ptsn, 0, 0, rotat)

            for idxp_ in range(posNum):
                idxp = idxp_+iIdx
                t_ = [trajList[idxp][0], 0, trajList[idxp][1]]
                t_ = get_rotate_vec(t_, controlAngle[0], 0)
                t_ = get_rotate_vec(t_, controlAngle[2], 1)
                t_ = get_rotate_vec(t_, controlAngle[1], 2)
                star = np.arctan(t_[1] / np.sqrt(t_[0] ** 2 + t_[2] ** 2))
                sta = sta_ + star
                sta = -sta
                transM_X = get_pitch_rot(sta)
                transM_Z = get_roll_rot(rollD)
                transM_ZX = np.dot(transM_Z, transM_X)
                R_w2c = get_Rw2c(ptsn[idxp_], org_vec, transM_ZX, tuple(controlAngle))
                qv = posPar.rotmat2qvec(R_w2c).tolist()
                t = [t_[0]+controlPoint[0], -t_[1]+heightList[idxp]+controlPoint[2], t_[2]+controlPoint[1]]
                # R = np.linalg.inv(R_w2c)
                # tn = get_Tw2c(t, R)
                tn = get_Tw2c_1(t, R_w2c)
                tv = tn.tolist()
                posList.append(qv + tv)

        cott[rIdx] += 1

    return posList


def UniSampling(number):
    return np.linspace(0, 360, number, endpoint=True)


def LinerSampling(start, stop, number):
    return np.linspace(start, stop, number, endpoint=True)


def get_loop_pos(radius, heightS, heightE, posNum, regionPoint,
                 yawNumRadPerPos=0,
                 relativeV=(0, 0, 0, 0, 0, 0)):
    """
    :param radius:
    :param heightE:
    :param posNum:
    :param regionPoint: 区域中心点
    :param relativeV:前三维为绕着x/y/z所对应的坐标基底进行顺时针旋转的度数，后三维为需要对控制点沿着x/y/z进行移动的大小
    :return:
    """
    yawRands = []
    yawLen = 0
    yawSeg = 0
    if type(yawNumRadPerPos).__name__ == 'int':
        yawLen = 1 if not yawNumRadPerPos else yawNumRadPerPos
        yawSeg = yawNumRadPerPos
        if yawNumRadPerPos > 0:  # 用于产生测试集，对偏航角进行随机扰动
            random_seed = 12345  # 随机种子
            random_generator = random.Random(random_seed)  # 创建带有指定种子的随机数生成器
            num_samples = posNum*yawNumRadPerPos  # 生成的随机浮点数的数量
            yawRands += [random_generator.uniform(-90, 90) for _ in range(num_samples)]  # 生成一系列随机浮点数
        else:  # 用于产生训练集
            yawRands = [0]
    elif type(yawNumRadPerPos).__name__ == 'tuple':
        for _ in range(posNum):
            yawRands += list(yawNumRadPerPos)
        yawLen = len(yawNumRadPerPos)
        yawSeg = yawLen

    coord = []
    # posNum = 50
    print(posNum)
    angle_list = UniSampling(posNum)
    # loopNum = 10
    loopNum = len(angle_list)
    temp1 = angle_list[loopNum:]
    temp2 = angle_list[:loopNum]
    angle_list = np.concatenate((temp1, temp2))

    height_list = LinerSampling(heightS, heightE, posNum)  # 无人机环绕上升时，每一圈上每一个采集点处的高度
    r_list = LinerSampling(radius, radius, posNum)  # 生成每一个位置所对应的半径，因为环绕的建筑的俯瞰形状很可能不是一个规则的圆

    controlPoint = [regionPoint[i]+relativeV[i+3] for i in range(3)]
    controlAngle = [relativeV[i] for i in range(3)]
    # circle formula
    org = [0, 0]
    for angle in range(angle_list.shape[0]):
        x = org[0] + r_list[angle] * np.cos(np.radians(angle_list[angle]))
        y = org[1] + r_list[angle] * np.sin(np.radians(angle_list[angle]))
        coord.append([x, y])

    coord = np.asarray(coord)
    org_vec = np.array([org[0], 0, org[1]+1])  # 参考向量，指向成像场景中心

    # r = 3  # 计算俯仰角时所考虑的半径，此处俯仰角的调整是为了相机镜头始终朝着重建场景中心
    # sta = np.arctan(height_list[i] / r)  # 这是为了让相机镜头始终朝着建筑物中心

    sta_ = np.radians(0)
    posList = []
    for i in range(posNum):
        t_ = [coord[i][0], 0, coord[i][1]]
        t_ = get_rotate_vec(t_, controlAngle[0], 0)
        t_ = get_rotate_vec(t_, controlAngle[2], 1)
        t_ = get_rotate_vec(t_, controlAngle[1], 2)
        star = np.arctan(t_[1]/np.sqrt(t_[0]**2 + t_[2]**2))
        sta = sta_ + star
        sta = -sta
        transM_X = get_pitch_rot(sta)
        transM_Z = get_roll_rot(np.pi)
        transM_ZX = np.dot(transM_Z, transM_X)

        for ridx in range(yawLen):
            print(i, yawSeg, ridx)
            coordYawRand = rotate_coordinate(coord[i], 0, 0, yawRands[i*yawSeg+ridx])
            R_w2c = get_Rw2c([-coordYawRand[0], -coordYawRand[1]], org_vec, transM_ZX, tuple(controlAngle))
            qv = posPar.rotmat2qvec(R_w2c)
            # 获取从世界坐标系到相机坐标系的变换矩阵（不理解为啥此处求完逆之后还要再一次求逆，第一次求逆有何作用）
            t = [t_[0] + controlPoint[0], -t_[1]+height_list[i] + controlPoint[2], t_[2]+controlPoint[1]]
            print(t, R_w2c)
            tv = get_Tw2c_1(t, R_w2c)
            # 将变换矩阵转为四元数+平移向量的形式，并存放到images.bin里面以进行查看生成的轨迹效果
            posList.append(qv.tolist() + tv.tolist())

    return posList


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


def tran_to_airsim(posList_):
    H_ = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, -1, 0]])
    H_inv_ = np.linalg.inv(H_)
    posLen_ = len(posList_)
    posListTran_ = []
    for pos_ in posList_:
        Rw2c = posPar.qvec2rotmat(pos_[:4])
        cen = np.array(pos_[4:]).reshape((3, 1))

        tw = -np.dot(H_, np.dot(np.linalg.inv(Rw2c), cen))

        Rw2c2_ = posPar.change_rot(Rw2c, axis=((0, 0), (1, 0), (2, -90)))
        Rw2c2 = posPar.change_rot(Rw2c2_, axis=((0, 0), (1, 0), (2, -1)))
        Rw2cH = np.dot(H_, np.dot(Rw2c2, H_inv_))
        qvw2c = posPar.rotmat2qvec(Rw2cH)

        twL = [tw[i_][0] for i_ in range(3)]
        posListTran_.append(qvw2c.tolist()+twL)

    return posListTran_


def tran_from_airsim(posList_):
    # 由于colmap中，相机姿态的参考向量为y轴
    # 而仿真场景中，相机姿态的参考向量为x轴
    # 仿真场景与colmap之间的坐标系变换为[[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    # 因此，为了相机姿态与期望的一致相应的仿射变换矩阵为：
    # [[0, 1, 0], [1, 0, 0], [0, 0, 1]]*[[1, 0, 0], [0, 0, 1], [0, 1, 0]] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    # H = np.array([[0, 0, 1],
    #               [1, 0, 0],
    #               [0, 1, 0]])  # colmap坐标系到仿真环境坐标系的变换矩阵
    H_ = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, -1, 0]])
    # H = np.array([[1, 0, 0],
    #               [0, 1, 0],
    #               [0, 0, 1]])
    H_inv_ = np.linalg.inv(H_)

    posListTran_ = []
    for pos_ in posList_:
        R = posPar.qvec2rotmat(pos_[:4])
        tva = np.array(pos_[4:]).reshape((3, 1))

        Ri_ = np.dot(H_inv_, np.dot(R, H_))
        Ri = posPar.change_rot(Ri_, axis=((0, 0), (1, 0), (2, -1)))
        Rp = np.dot(H_, np.dot(Ri, H_inv_))
        qvi = posPar.rotmat2qvec(Ri)

        tvai = -np.dot(H_inv_, np.dot(Rp, tva))

        tvi = [tvai[i][0] for i in range(3)]
        posListTran_.append(qvi.tolist() + tvi)

    return posListTran_


def tran_to_blender(posList_, isCamera=True):
    H_ = np.array([[1, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0]])
    H_inv_ = np.linalg.inv(H_)
    posListTran_ = []
    for pos_ in posList_:
        Rw2c = posPar.qvec2rotmat(pos_[:4])
        rotVec = np.array(posPar.rotmat2rotvec(Rw2c))

        # Rw2c_ = change_rot(Rw2c, axis=((0, 0), (1, 90), (2, 0)))
        # Rw2c_ = change_rot(Rw2c, axis=((0, 0), (1, 0), (2, 0)))
        Rw2c_ = Rw2c
        cen = np.array(pos_[4:]).reshape((3, 1))

        if not isCamera:
            tw = -np.dot(H_, np.dot(np.linalg.inv(Rw2c_), cen))
        else:
            tw = np.dot(H_, cen)

        Rw2cH = np.dot(H_, np.dot(Rw2c_, H_inv_))

        # rotVecH = np.dot(H_, rotVec)
        # theta = np.linalg.norm(rotVecH)
        # s = np.sin(theta)
        # c = np.cos(theta)
        # if np.isclose(theta, 0):
        #     srotVecH = rotVecH
        #     c = 1.
        # else:
        #     srotVecH = rotVecH / theta
        #     srotVecH = srotVecH * s
        # Rw2cH = posPar.rotvec2rotmat(srotVecH.tolist(), c)

        qvw2c = posPar.rotmat2qvec(Rw2cH)
        twL = [tw[i_][0] for i_ in range(3)]
        posListTran_.append(qvw2c.tolist() + twL)

    return posListTran_


def tum_txt_write(posList, workDir, fileName="auto_traj_fukan", axisType='airsim'):
    if not os.path.exists(workDir):
        print("have not valid workdir, if want to save, please sure that!!!!!!")
    else:
        # while 1:
        #     if not os.path.exists(os.path.join(workDir, fileName + ".txt")):
        #         break
        #     fileName += "(copy)"
        txtPath = os.path.join(workDir, fileName + ".txt")

        posListTran = []
        if axisType == 'colmap':
            posListTran = posList
        elif axisType == 'airsim':
            posListTran = tran_to_airsim(posList)
        posLen = len(posList)
        with open(txtPath, 'w') as f:
            f.write("# index tx ty tz qx qy qz qw\n")
            for i in range(posLen):
                f.write("{}".format(i))
                for j in range(3):
                    f.write(",{}".format(posList[i][4 + j]))
                for j in range(3):
                    f.write(",{}".format(posList[i][1 + j]))
                f.write(",{}".format(posList[i][0]))
                if i < (posLen-1):
                    f.write("\n")
            f.close()


def tum_txt_read(filePath, frameRatio=0.001, segc=",", axisType='airsim'):
    posListTmp = []
    posList = []
    frameSeg = 1./frameRatio
    fcot = 0

    with open(filePath, "r") as f:
        line = f.readline()
        while line:
            if line[0] != '#':
                # print(line)
                if fcot % frameSeg == 0:
                    line = line.strip()
                    lins = line.split(segc)
                    lies = [float(elem) for elem in lins[1:]]
                    posListTmp.append([lies[6], lies[3], lies[4], lies[5], lies[0], lies[1], lies[2]])
                fcot += 1
            line = f.readline()

        f.close()

    if axisType == 'colmap':
        posList = posListTmp
    elif axisType == 'airsim':
        posList = tran_from_airsim(posListTmp)
    return posList


def tum_txt_test(txtPath, binPath, frameRatio=0.001, segc=",", dstPath=""):
    posList = tum_txt_read(txtPath, frameRatio, segc)
    if len(posList) > 0:
        images_bin_write(binPath, posList, dstPath)
    else:
        print("have not test poses")

