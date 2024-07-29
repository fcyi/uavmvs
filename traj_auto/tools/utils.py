import copy

import cv2
import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt
import sys

sys.path.append("./")

import pos_parse as posPar
import geo_tools as gtls
import base_tools as btls


def line_traj_2D(psxy, pexy, residual, step, refineStepRatio=1):
    """
    生成psxyz与pexyz之间的直线型轨迹的xyz，并且产生下一段路线上的距离残差
    :param psxyz: 起始点
    :param pexyz: 终止点
    :param residual:上一段路线中没走完的长度，这一长度会累积在整段轨迹中
    :param step: 步长
    :param refineStepRatio:细化步长与普通步长之间的比例，一般用于有拐角的地方，设为1时细化步长不会起作用
    :return:
    """
    assert (psxy[0] != pexy[0]) or (psxy[1] != pexy[1]), "完全相同的起始终止点，产生不了直线路线"

    dxy = [pexy[i] - psxy[i] for i in range(2)]
    disxy = math.sqrt(dxy[0] ** 2 + dxy[1] ** 2)

    accumStart = residual
    accumEnd = disxy
    accumV, residualArc = btls.get_accumList(accumStart, accumEnd, step, refineStepRatio)

    accumLen = len(accumV)
    trajTmp = [[accumV[i], 0] for i in range(accumLen)]

    traj = []
    if accumLen > 0:
        if dxy[1] == 0:
            mulfac = -1 if psxy[0] > pexy[0] else 1
            traj = (mulfac*np.array(trajTmp)).tolist()
        else:
            dxy = np.array(dxy)
            dxyTmp = np.array([disxy, 0])

            # 计算转轴、转角以及旋转矩阵
            cosA = (dxyTmp[0]*dxy[0] + dxyTmp[1]*dxy[1]) / (dxyTmp[0]**2 + dxyTmp[1]**2)
            sinA = (dxyTmp[0]*dxy[1] - dxyTmp[1]*dxy[0]) / (dxyTmp[0]**2 + dxyTmp[1]**2)
            rotationMat = np.array([[cosA, -sinA],
                                    [sinA, cosA]])

            trajTmpArray = np.array(trajTmp)
            trajTmpTransArray = np.matmul(rotationMat, trajTmpArray.transpose())
            traj = trajTmpTransArray.transpose().tolist()
        for i in range(accumLen):
            traj[i][0] += psxy[0]
            traj[i][1] += psxy[1]

    return traj, residualArc


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


def get_Rw2c(tangen_vec, org_vec, transM_ZX):
    # transM_ZX: 翻滚角对应的旋转矩阵与俯仰角对应的旋转矩阵的矩阵乘法
    # 此处之所以使用[x, z, y]形式的坐标，是为了使用左手系，之所以令z分量为0，是为了让坐标向量与参考向量
    # 计算切向方向，此处假设坐标系原点为[0, 0, 0]
    # 计算切向方向时，不需要考虑高度这些采集位置都处于0高度的位置
    # 即使旋转在平移之前，所以绕着z轴的旋转也会影响到偏航角
    tangen3_vec = np.array([tangen_vec[0], 0, tangen_vec[1]])
    tangen3_norm = np.linalg.norm(tangen3_vec)
    tangen3_vec = tangen3_vec / tangen3_norm
    R_w2c = get_yaw_rot(org_vec, tangen3_vec)
    R_w2c = np.dot(R_w2c, np.array([[-1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, 1]]))
    R_w2c = np.dot(transM_ZX, R_w2c)
    return R_w2c


def get_Tw2c_1(t, R, isTranspose=False):
    # isTranspose==false: w2c, isTranspose==true: c2w
    # t: 相机到世界坐标系的平移向量
    # R: 旋转矩阵
    rotM = R if not isTranspose else R.transpose()
    t = np.reshape(t, (3, 1))
    return (-np.dot(rotM, t)).reshape((3,))


def get_traj_by_node_sim(nodeList, step, rratio, yawType=-2, isClose=False):
    """
    nodeList: 航线节点
    step: 航向间隔
    rstep: 拐弯处的航向间隔
    yawType: 偏航角类型，1-相机朝向与飞机行进方向一致，2-相机镜头朝着行进方向逆时针90度的方向，0-相机镜头朝着行进方向顺时针90度的方向，-2-大疆的行进方向
    """
    traj_ = []
    rest_ = 0
    assert yawType in (0, 1, 2, -2), '偏航角定义存在问题'

    lineds = []  # 每一段直线或弧线所包含的采集位置数目
    nodeLen = len(nodeList)
    travLen = nodeLen if isClose else nodeLen-1
    for i in range(travLen):
        j = (i+1) % nodeLen
        trajt_, rest_ = line_traj_2D(nodeList[i], nodeList[j], rest_, step, rratio)
        traj_ += trajt_
        lineds.append(len(trajt_))

    cot = 0
    # 0-行进方式为顺时针，1-行进方向为逆时针
    suniL = [0, 0, 1, 1]
    # 大疆的相机朝向
    yawList = [0, 1, 2, 1]  # 遍历的路线为从上到下，从左到右
    for i in range(len(lineds)):  # 对每一条路线进行遍历
        posNum = lineds[i]
        sunit = suniL[i % 4]

        if yawType != -2:
            yawd = yawType
        else:
            yawd = yawList[i % 4]

        for pi in range(posNum):
            # 获取位置的路线信息(0表示直线)、镜头方向信息、路线行进是否为逆时针（主要是针对于弧型路线，求相机朝向时很必要）
            traj_[cot] += [0, yawd, sunit]
            cot += 1

    return traj_


def get_pos_by_traj_sim(trajList, heightList, regionPoint, pitchD=30):
    # 根据轨迹的二维坐标、所在路线形状以及相机朝向，确定相机位姿
    posList = []
    org_vec = np.array([0, 0, 1])
    sta_ = np.radians(pitchD)
    rollD = np.pi

    trajLen = len(trajList)
    assert trajLen > 1, "路线上的采集位置数目少于2，航向难以确定"

    for tidx in range(trajLen):
        if tidx == 0:
            tangen_vec = [-trajList[tidx+1][0] + trajList[tidx][0], -trajList[tidx+1][1] + trajList[tidx][1]]
        elif tidx == trajLen-1:
            tangen_vec = [-trajList[tidx][0] + trajList[tidx-1][0], -trajList[tidx][1] + trajList[tidx-1][1]]
        else:
            tangen_vec = [(-trajList[tidx + 1][0] + trajList[tidx-1][0])/2.,
                          (-trajList[tidx + 1][1] + trajList[tidx-1][1])/2.]

        if trajList[tidx][3] == 0:  # 行进方向顺时针90度
            tangen_vec = gtls.rotate_coordinate(tangen_vec, 0, 0, -90)
        elif trajList[tidx][3] == 1:  # 与相近方向上一致
            tangen_vec = gtls.rotate_coordinate(tangen_vec, 0, 0, -180)
        elif trajList[tidx][3] == 2:  # 行进方向逆时针90度
            tangen_vec = gtls.rotate_coordinate(tangen_vec, 0, 0, 90)

        t_ = [trajList[tidx][0], 0, trajList[tidx][1]]
        transM_X = get_pitch_rot(sta_)
        transM_Z = get_roll_rot(rollD)
        transM_ZX = np.dot(transM_Z, transM_X)
        R_w2c = get_Rw2c(tangen_vec, org_vec, transM_ZX)
        qv = posPar.rotmat2qvec(R_w2c).tolist()

        t = [t_[0] + regionPoint[0], -t_[1] + heightList[tidx] + regionPoint[2], t_[2] + regionPoint[1]]
        tn = get_Tw2c_1(t, R_w2c)
        tv = tn.tolist()
        # 将变换矩阵转为四元数+平移向量的形式，并存放到images.bin里面以进行查看生成的轨迹效果
        posList.append(qv + tv)

    return posList


def get_loop_pos(radius, heightS, heightE, posNum, regionPoint,
                 pitchD=30,
                 yawNumRadPerPos=0):
    """
    :param radius:
    :param heightE:
    :param posNum:
    :param regionPoint: 区域中心点
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
    angle_list = btls.UniSampling(posNum)
    loopNum = len(angle_list)
    temp1 = angle_list[loopNum:]
    temp2 = angle_list[:loopNum]
    angle_list = np.concatenate((temp1, temp2))

    height_list = btls.LinerSampling(heightS, heightE, posNum)  # 无人机环绕上升时，每一圈上每一个采集点处的高度
    r_list = btls.LinerSampling(radius, radius, posNum)  # 生成每一个位置所对应的半径，因为环绕的建筑的俯瞰形状很可能不是一个规则的圆

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

    sta_ = np.radians(pitchD)
    posList = []
    for i in range(posNum):
        t_ = [coord[i][0], 0, coord[i][1]]
        transM_X = get_pitch_rot(sta_)
        transM_Z = get_roll_rot(np.pi)
        transM_ZX = np.dot(transM_Z, transM_X)

        for ridx in range(yawLen):
            coordYawRand = gtls.rotate_coordinate(coord[i], 0, 0, yawRands[i*yawSeg+ridx])
            R_w2c = get_Rw2c([-coordYawRand[0], -coordYawRand[1]], org_vec, transM_ZX)
            qv = posPar.rotmat2qvec(R_w2c)
            # 获取从世界坐标系到相机坐标系的变换矩阵（不理解为啥此处求完逆之后还要再一次求逆，第一次求逆有何作用）
            t = [t_[0] + regionPoint[0], -t_[1]+height_list[i] + regionPoint[2], t_[2]+regionPoint[1]]
            tv = get_Tw2c_1(t, R_w2c)
            # 将变换矩阵转为四元数+平移向量的形式，并存放到images.bin里面以进行查看生成的轨迹效果
            posList.append(qv.tolist() + tv.tolist())

    return posList


def get_polygon_pos(vertexs, heightS, heightE, posNum, regionPoint, rratio, pitchD=30, isFocus=False):
    """
    :param radius:
    :param heightE:
    :param posNum:
    :param regionPoint: 区域中心点
    :return:
    """
    polgonLen = gtls.get_polygon_len(vertexs)
    stepL = polgonLen / float(posNum)
    stepR = 4*stepL
    stepS = 0
    while stepL < stepR:
        stepS = (stepL+stepR) / 2
        posTemp = get_traj_by_node_sim(vertexs, stepS, rratio, isClose=True)
        posTempLen = len(posTemp)
        if (abs(posTempLen-posNum) <= 2) or abs(stepL-stepR) <= 1e-6:
            break
        elif posTempLen > posNum:
            stepL = stepS
        else:
            stepR = stepS

    coordTmp = get_traj_by_node_sim(vertexs, stepS, rratio, isClose=True)
    posNumTmp = len(coordTmp)
    height_list = btls.LinerSampling(heightS, heightE, posNumTmp)  # 无人机环绕上升时，每一圈上每一个采集点处的高度

    org = [0, 0]
    coord = []
    for coord_ in coordTmp:
        coord.append(coord_[:2])

    coord = np.asarray(coord)
    org_vec = np.array([org[0], 0, org[1]+1])  # 参考向量，指向成像场景中心

    sta_ = np.radians(pitchD)
    posList = []
    for i in range(posNumTmp):
        t_ = [coord[i][0], 0, coord[i][1]]
        if i == 0:
            tangen_vec = [-coord[i+1][0] + coord[i][0], -coord[i+1][1] + coord[i][1]]
        elif i == posNumTmp-1:
            tangen_vec = [-coord[i][0] +coord[i-1][0], -coord[i][1] + coord[i-1][1]]
        else:
            tangen_vec = [(-coord[i + 1][0] + coord[i-1][0])/2.,
                          (-coord[i + 1][1] + coord[i-1][1])/2.]

        if not isFocus:
            tangen_vec = gtls.rotate_coordinate(tangen_vec, 0, 0, -90)
        else:
            tangen_vec = [-coord[i][0], -coord[i][1]]

        transM_X = get_pitch_rot(sta_)
        transM_Z = get_roll_rot(np.pi)
        transM_ZX = np.dot(transM_Z, transM_X)
        R_w2c = get_Rw2c(tangen_vec, org_vec, transM_ZX)
        qv = posPar.rotmat2qvec(R_w2c)
        t = [t_[0] + regionPoint[0], -t_[1]+height_list[i] + regionPoint[2], t_[2]+regionPoint[1]]
        tv = get_Tw2c_1(t, R_w2c)
        posList.append(qv.tolist() + tv.tolist())

    return posList, stepS


def change_rot_for_airsim(srcRot):
    # 参考pos_parse中的change_rot()方法
    rottmp = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])
    rot = np.dot(srcRot, rottmp)
    s1 = -rot[2][1]
    yawA = 0  # 偏航角处于[0, pi]
    if np.isclose(s1, 1) or s1 >= 1.:
        pitchA = np.pi / 2
    elif np.isclose(s1, -1) or s1 <= -1.:
        pitchA = -np.pi / 2
    else:
        pitchA = np.arcsin(s1)  # 俯仰角处于(-pi/2, pi/2)
    cp = np.cos(pitchA)

    if not np.isclose(pitchA, np.pi / 2) or not np.isclose(pitchA, -np.pi / 2):
        sy = rot[2][0] / cp
        cy = rot[2][2] / cp
        if cy > 1:
            cy = 1.0
        elif cy < -1:
            cy = -1.
        if sy > 1:
            sy = 1.0
        elif sy < -1:
            sy = -1.
        yawA = np.arccos(cy)
        if sy < 0:
            yawA = 2 * np.pi - yawA
    else:
        yawA = np.arccos(-rot[0][0])

    yawP = np.array([[np.cos(yawA), 0, -np.sin(yawA)],
                     [0, 1, 0],
                     [np.sin(yawA), 0, np.cos(yawA)]])
    yawAH = yawA-90
    yawPH = np.array([[np.cos(yawAH), 0, -np.sin(yawAH)],
                     [0, 1, 0],
                     [np.sin(yawAH), 0, np.cos(yawAH)]])
    rpP = np.dot(rot, yawP.transpose())  # np.dot(rollP, pitchP)
    rpPH = np.array([
        [rpP[2][2], -rpP[1][2], rpP[0][2]],
        [-rpP[2][1], rpP[1][1], -rpP[0][1]],
        [rpP[2][0], -rpP[1][0], rpP[0][0]]
    ], dtype=rpP.dtype)

    resRot = np.dot(rpPH, yawPH)
    resRot = np.dot(resRot, rottmp)
    return resRot


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

        # 注意，airsim要求的是世界坐标系下的旋转和平移，此处的旋转仍然是相机坐标系下，而平移是世界坐标系下，
        # 若用相机坐标则会导致飞机的俯仰角和翻滚角出现问题
        # 此外，不知为何在airsim中colmap中的俯仰角的负数会被用作翻滚角，翻滚角会被用作俯仰角，此外偏航角还差了90度
        # Rw2c2_ = posPar.change_rot(Rw2c, axis=((0, pitchD), (1, pitchD), (2, -90)))
        Rw2c2_ = change_rot_for_airsim(Rw2c)
        Rw2c2 = Rw2c2_.transpose()
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
    atest = np.pi/4
    ctest = np.cos(atest)
    stest = np.sin(atest)
    # 绕x轴顺时针旋转
    # H_ = np.array([[1, 0, 0],
    #                [0, ctest, -stest],
    #                [0, stest, ctest]])
    #
    # RotModX = np.array([[1, 0, 0],
    #                     [0, 0, 1],
    #                     [0, 1, 0]])
    #
    # HRot_ = np.dot(np.dot(RotModX, H_), RotModX.transpose())

    ##################################################################

    # 绕z轴顺时针旋转
    H_ = np.array([[ctest, 0, stest],
                   [0, 1, 0],
                   [-stest, 0, ctest]])

    RotModX = np.array([[0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0]])

    HRot_ = np.dot(np.dot(RotModX, H_), RotModX.transpose())

    ####################################################################

    # 绕y轴顺时针旋转
    # H_ = np.array([[ctest, -stest, 0],
    #                [stest, ctest, 0],
    #                [0, 0, 1]])
    #
    # RotModX = np.array([[0, 1, 0],
    #                     [1, 0, 0],
    #                     [0, 0, 1]])
    #
    # HRot_ = np.dot(np.dot(RotModX, H_), RotModX.transpose())

    #####################################################################

    posListTran_ = []
    for pos_ in posList_:
        Rw2c = posPar.qvec2rotmat(pos_[:4])
        cen = np.array(pos_[4:]).reshape((3, 1))
        tw = -np.dot(Rw2c.transpose(), cen)

        # rotVec = np.array(posPar.rotmat2rotvec(Rw2c))
        # rotVec_ = rotVec
        # theta = np.linalg.norm(rotVec_)
        # s = np.sin(theta)
        # c = np.cos(theta)
        # if np.isclose(theta, 0):
        #     srotVec_ = rotVec_
        #     c = 1.
        # else:
        #     srotVec_ = rotVec_ / theta
        #     srotVec_ = srotVec_ * s
        # srotVec_ = np.dot(HRot_, srotVec_)
        # Rw2c_ = posPar.rotvec2rotmat(srotVec_.tolist(), c)

        # Rw2c_ = Rw2c
        Rw2c_ = np.dot(Rw2c, HRot_)
        # Rw2c_ = np.dot(H_, Rw2c)
        print("===================")
        print(Rw2c)
        print(Rw2c_)
        tw_ = np.dot(H_, tw)
        cen_ = -np.dot(Rw2c_, tw_)
        print(tw)
        print(tw_)
        print("-----------------")

        qvw2c_ = posPar.rotmat2qvec(Rw2c_)
        cenL_ = [cen_[i_][0] for i_ in range(3)]
        posListTran_.append(qvw2c_.tolist() + cenL_)

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
        posLen = len(posListTran)
        with open(txtPath, 'w') as f:
            f.write("# index tx ty tz qx qy qz qw\n")
            for i in range(posLen):
                f.write("{}".format(i))
                for j in range(3):
                    f.write(" {}".format(posListTran[i][4 + j]))
                for j in range(3):
                    f.write(" {}".format(posListTran[i][1 + j]))
                f.write(" {}".format(posListTran[i][0]))
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
        btls.images_bin_write(binPath, posList, dstPath)
    else:
        print("have not test poses")

