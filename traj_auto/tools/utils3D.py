import math
import numpy as np
from scipy.spatial.transform import Rotation


def line_traj_xyz(psxyz, pexyz, residual, step, refineStepRatio=1):
    """
    生成psxyz与pexyz之间的直线型轨迹的xyz，并且产生下一段路线上的距离残差
    :param psxyz: 起始点
    :param pexyz: 终止点
    :param residual:上一段路线中没走完的长度，这一长度会累积在整段轨迹中
    :param step: 步长
    :param refineStepRatio:细化步长与普通步长之间的比例，一般用于有拐角的地方，设为1时细化步长不会起作用
    :return:
    """
    assert (psxyz[0] != pexyz[0]) or (psxyz[1] != pexyz[1]) or (psxyz[2] != pexyz[2]), "完全相同的起始终止点，产生不了直线路线"

    dxyz = [pexyz[i] - psxyz[i] for i in range(3)]
    disxyz = math.sqrt(dxyz[0] ** 2 + dxyz[1] ** 2 + dxyz[2] ** 2)

    accumStart = residual
    accumEnd = disxyz
    accumV, residualArc = get_accumList(accumStart, accumEnd, step, refineStepRatio)

    accumLen = len(accumV)
    trajTmp = [[accumV[i], 0, 0] for i in range(accumLen)]

    traj = []
    if (dxyz[1] == 0) and (dxyz[2] == 0):
        mulfac = -1 if psxyz[0] > pexyz[0] else 1
        traj = (mulfac*np.array(trajTmp)).tolist()
    else:
        dxyz = np.array(dxyz)
        dxyzTmp = np.array([disxyz, 0, 0])
        # 计算转轴、转角以及旋转矩阵
        rotationAxis, rotationAngle = get_rotation_axis_angle(dxyzTmp, dxyz, 1)
        assert rotationAxis is not None, "计算旋转向量时，存在共线问题"
        rotationMat = Rotation.from_rotvec(rotationAxis * rotationAngle).as_matrix()
        trajTmpArray = np.array(trajTmp)
        trajTmpTransArray = np.matmul(rotationMat, trajTmpArray.transpose())
        traj = trajTmpTransArray.transpose().tolist()
    for i in range(accumLen):
        traj[i][0] += psxyz[0]
        traj[i][1] += psxyz[1]
        traj[i][2] += psxyz[2]

    return traj, residualArc


def circular_traj(psxyz, pmxyz, pexyz, residual, step, refineStepRatio=1):
    # 圆弧型轨迹必须提供3个点，否则圆弧方向是不能确定的，若提供的点数超过3个，那么规整的圆形轨迹是比较难拟合出来的
    # 需要注意的是，这3个点要依次给出
    pcxyz = get_circular_center_base_3point3D(psxyz, pmxyz, pexyz)

    # Step 1: 计算单位向量
    psxyzA, pmxyzA, pexyzA, pcxyzA = np.array(psxyz), np.array(pmxyz), np.array(pexyz), np.array(pcxyz)
    vec_sc = psxyzA - pcxyzA
    vec_ec = pexyzA - pcxyzA
    vec_mc = pmxyzA - pcxyzA
    pradius = np.linalg.norm(vec_sc)

    # 计算转轴、转角、弧长
    # 注意此处通过叉积计算转轴和转角时有问题的，必须要根据3个点的次序进行处理，不能直接使用
    vec_ms = pmxyzA - psxyzA
    vec_es = pexyzA - psxyzA

    rotationAxis, _ = get_rotation_axis_angle(vec_ms, vec_es, 0)
    rotationAxisMEC, rotationAngleMEC = get_rotation_axis_angle(vec_mc, vec_ec, 1)
    rotationAxisSMC, rotationAngleSMC = get_rotation_axis_angle(vec_sc, vec_mc, 1)

    rotAngleSM = handle_rot_angle_based_rot_axis(rotationAxis, rotationAxisSMC, rotationAngleSMC)
    rotAngleME = handle_rot_angle_based_rot_axis(rotationAxis, rotationAxisMEC, rotationAngleMEC)
    assert (rotAngleSM != -999) and (rotAngleME != -999), "转轴求取有问题，不平行"
    rotationAngle = rotAngleSM + rotAngleME
    seArc = rotationAngle * pradius

    # 根据弧长计算弧长增量，再根据弧长增量计算角度增量，再根据角度增量，计算旋转矩阵，再根据旋转矩阵获取位姿点的xyz
    traj = []
    residualArc = 0
    if residual > seArc:
        residualArc = residual - seArc
    elif residual == seArc:
        traj.append(pexyz)
    else:
        arcTmp = residual
        # 获取弧长累积量
        arcV, residualArc = get_accumList(arcTmp, seArc, step, refineStepRatio)

        # 计算位姿
        for arcK in arcV:
            if arcK == 0:
                traj.append(psxyz)
            else:
                rotAngle = arcK / pradius
                rotMat = Rotation.from_rotvec(rotationAxis * rotAngle).as_matrix()
                trajTmpTransArray = np.matmul(rotMat, vec_sc)
                trajTmpTransArray = trajTmpTransArray + pcxyzA
                traj.append(trajTmpTransArray.tolist())

    # traj.append(pcxyz)

    return traj, residualArc


def get_circular_center_base_3point3D(pd1, pd2, pd3):
    # 这是根据3个点确定圆心的函数，请给3个点，不要多也不要少
    assert (((pd3[0] != pd1[0]) or (pd3[1] != pd1[1]) or (pd3[2] != pd1[2])) and
            ((pd3[0] != pd2[0]) or (pd3[1] != pd2[1]) or (pd3[2] != pd2[2])) and
            ((pd2[0] != pd1[0]) or (pd2[1] != pd1[1]) or (pd2[2] != pd1[2]))), "这三个点中存在相同的点"
    pdA3, pdA1, pdA2 = np.array(pd3), np.array(pd1), np.array(pd2)
    vec_12 = pdA2 - pdA1
    vec_13 = pdA3 - pdA1
    assert np.linalg.norm(np.cross(vec_12, vec_13)) != 0, "共线的三个点形成不了三角形，基于这三点是不能给确定相应的外接圆的圆心"

    # 基于线性代数的知识进行求解，
    # 参考链接：https://blog.csdn.net/webzhuce/article/details/88371649?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-8-88371649-blog-99209020.235^v43^pc_blog_bottom_relevance_base1&spm=1001.2101.3001.4242.5&utm_relevant_index=9
    x1, y1, z1 = pd1[0], pd1[1], pd1[2]
    x2, y2, z2 = pd2[0], pd2[1], pd2[2]
    x3, y3, z3 = pd3[0], pd3[1], pd3[2]
    a1 = (y1 * z2 - y2 * z1 - y1 * z3 + y3 * z1 + y2 * z3 - y3 * z2)
    b1 = -(x1 * z2 - x2 * z1 - x1 * z3 + x3 * z1 + x2 * z3 - x3 * z2)
    c1 = (x1 * y2 - x2 * y1 - x1 * y3 + x3 * y1 + x2 * y3 - x3 * y2)
    d1 = -(x1 * y2 * z3 - x1 * y3 * z2 - x2 * y1 * z3 + x2 * y3 * z1 + x3 * y1 * z2 - x3 * y2 * z1)

    a2 = 2 * (x2 - x1)
    b2 = 2 * (y2 - y1)
    c2 = 2 * (z2 - z1)
    d2 = x1 * x1 + y1 * y1 + z1 * z1 - x2 * x2 - y2 * y2 - z2 * z2

    a3 = 2 * (x3 - x1)
    b3 = 2 * (y3 - y1)
    c3 = 2 * (z3 - z1)
    d3 = x1 * x1 + y1 * y1 + z1 * z1 - x3 * x3 - y3 * y3 - z3 * z3

    centerpoint = [0, 0, 0]
    fac = 1. / (a1 * b2 * c3 - a1 * b3 * c2 - a2 * b1 * c3 + a2 * b3 * c1 + a3 * b1 * c2 - a3 * b2 * c1)
    centerpoint[0] = -(b1 * c2 * d3 - b1 * c3 * d2 - b2 * c1 * d3 + b2 * c3 * d1 + b3 * c1 * d2 - b3 * c2 * d1) * fac
    centerpoint[1] = (a1 * c2 * d3 - a1 * c3 * d2 - a2 * c1 * d3 + a2 * c3 * d1 + a3 * c1 * d2 - a3 * c2 * d1) * fac
    centerpoint[2] = -(a1 * b2 * d3 - a1 * b3 * d2 - a2 * b1 * d3 + a2 * b3 * d1 + a3 * b1 * d2 - a3 * b2 * d1) * fac

    return centerpoint


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
    return accumV, residualArc


def handle_rot_angle_based_rot_axis(rotAxis0, rotAxis1, rotAngle):
    if rotAxis1 is not None:
        dotProd = np.dot(rotAxis0, rotAxis1)
        angle = np.arccos(dotProd)
        rotAgl = -999
        if np.isclose(angle, 0):
            rotAgl = rotAngle
        elif np.isclose(angle, np.pi):
            rotAgl = 2 * np.pi - rotAngle
    else:
        rotAgl = rotAngle
    return rotAgl


def get_rotation_axis_angle(rvec1, rvec2, retAgl=1):
    axCross = np.cross(rvec1, rvec2)
    axCrossNorm = np.linalg.norm(axCross)
    if np.isclose(axCrossNorm, 0):
        rotAxis = None
        if retAgl == 1:
            dotProd = np.dot(rvec1, rvec2)
            rotAngle = np.arccos(dotProd / (np.linalg.norm(rvec1) * np.linalg.norm(rvec2)))
        else:
            rotAngle = -999
    else:
        rotAxis = axCross / axCrossNorm
        if retAgl == 1:
            rotAngle = np.arccos(np.dot(rvec1, rvec2) / (np.linalg.norm(rvec1) * np.linalg.norm(rvec2)))
        else:
            rotAngle = -999
    return rotAxis, rotAngle