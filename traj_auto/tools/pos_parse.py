import numpy as np


# 旋转矩阵转欧拉角，之后再对某种角度进行反方向的转动
# 此处的旋转矩阵所对应的欧拉分解为：翻滚角*俯仰角*偏航角
def parse_rot(Rot, axis=((0, 0), (1, 0), (2, 0))):
    # Rot = rollP * pitchP * yawP
    # Rtmp = np.array([[-1, 0, 0],
    #                  [0, -1, 0],
    #                  [0, 0, 1]])
    # Rot_ = np.dot(Rot, Rtmp)
    # rollRot = [cr, sr, 0; -sr, cr, 0; 0, 0, 1;]
    # pichRot = [1, 0, 0; 0, cp, sp; 0, -sp, cp;]
    # yawRot = [cy, 0, sy; 0, 1, 0; sy, 0, cy;]
    # Rot = rollRot * pichRot * yawRot
    #     = [cr+cy+sr*sp*sy, sr*cp, -cr*sy+sr*sp*cy; -sr*cy+cr*sp*sy, cr*cp, sr*sy+cr*sp*cy; cp*sy, -sp, cp*cy;]
    s1 = -Rot[2][1]

    rollA = 0  # 翻滚角处于[0, pi]
    yawA = 0  # 偏航角处于[0, pi]
    pitchA = 0
    if np.isclose(s1, 1) or s1 >= 1.:
        pitchA = np.pi/2
    elif np.isclose(s1, -1) or s1 <= -1.:
        pitchA = -np.pi/2
    else:
        pitchA = np.arcsin(s1)  # 俯仰角处于(-pi/2, pi/2)
    cp = np.cos(pitchA)

    if not np.isclose(pitchA, np.pi/2) or not np.isclose(pitchA, -np.pi/2):  # np.isclose(cp, 0)的约束没有np.isclose(pitchA, np.pi/2) or np.isclose(pitchA, -np.pi/2)强
        sr = Rot[0][1]/cp
        cr = Rot[1][1]/cp
        sy = Rot[2][0]/cp
        cy = Rot[2][2]/cp
        if cr > 1:
            cr = 1.0
        elif cr < -1:
            cr = -1.
        if sr > 1:
            sr = 1.0
        elif sr < -1:
            sr = -1.
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
            yawA = 2*np.pi - yawA
        rollA = np.arccos(cr)
        if sr < 0:
            rollA = 2*np.pi - rollA
    else:
        # 注意，此处的解法是基于已知的roll == np.pi的情况，若roll有修改则必须进行相应的改动，
        # 若roll未知，则是存在歧义解，只能得到rollSyaw(pitchA = np.pi/2)或rollPyaw(pitchA == -np.pi/2)的结果，这也是欧拉分解的缺陷
        # https://blog.csdn.net/YanDabbs/article/details/135629614?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-12-135629614-blog-138961638.235^v43^pc_blog_bottom_relevance_base5&spm=1001.2101.3001.4242.7&utm_relevant_index=13
        # rollSyaw = np.arctan(Rot[0][2]/Rot[0][0])  # roll - yaw
        rollA = np.pi
        yawA = np.arccos(-Rot[0][0])

    for axe in axis:
        if axe[1] == 0:
            pass
        elif axe[1] == -1:
            if axe[0] == 0:
                rollA = -rollA
            elif axe[0] == 1:
                pitchA = -pitchA
            elif axe[0] == 2:
                yawA = -yawA
        else:
            arc = np.radians(axe[1])
            if axe[0] == 0:
                rollA += arc
            elif axe[0] == 1:
                pitchA += arc
            elif axe[0] == 2:
                yawA += arc
    rollP = np.array([[np.cos(rollA), np.sin(rollA), 0],
                      [-np.sin(rollA), np.cos(rollA), 0],
                      [0, 0, 1]])
    pitchP = np.array([[1, 0, 0],
                       [0, np.cos(pitchA), np.sin(pitchA)],
                       [0, -np.sin(pitchA), np.cos(pitchA)]])
    yawP = np.array([[np.cos(yawA), 0, -np.sin(yawA)],
                     [0, 1, 0],
                     [np.sin(yawA), 0, np.cos(yawA)]])

    Rott = np.dot(rollP, np.dot(pitchP, yawP))
    return Rott


def change_rot(srcRot, axis=((0, 0), (1, 0), (2, 0))):
    rottmp = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])

    rot = np.dot(srcRot, rottmp)
    resRot = parse_rot(rot, axis)
    resRot = np.dot(resRot, rottmp)
    return resRot


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    # reference: 《New Method for Extracting the Quaternion from a Rotation Matrix》
    # return: [q0, q1, q2, q3] == [qw, qx, qy, qz]
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )

    # 对K进行特征值、特征向量计算，基于最大特征值对应的特征向量计算四元数
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def rotvec2rotmat(n_vector, c):
    # n_vector, 旋转向量的单位向量与转角的正弦量的乘积
    # c, 旋转向量对应的转角的余弦
    s = np.linalg.norm(n_vector)
    # 为了将向量之间的叉积转为矩阵运算
    # n_vector x a =  n_vector_invert * a
    n_vector_invert = np.array((
        [0, -n_vector[2], n_vector[1]],
        [n_vector[2], 0, -n_vector[0]],
        [-n_vector[1], n_vector[0], 0]
    ))
    I = np.eye(3)

    # 罗德里格斯公式
    # nv = n_vector = sin(the)*n = s*n, nvi = n_vector_invert = n_vector^，ni = n^，c = cos(the)
    # n^T * n = 1
    # R_w2c = I + nvi + nvi*nvi/(1+c)
    # (1+c)*R_w2c = (1+c)*I + (1+c)*nvi + nvi*nvi
    #             = (1+c)*I + (1+c)*s*ni + s*s*ni*ni
    #             = (1+c)*I + (1+c)*s*ni + s*s*(n*n^T-I)
    #             = (1+c)*I + (1+c)*s*ni + (1-c*c)*(n*n^T-I)
    #             = (1+c)*I + (1+c)*s*ni + (1+c)*(1-c)*(n*n^T-I)
    #             = (1+c)*c*I + (1+c)*s*ni + (1+c)*(1-c)*(n*n^T)

    if c != -1:
        R_w2c = I + n_vector_invert + np.dot(n_vector_invert, n_vector_invert) / (1 + c)
    else:
        if not np.isclose(s, 0):
            nvn = np.array([n_vector / s])
            R_w2c = -I + 2 * np.dot(nvn.transpose(), nvn)
        else:
            # 旋转180度引发的共线情况，这时候旋转向量需要重新求解，否则得到的旋转矩阵有问题
            # 在这种情况下，默认旋转向量为为[0, 1, 0]
            # 注意，np.cross(a, b) = a x b
            # tv = rotate_coordinate([targt_vec[0], targt_vec[2]], 0, 0, 10)
            # nv_ = np.cross(org_vec, [tv[0], 0, tv[1]])
            # s_ = np.linalg.norm(nv_)
            # nvn = np.array([nv_ / s_])
            nvn = np.array([[0., 1., 0.]])
            R_w2c = -I + 2 * np.dot(nvn.transpose(), nvn)

    return R_w2c


def rotmat2rotvec(rotMat):
    tr = rotMat[0][0] + rotMat[1][1] + rotMat[2][2]
    costheta = (tr - 1.0) * 0.5
    if costheta > 1:
        costheta = 1

    if costheta < -1:
        costheta = -1

    theta = np.arccos(costheta)
    s = np.sin(theta)
    factor = 1. if (np.abs(s) < 1e-5) else (theta/s)
    factor *= 0.5
    w = [(rotMat[2][1] - rotMat[1][2])*factor, (rotMat[0][2] - rotMat[2][0])*factor, (rotMat[1][0] - rotMat[0][1])*factor]
    return w
