import math
import numpy as np
from scipy.spatial import KDTree


def get_intrinsic_params(resoluation_, fov_, camModel_='SIMPLE_PINHOLE'):
    """
    在airsim中水平方向的fov与垂直方向的fov是一致的，在设置文件settings.json中调整的fov其实也就是这个参数
    resoluation_: 图像的宽度和高度（单位为像素）
    fov_: 视场角（单位为度）
    return: [fx_, fy_, cx_, cy_, width_, height_], 相机中心，水平和垂直方向的焦距（中心、焦距的单位由resoluation决定）
    """
    width_, height_ = resoluation_[:]
    cx_ = width_ / 2.
    cy_ = height_ / 2.
    if camModel_ == 'SIMPLE_PINHOLE':
        fx_ = width_ / 2. / math.tan(math.radians(fov_ / 2.))
        fy_ = width_ / 2. / math.tan(math.radians(fov_ / 2.))  # 注意，此处不是height_ / 2. / math.tan(math.radians(fov_ / 2.))，因为使用的是simple_pinhole相机模型
        return [camModel_, width_, height_, fx_, cx_, cy_]
    elif camModel_ == 'PINHOLE':
        fx_ = width_ / 2. / math.tan(math.radians(fov_ / 2.))
        fy_ = height_ / 2. / math.tan(math.radians(fov_ / 2.))  # 注意，此处不是height_ / 2. / math.tan(math.radians(fov_ / 2.))，因为使用的是simple_pinhole相机模型
        return [camModel_, width_, height_, fx_, fy_, cx_, cy_]
    else:
        raise Exception


def qvec_norm(qvec_):
    qvecNorm_ = np.linalg.norm(qvec_)
    qvecN_ = np.copy(qvec_)
    if abs(qvecNorm_ - 1.0) > 1e-6:
        print(f"Warning: Quaternion {qvec_} is not normalized.")
        qvecN_ = qvecN_ / qvecNorm_
    return qvecN_


# Changes the sign of the quaternion components. This is not the same as the inverse.
def qvec_inverseSign(qvec_):
    return -qvec_


# 通过点积判断两个四元数之间是否相似
# Returns true if the two input quaternions are close to each other. This can
# be used to check whether or not one of two quaternions which are supposed to
# be very similar but has its component signs reversed (q has the same rotation as -q)
def qvec_areClose(qvec0_, qvec1_, epsilon=1e-6):
    qvec0N_ = qvec_norm(qvec0_)
    qvec1N_ = qvec_norm(qvec1_)

    qvecDot_ = np.dot(qvec0N_, qvec1N_)

    if np.isclose(abs(qvecDot_), 1.-epsilon):
        return True
    else:
        return False


# 通过距离判断两个四元数之间是否相似
def qvec_areCloseSim(qvec0_, qvec1_, thres_=1e-3):
    return np.linalg.norm(qvec0_, qvec1_) < thres_


# 判断给定的四元数是否接近
def qvecs_areClose(qvecs_, useSpatialSort=False, epsilon_=1e-6):
    qvecsNum_ = qvecs_.shape[0]
    resFlg_ = True
    if not useSpatialSort:
        for i_ in range(qvecsNum_):
            for j_ in range(i_ + 1, qvecsNum_):
                if not qvec_areClose(qvecs_[i_], qvecs_[j_], epsilon_):
                    resFlg_ = False
                    break
            if not resFlg_:
                break

    else:
        # !!! bug: 此处的空间排序有问题，因为没有考虑kd树的距离度量方式和四元数的距离度量方式（基于向量之间的欧几里得距离）是否一致，不一致的距离度量方式不应该使用同一个阈值
        # 通过空间排序尝试对暴力遍历进行加速
        # 构建 k-d 树
        tree_ = KDTree(qvecs_)

        # 遍历每个四元数，查找接近的四元数
        for index_, qvec_ in enumerate(qvecs_):
            # 查找在阈值范围内的所有点
            indices_ = tree_.query_ball_point(qvec_, epsilon_)

            # 不包括自身，检查接近的四元数
            for i_ in indices_:
                if i_ != index_ and not qvec_areCloseSim(qvec_, qvecs_[i_], epsilon_):
                    resFlg_ = False
                    break
            if not resFlg_:
                break

    return resFlg_


def qvec_isNeedInverseSign(qvec0_, qvec1_):
    qvecDot_ = np.dot(qvec0_, qvec1_)
    return True if qvecDot_ < 0. else False


def qvec2rotmat(qvec_):
    return np.array(
        [
            [
                1 - 2 * qvec_[2] ** 2 - 2 * qvec_[3] ** 2,
                2 * qvec_[1] * qvec_[2] - 2 * qvec_[0] * qvec_[3],
                2 * qvec_[3] * qvec_[1] + 2 * qvec_[0] * qvec_[2],
            ],
            [
                2 * qvec_[1] * qvec_[2] + 2 * qvec_[0] * qvec_[3],
                1 - 2 * qvec_[1] ** 2 - 2 * qvec_[3] ** 2,
                2 * qvec_[2] * qvec_[3] - 2 * qvec_[0] * qvec_[1],
            ],
            [
                2 * qvec_[3] * qvec_[1] - 2 * qvec_[0] * qvec_[2],
                2 * qvec_[2] * qvec_[3] + 2 * qvec_[0] * qvec_[1],
                1 - 2 * qvec_[1] ** 2 - 2 * qvec_[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R_):
    # reference: 《New Method for Extracting the Quaternion from a Rotation Matrix》
    # return: [q0, q1, q2, q3] == [qw, qx, qy, qz]
    Rxx_, Ryx_, Rzx_, Rxy_, Ryy_, Rzy_, Rxz_, Ryz_, Rzz_ = R_.flat
    K_ = (
        np.array(
            [
                [Rxx_ - Ryy_ - Rzz_, 0, 0, 0],
                [Ryx_ + Rxy_, Ryy_ - Rxx_ - Rzz_, 0, 0],
                [Rzx_ + Rxz_, Rzy_ + Ryz_, Rzz_ - Rxx_ - Ryy_, 0],
                [Ryz_ - Rzy_, Rzx_ - Rxz_, Rxy_ - Ryx_, Rxx_ + Ryy_ + Rzz_],
            ]
        )
        / 3.0
    )
    # 对K进行特征值、特征向量计算，基于最大特征值对应的特征向量计算四元数
    eigvals_, eigvecs_ = np.linalg.eigh(K_)
    qvec_ = eigvecs_[[3, 0, 1, 2], np.argmax(eigvals_)]
    if qvec_[0] < 0:
        qvec_ *= -1
    return qvec_


# 位姿求逆
def pose_inverse(pose_):
    qvec_ = np.copy(pose_[:4])
    qvec_ = qvec_norm(qvec_)
    qvec_ = qvec_.tolist()
    RTmp_ = qvec2rotmat(qvec_[:])
    TTmp_ = np.eye(4, 4)
    TTmp_[:3, :3] = RTmp_
    TTmp_[:3, 3] = np.copy(pose_[4:])
    TInvTmp_ = np.linalg.inv(TTmp_)
    RInvTmp_ = TInvTmp_[:3, :3]
    tInvTmp_ = TInvTmp_[:3, 3]
    qvecInv_ = rotmat2qvec(RInvTmp_)
    qvecInv_ = qvec_norm(qvecInv_)
    poseNew_ = qvecInv_.tolist() + tInvTmp_.tolist()
    return poseNew_


# 位姿与变换矩阵的互转
def pose7_2_poseM(poseSrc_):
    poseSrcTmp_ = None
    if type(poseSrc_).__name__ == 'list':
        if len(poseSrc_) == 7:
            poseSrcTmp_ = poseSrc_
        elif len(poseSrc_) == 3:
            poseSrcTmp_ = np.array(poseSrc_)
        else:
            raise Exception
    elif type(poseSrc_).__name__ == 'ndarray':
        if len(poseSrc_.shape) == 2:
            poseSrcTmp_ = poseSrc_
        elif len(poseSrc_.shape) == 1:
            poseSrcTmp_ = poseSrc_.tolist()
        else:
            raise Exception
    else:
        raise Exception

    if type(poseSrcTmp_).__name__ == 'list':
        qvec_ = np.copy(poseSrcTmp_[:4])
        qvec_ = qvec_norm(qvec_)
        qvec_ = qvec_.tolist()
        RTmp_ = qvec2rotmat(qvec_[:])
        poseRes_ = np.eye(4, 4)
        poseRes_[:3, :3] = RTmp_
        poseRes_[:3, 3] = np.copy(poseSrcTmp_[4:])
    else:
        RTmp_ = poseSrcTmp_[:3, :3]
        tTmp_ = poseSrcTmp_[:3, 3]
        qvec_ = rotmat2qvec(RTmp_)
        qvec_ = qvec_norm(qvec_)
        poseRes_ = qvec_.tolist() + tTmp_.tolist()
    return poseRes_


# 像素坐标转相机坐标
def depth2xyz(depthMap_, K_, flatten_=False, depthScale_=1.0):
    fx_, fy_ = K_[0, 0], K_[1, 1]
    cx_, cy_ = K_[0, 2], K_[1, 2]
    h_, w_ = np.mgrid[0:depthMap_.shape[0], 0:depthMap_.shape[1]]
    z_ = depthMap_ / depthScale_
    x_ = (w_ - cx_) * z_ / fx_
    y_ = (h_ - cy_) * z_ / fy_
    xyz_ = np.dstack((x_, y_, z_)) if not flatten_ else np.dstack((x_, y_, z_)).reshape(-1, 3)
    # xyz=cv2.rgbd.depthTo3d(depth_map,depth_cam_matrix)
    return xyz_


# 相机坐标转世界坐标
def pc_cam_to_pc_world(pc_, extrinsic_):
    """
        pc          相机坐标系下的一个点云 1, 3
        extrinsic   相机位姿，Tcw世界到相机 (4, 4)
    """
    extrInv_ = np.linalg.inv(extrinsic_)  # Twc
    R_ = extrInv_[:3, :3]
    T_ = extrInv_[:3, 3]
    pc_ = (R_ @ pc_.T).T + T_   # Rwc * Pc + Twc = Pw
    return pc_


# 通过球面插值实现求解两个四元数之间的平均插值结果
def slerp_interpolate(qvec0_, qvec1_, t_):
    qvec0Name_ = type(qvec0_).__name__
    qvec1Name_ = type(qvec0_).__name__
    assert qvec1Name_ == qvec1Name_ and ((qvec0Name_ == 'list') or (qvec0Name_ == 'ndarray'))

    if qvec1Name_ == 'list':
        qvec0Tmp_ = np.array(qvec0_)
        qvec1Tmp_ = np.array(qvec1_)
    else:
        qvec0Tmp_ = np.copy(qvec0_)
        qvec1Tmp_ = np.copy(qvec1_)

    cosa_ = np.dot(qvec0Tmp_[:4], qvec1Tmp_[:4])
    # If the dot product is negative, the quaternions have opposite handed-ness and slerp won't take
    # the shorter path. Fix by reversing one quaternion.
    if cosa_ < 0.:
        qvec1Tmp_ = -1 * qvec1Tmp_
        cosa_ = -cosa_

    # If the inputs are too close for comfort, linearly interpolate
    if cosa_ > 0.9995:
        k0_ = 1. - t_
        k1_ = t_
    elif np.isclose(cosa_, 0.):
        a_ = np.pi / 2.
        k0_ = np.sin((1. - t_) * a_)
        k1_ = np.sin(t_ * a_)
    else:
        sina_ = np.sqrt(1. - cosa_*cosa_)
        a_ = np.arctan2(sina_, cosa_)
        k0_ = np.sin((1.-t_)*a_) / sina_
        k1_ = np.sin(t_*a_) / sina_

    qvecRes_ = k0_ * qvec0Tmp_ + k1_ * qvec1Tmp_
    return qvecRes_


# 四元数平滑加速版
# reference: https://www.cnblogs.com/21207-iHome/p/6952004.html
# Get an average (mean) from more then two quaternions (with two, slerp would be used).
# Note: this only works if all the quaternions are relatively close together.
# Usage:
# -Cumulative is an external Vector4 which holds all the added x y z and w components.
# -newRotation is the next rotation to be added to the average pool
# -firstRotation is the first quaternion of the array to be averaged
# -addAmount holds the total amount of quaternions which are currently added
# This function returns the current average quaternion
def qvec_average_accum(cum_, qvecFirst_, qvecNew_, addDet_):
    # Before we add the new rotation to the average (mean), we have to check whether the quaternion has to be inverted. Because
    # q and -q are the same rotation, but cannot be averaged, we have to make sure they are all the same.
    if qvec_isNeedInverseSign(qvecNew_, qvecFirst_):
        qvecNewTmp_ = qvec_inverseSign(qvecNew_)
    else:
        qvecNewTmp_ = np.copy(qvecNew_)

    # Average the values
    # !!! bug这是原版的实现方式，个人认为有问题，因为显然可以发现越后面的四元数占的权重越大
    # addDet_ = 1. / float(addAmount_)
    # cumTmp_ = cum_ + qvecNewTmp_
    # qvecAvgTmp_ = addDet_ * cumTmp_
    # fix the bug
    qvecAvgTmp_ = cum_ + addDet_ * qvecNewTmp_

    # note: if speed is an issue, you can skip the normalization step
    qvecAvgTmpN_ = qvec_norm(qvecAvgTmp_)
    return qvecAvgTmpN_


# 四元数平滑
def qvec_avg_markley(qvecs_, weights_):
    # Q is an Nx4 matrix of quaternions
    # weights is an Nx1 vector, a weight for each quaternion.
    # qvecsAvg_ avg the weightedaverage quaternion
    assert type(qvecs_).__name__ == 'ndarray' and type(weights_).__name__ == 'ndarray'
    assert qvecs_.shape[0] == weights_.shape[0] and len(qvecs_.shape) == 2 and len(weights_.shape) == 1

    if not qvecs_areClose(qvecs_):
        # Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman. "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30, no. 4 (2007): 1193-1197.

        weightsDiag_ = np.diag(weights_)
        # M_ = weights_[0]*(qvecs_[0].T@qvecs_[0]) + ... + weights_[i]*(qvecs_[i].T@qvecs_[i])
        M_ = qvecs_.T @ weightsDiag_ @ qvecs_
        wSum_ = np.sum(weights_)

        # Scale
        M_ = M_ / float(wSum_)

        # The average quaternion is the eigenvector of M corresponding to the maximum eigenvalue.
        # Get the eigenvector corresponding to the largest eigenvalue
        # 计算特征值和特征向量
        eigenvalues_, eigenvectors_ = np.linalg.eig(M_)

        # 找到最大特征值的索引
        maxIndex_ = np.argmax(eigenvalues_)

        # 获取对应的特征向量
        qvecsAvg_ = eigenvectors_[:, maxIndex_]
    else:
        qvecFirst_ = qvecs_[0]
        qvecsNum_ = qvecs_.shape[0]
        qvecsAvg_ = np.zeros((4,))
        for i_ in range(qvecsNum_):
            qvecsAvg_ = qvec_average_accum(qvecsAvg_, qvecFirst_, qvecs_[i_], weights_[i_])

    return qvecsAvg_  # Return as a 1D array





