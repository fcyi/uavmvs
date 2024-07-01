import numpy as np
import matplotlib.pyplot as plt
from read_write_model import *
import math

import sys
sys.path.append('tools')
import tools.utils as utl


def UniSampling(number):
    return np.linspace(0, 360, number, endpoint=True)


def LinerSampling(start, stop, number):
    return np.linspace(start, stop, number, endpoint=True)


# 获取偏航角对应的旋转矩阵（也就是绕着第2个坐标分量进行旋转，对应的旋转矩阵）
def get_yaw_rot(org_vec, targt_vec):
    # org_vec为参考向量，targt_vec为坐标与原点所连成的归一化后的单位向量
    # 相当于采集位置计算关于建筑中心的偏航角，首先偏航的转轴是垂直于坐标向量以及参考向量所构造的平面，其次由于targt_vec与org_vec皆为单位向量，
    # 若偏航角为the，cos(the) = c，sin(the) = s
    c = np.dot(org_vec, targt_vec)
    n_vector = np.cross(org_vec, targt_vec)  # s*n = n_vector，n为旋转向量
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
    R_w2c = I + n_vector_invert + np.dot(n_vector_invert, n_vector_invert) / (1 + c)
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
    tangen3_vec = np.array([tangen_vec[0], 0, tangen_vec[1]])
    tangen3_norm = np.linalg.norm(tangen3_vec)
    tangen3_vec = tangen3_vec / tangen3_norm
    R_w2c = get_yaw_rot(org_vec, tangen3_vec)
    R_w2c = np.dot(R_w2c, np.array([[-1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, 1]]))
    R_w2c = np.dot(transM_ZX, R_w2c)
    return R_w2c


def get_Tw2c(t, R):
    # t: 相机到世界坐标系的平移向量
    t = np.reshape(t, (3, 1))
    Rt = np.hstack((R, t))
    Rt = np.vstack((Rt, np.array([0, 0, 0, 1]).T))
    Rt = np.linalg.inv(Rt)
    return Rt[:3, 3]


def images_bin_write(binPath, posList):
    data = read_images_binary(binPath)
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

    write_images_binary(data, binPath)


if __name__ == '__main__':
    control_point = [-0.009859, 4.297023, 0.11011]

    coord = []

    posNum = 50
    angle_list = UniSampling(posNum)

    temp1 = angle_list[10:]
    temp2 = angle_list[:10]
    angle_list = np.concatenate((temp1,temp2))

    height_list = LinerSampling(-1.5, -1.5, posNum)  # 无人机环绕上升时，每一圈上每一个采集点处的高度
    r_list = LinerSampling(5, 5, posNum)  # 生成每一个位置所对应的半径，因为环绕的建筑的俯瞰形状很可能不是一个规则的圆

    # circle formula
    org = [0, 0]
    a = org[0]
    b = org[1]
    r = 3   # 计算俯仰角时所考虑的半径，此处俯仰角的调整是为了相机镜头始终朝着重建场景中心
    # angle = 30
    for angle in range(angle_list.shape[0]):
        x = a + r_list[angle] * np.cos(np.radians(angle_list[angle]))
        y = b + r_list[angle] * np.sin(np.radians(angle_list[angle]))
        coord.append([x, y])

    coord = np.asarray(coord)
    H = -3
    org_vec = np.array([0, 0, 1])  # 参考向量，指向成像场景中心

    posList = []
    for i in range(posNum):
        # 计算切向方向，此处假设坐标系原点为[0, 0, 0]
        # 计算切向方向时，不需要考虑高度这些采集位置都处于0高度的位置
        sta = np.arctan(height_list[i] / r)
        sta = -sta
        transM_X = get_pitch_rot(sta)
        transM_Z = get_roll_rot(np.pi)
        transM_ZX = np.dot(transM_Z, transM_X)

        R_w2c = get_Rw2c([coord[i][0], coord[i][1]], org_vec, transM_ZX)
        qv = rotmat2qvec(R_w2c)

        # 获取从世界坐标系到相机坐标系的变换矩阵（不理解为啥此处求完逆之后还要再一次求逆，第一次求逆有何作用）
        R = np.linalg.inv(R_w2c)
        t = [-coord[i][0]+control_point[0], height_list[i]+control_point[1], -coord[i][1]+control_point[2]]

        tv = get_Tw2c(t, R)
        # 将变换矩阵转为四元数+平移向量的形式，并存放到images.bin里面以进行查看生成的轨迹效果
        posList.append(qv.tolist()+tv.tolist())

    images_bin_write("/home/hongqingde/workspace_git/test/cdata_sparse/images.bin", posList)