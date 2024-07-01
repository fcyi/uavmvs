import numpy as np
import matplotlib.pyplot as plt
from read_write_model import *
import math
# import open3d as o3d
# import pyransac3d as pyrsc


def UniSampling(number):
    return np.linspace(0, 360, number, endpoint=True)


def LinerSampling(start, stop, number):
    return np.linspace(start, stop, number, endpoint=True)


control_point = [-0.009859, 1.297023, 0.11011]

data = read_images_binary("/home/hongqingde/workspace_git/test/cdata_sparse/images.bin")
coord = []

posNum = 50
angle_list = UniSampling(posNum)

temp1 = angle_list[10:]
temp2 = angle_list[:10]
angle_list = np.concatenate((temp1,temp2))


height_list = LinerSampling(-1.5, -1.5, posNum)  # 无人机环绕上升时，每一圈上每一个采集点处的高度
r_list = LinerSampling(2.5,2.5, posNum)  # 生成每一个位置所对应的半径，因为环绕的建筑的俯瞰形状很可能不是一个规则的圆，因此
# r_list = np.ones(500)*3.0
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
# X = coord[:,0]
# Y = coord[:,1]
# plt.plot(X,Y,"o")
# plt.show()
H = -3
org_vec = np.array([0, 0, 1])  # 参考向量，指向成像场景中心

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

for i in range(posNum):
    # print(-coord[i][0], H, -coord[i][1])
    # 计算切向方向，此处假设坐标系原点为[0, 0, 0]
    # 计算切向方向时，不需要考虑高度这些采集位置都处于0高度的位置
    targt_vec = np.array([coord[i][0], 0, coord[i][1]])  # 此处之所以使用[x, z, y]形式的坐标，是为了使用左手系，之所以令z分量为0，是为了让坐标向量与参考向量
    norm = np.linalg.norm(targt_vec)
    targt_vec = targt_vec/norm

    # 相当于采集位置计算关于建筑中心的偏航角，首先偏航的转轴是垂直于坐标向量以及参考向量所构造的平面，其次由于targt_vec与org_vec皆为单位向量，
    # 若偏航角为the，cos(the) = c，sin(the) = s
    c = np.dot(org_vec, targt_vec)
    n_vector = np.cross(org_vec, targt_vec)  # s*n = n_vector，n为旋转向量
    s = np.linalg.norm(n_vector)
    # print(c, s)

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
    R_w2c = I + n_vector_invert + np.dot(n_vector_invert, n_vector_invert)/(1+c)
    R_w2c = np.dot(R_w2c, np.array([[-1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0,  1]]))
    # R_w2c = np.dot(np.array([[-1, 0, 0],
    #                          [0, -1, 0],
    #                          [0,  0, 1]]), R_w2c)
    # 绕着第一个坐标基逆时针旋转sta度，sta即为俯仰角
    sta = np.arctan(height_list[i]/r)
    sta = -sta
    transM_X = np.array([[1, 0, 0],
                         [0, np.cos(sta), np.sin(sta)],
                         [0, -np.sin(sta), np.cos(sta)]])
    R_w2c = np.dot(transM_X, R_w2c)
    # 绕着第三个坐标基顺时针旋转180度，也就是说翻滚角为180度
    transM_Z = np.array([[np.cos(np.pi), np.sin(np.pi), 0],
                         [-np.sin(np.pi), np.cos(np.pi), 0],
                         [0, 0, 1]])
    R_w2c = np.dot(transM_Z, R_w2c)
    # print(org_vec)
    # print(np.matmul(org_vec, R_w2c))
    # print(targt_vec)
    # Rt_c2w = np.linalg.inv(R_w2c)

    # 获取从世界坐标系到相机坐标系的变换矩阵（不理解为啥此处求完逆之后还要再一次求逆，第一次求逆有何作用）
    R = np.linalg.inv(R_w2c)
    t = [-coord[i][0]+control_point[0], height_list[i]+control_point[1], -coord[i][1]+control_point[2]]
    # t = [-coord[i][0], height_list[i], -coord[i][1]]
    t = np.reshape(t, (3, 1))
    Rt = np.hstack((R, t))
    Rt = np.vstack((Rt, np.array([0, 0, 0, 1]).T))
    Rt = np.linalg.inv(Rt)

    # sta = np.arctan(H/r)
    # transM = np.array([[1,0,0],
    #                    [0,np.cos(sta),np.sin(sta)],
    #                   [0,-np.sin(sta),np.cos(sta)]])
    # R_w2c = np.dot(transM,R_w2c)

    # 将变换矩阵转为四元数+平移向量的形式，并存放到images.bin里面以进行查看生成的轨迹效果
    qv = rotmat2qvec(R_w2c)
    tv = Rt[:3, 3]
    image1 = data[firstKey]
    image1 = image1._replace(id=maxKeyId+i,qvec=qv,tvec=tv)
    data[maxKeyId+i] = image1


# normal_vec = np.asarray(normal_vec)
# X = normal_vec[:,0]
# Y = normal_vec[:,2]
# plt.plot(X,Y,"o")
# plt.show()

# target1 = np.array([1,0,0])
# norm = np.linalg.norm(target1)
# targt_vec = target1/norm
# c = np.dot(targt_vec,org_vec)
# n_vector = np.cross(targt_vec,org_vec)
# s = np.linalg.norm(n_vector)
# #print(c, s)
#
# n_vector_invert = np.array((
#     [0,-n_vector[2],n_vector[1]],
#     [n_vector[2],0,-n_vector[0]],
#     [-n_vector[1],n_vector[0],0]
#     ))
# I = np.eye(3)
# # 核心公式：见上图
# R_w2c = I + n_vector_invert + np.dot(n_vector_invert, n_vector_invert)/(1+c)

#R_w2c = np.linalg.inv(R_w2c)
# qv = rotmat2qvec(R_w2c)
# tv = np.array([0,0,0])
# image1 = data[2945]
# data[2945] = image1._replace(qvec=qv)
# data[2945] = image1._replace(tvec=tv)

# R = np.array([[1,0,0],
#               [0,1,0],
#               [0,0,1]])
#
# qv = rotmat2qvec(R)
# tv = np.array([0,0,0])
# image1 = data[2945]
# image1 = image1._replace(qvec=qv)
# image1 = image1._replace(tvec=tv)
# data[2945] = image1
#
# qv = rotmat2qvec(R_w2c)
# tv = np.array([0,0,0])
# image2 = data[2946]
# image2 = image2._replace(qvec=qv)
# image2 = image2._replace(tvec=tv)
# data[2946] = image2

write_images_binary(data, "/home/hongqingde/workspace_git/test/cdata_sparse/images.bin")

    #org = np.reshape(org_vec,(1,3))
    #print(R_w2c)
    # print(np.matmul(org,R_w2c))
    # print(targt_vec)