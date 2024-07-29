import numpy as np
import matplotlib.pyplot as plt
from read_write_model import *
import math
import open3d as o3d
import pyransac3d as pyrsc

def UniSampling(number):
    return np.linspace(0, 360, number, endpoint=True)

def LinerSampling(start,stop,number):
    return np.linspace(start, stop, number, endpoint=True)

control_point = [-0.009859,1.297023,0.11011]

# data = read_images_binary("/media/kim/newSpace/data/tower/sparse/images.bin")
coord = []

angle_list = UniSampling(500)

temp1 = angle_list[10:]
temp2 = angle_list[:10]
angle_list = np.concatenate((temp1,temp2))


height_list = LinerSampling(-1.5, -1.5,500)
r_list = LinerSampling(2.5,2.5,500)
#r_list = np.ones(500)*3.0
#circle formula
org = [0, 0]
a = org[0]
b = org[1]
r = 3
#angle = 30
for angle in range(angle_list.shape[0]):
    x = a + r_list[angle] * np.cos(np.radians(angle_list[angle]))
    y = b + r_list[angle] * np.sin(np.radians(angle_list[angle]))
    coord.append([x,y])

coord = np.asarray(coord)
# X = coord[:,0]
# Y = coord[:,1]
# plt.plot(X,Y,"o")
# plt.show()
H = -3
org_vec = np.array([0,0,1])

for i in range(500):
    #print(-coord[i][0],H,-coord[i][1])
    targt_vec = np.array([coord[i][0],0,coord[i][1]])
    norm = np.linalg.norm(targt_vec)
    targt_vec = targt_vec/norm

    c = np.dot(org_vec, targt_vec)
    n_vector = np.cross(org_vec, targt_vec)
    s = np.linalg.norm(n_vector)
    #print(c, s)

    n_vector_invert = np.array((
        [0,-n_vector[2],n_vector[1]],
        [n_vector[2],0,-n_vector[0]],
        [-n_vector[1],n_vector[0],0]
        ))
    I = np.eye(3)
    # 核心公式：见上图
    R_w2c = I + n_vector_invert + np.dot(n_vector_invert, n_vector_invert)/(1+c)
    R_w2c = np.dot(R_w2c, np.array([[-1,0,0],
                                         [0,-1,0],
                                        [0,0,1]]))
    # R_w2c = np.dot(np.array([[-1, 0, 0],
    #                                 [0, -1, 0],
    #                                 [0, 0, 1]]),R_w2c)
    sta = np.arctan(height_list[i]/r)
    sta = -sta
    transM_X = np.array([[1,0,0],
                       [0,np.cos(sta),np.sin(sta)],
                      [0,-np.sin(sta),np.cos(sta)]])
    R_w2c = np.dot(transM_X,R_w2c)
    transM_Z = np.array([[np.cos(np.pi),np.sin(np.pi),0],
                       [-np.sin(np.pi),np.cos(np.pi),0],
                      [0,0,1]])
    R_w2c = np.dot(transM_Z, R_w2c)
    # print(org_vec)
    # print(np.matmul(org_vec, R_w2c))
    # print(targt_vec)
    #Rt_c2w = np.linalg.inv(R_w2c)
    R = np.linalg.inv(R_w2c)
    t = [-coord[i][0]+control_point[0],height_list[i]+control_point[1],-coord[i][1]+control_point[2]]
    #t = [-coord[i][0], height_list[i], -coord[i][1]]
    t = np.reshape(t,(3,1))
    Rt = np.hstack((R,t))
    Rt = np.vstack((Rt,np.array([0,0,0,1]).T))
    Rt = np.linalg.inv(Rt)

    # sta = np.arctan(H/r)
    # transM = np.array([[1,0,0],
    #                    [0,np.cos(sta),np.sin(sta)],
    #                   [0,-np.sin(sta),np.cos(sta)]])
    # R_w2c = np.dot(transM,R_w2c)
    qv = rotmat2qve
    c(R_w2c)
    tv = Rt[:3,3]
    image1 = data[1]
    image1 = image1._replace(id=i,qvec=qv,tvec=tv)
    data[1+i] = image1
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

write_images_binary(data, "/media/kim/newSpace/data/tower_render/sparse/images.bin")

    #org = np.reshape(org_vec,(1,3))
    #print(R_w2c)
    # print(np.matmul(org,R_w2c))
    # print(targt_vec)