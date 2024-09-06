import os
import sys
import collections

import numpy as np
import geo_tools as gtls

import matplotlib.pyplot as plt

from PIL import Image
from typing import NamedTuple
from pcp_tools.colmap_loader import (read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
                                     read_extrinsics_binary, read_intrinsics_binary)

from read_write_model import read_points3D_binary, read_points3D_text, read_next_bytes


class CameraInfoSim(NamedTuple):
    KInv: np.array
    centerN: np.array
    center: np.array


FrameInfo = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "points3D"])


def search_pt_idx(p3dIdx_, points3D_):
    first_ = 0
    last_ = len(points3D_)
    while first_ <= last_:
        midPoint_ = first_ + (last_ - first_) // 2

        if points3D_[midPoint_][0] == p3dIdx_:
            return midPoint_
        elif points3D_[midPoint_][0] < p3dIdx_:
            first_ = midPoint_ + 1
        else:
            last_ = midPoint_ - 1
    return -1


def readColmapSceneInfo(path, isDegree=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        points3D_file = os.path.join(path, "sparse/0/points3D.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        p3ds_ = read_points3D_binary(points3D_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        points3D_file = os.path.join(path, "sparse/0/points3D.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        p3ds_ = read_points3D_binary(points3D_file)

    # reading_dir = "images" if images is None else images
    # # colmap中存储的旋转是相机坐标系到世界坐标系的旋转，平移量是相机坐标系下的平移，但是由于读取时的对旋转矩阵的转置操作，
    # # 读取到的外参中的旋转是世界坐标系到相机坐标系的旋转，平移量是相机坐标系下的平移
    # cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
    #                                        images_folder=os.path.join(path, reading_dir))
    # cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    frameInfos_ = {}

    for frameId_ in cam_extrinsics.keys():
        camEx_ = cam_extrinsics[frameId_]

        points3D_ = []
        p3dNum_ = len(camEx_.point3D_ids)
        for pIdx_ in range(p3dNum_):
            if camEx_.point3D_ids[pIdx_] != -1:
                points3D_.append([camEx_.point3D_ids[pIdx_], camEx_.xys[pIdx_, 0], camEx_.xys[pIdx_, 1]])

        points3D_ = sorted(points3D_, key=lambda x: x[0])
        frameInfos_[frameId_] = FrameInfo(id=camEx_.id, qvec=camEx_.qvec, tvec=camEx_.tvec,
                                          camera_id=camEx_.camera_id, name=camEx_.name, points3D=points3D_)

    cam_extrinsics.clear()

    KInvs_ = {}
    for camId_ in cam_intrinsics.keys():
        camParams_ = cam_intrinsics[camId_].params
        camH_ = cam_intrinsics[camId_].height
        camW_ = cam_intrinsics[camId_].width
        camCenter_ = np.array([float(camW_) / 2., float(camH_) / 2., 1.])
        KTmp_ = np.array([
            [camParams_[0], 0, camParams_[2]],
            [0, camParams_[1], camParams_[3]],
            [0, 0, 1]
        ])
        KInvT_ = np.linalg.inv(KTmp_)
        camCenterN_ = np.dot(KInvT_, camCenter_)
        KInvs_[camId_] = CameraInfoSim(KInv=KInvT_, centerN=camCenterN_, center=camCenter_)

    ptViewStatus_ = []
    cott_ = 0

    if isDegree:
        for p3dId_ in p3ds_.keys():
            print(cott_)
            cott_ += 1
            p3d_ = p3ds_[p3dId_]

            ptToFrameViews_ = []
            for frameId_ in p3d_.image_ids:
                frameEx_ = frameInfos_[frameId_]

                p2dIdx_ = search_pt_idx(p3dId_, frameEx_.points3D)
                assert p2dIdx_ != -1
                frameKInv_ = KInvs_[frameEx_.camera_id].KInv
                frameCenterN_ = KInvs_[frameEx_.camera_id].centerN
                framePt_ = np.array([frameEx_.points3D[p2dIdx_][0], frameEx_.points3D[p2dIdx_][1], 1])
                framePtN_ = np.dot(frameKInv_, framePt_)

                ptToFrameViews_.append(gtls.calc_cross_degree(frameCenterN_, framePtN_))

            assert len(ptToFrameViews_) > 1, "single image couldn't generate stereo points"
            ptToFrameViews_ = np.array(ptToFrameViews_)
            viewStatusAvg_ = np.mean(ptToFrameViews_)
            viewStatusStd_ = np.std(ptToFrameViews_)
            ptViewStatus_.append([viewStatusAvg_, viewStatusStd_])
    else:
        for p3dId_ in p3ds_.keys():
            print(cott_)
            cott_ += 1
            p3d_ = p3ds_[p3dId_]

            ptToFrameViews_ = []
            for frameId_ in p3d_.image_ids:
                frameEx_ = frameInfos_[frameId_]

                p2dIdx_ = search_pt_idx(p3dId_, frameEx_.points3D)
                assert p2dIdx_ != -1
                frameCenter_ = KInvs_[frameEx_.camera_id].center
                framePt_ = np.array([frameEx_.points3D[p2dIdx_][0], frameEx_.points3D[p2dIdx_][1], 1.])
                ptToFrameViews_.append(np.linalg.norm(frameCenter_-framePt_))

            assert len(ptToFrameViews_) > 1, "single image couldn't generate stereo points"
            ptToFrameViews_ = np.array(ptToFrameViews_)
            viewStatusAvg_ = np.min(ptToFrameViews_)
            viewStatusStd_ = np.std(ptToFrameViews_)
            ptViewStatus_.append([viewStatusAvg_, viewStatusStd_])

    viewStatusAvgs_ = [stus_[0] for stus_ in ptViewStatus_]
    viewStatusStds_ = [stus_[1] for stus_ in ptViewStatus_]

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(np.array(viewStatusAvgs_), bins=360, alpha=0.7, color='blue', edgecolor='black')
    # plt.hist(np.array(viewStatusStds_), bins=360, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Data Distribution Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    plt.legend()
    # 显示图形
    plt.show()


if __name__ == '__main__':
    path = "/home/hongqingde/devdata/workspace_gitmp/output/20240905_fukan_gps_gai2"
    images = "train/images"
    eval = False
    readColmapSceneInfo(path, isDegree=False)

    pass
