import cv2
import open3d as o3d
import os
import math
import numpy as np
import argparse
import collections
import sqlite3
import sys
import random
import subprocess
import struct
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(".")
from .baseTools.colmap_read_write_model import (read_images_text_from_GTAsColPose, blob_to_array,
                                                Point3D, Camera, COLMAPDatabase, camTodatabase, CAMERA_MODEL_IDS,
                                                write_points3D_binary, write_images_binary, write_cameras_binary)
from .baseTools.pose_process import depth2xyz, pc_cam_to_pc_world, get_intrinsic_params
from .baseTools.point3D_process import voxel_random_filter, voxel_texture_filter
from .baseTools.file_process import add_write_permission_to_files_chmod


def process_image(id_, w2c_, depthDir_, imagesDir_, K_, depthLimits_=(1000, 4, 40, 200)):
    depthClip_, num2_, num1_, ZLimit_ = depthLimits_[:]
    depthPath_ = os.path.join(depthDir_, id_ + ".npy")
    depth_ = np.load(depthPath_)
    depth_ = np.clip(depth_, 0, depthClip_)
    # col_depthf = os.path.join(depthDir_, id_ + ".png.geometric.bin")
    # if not os.path.exists(col_depthf):
    #     print("skip visualize for the file does not exists: {}".format(col_depthf))
    #     pass
    # pre_depth_np = read_array(col_depthf)
    # target_height, target_width = 1080, 1920
    # depth_height, depth_width = pre_depth_np.shape[:2]
    # pad_height = target_height - depth_height
    # pad_width = target_width - depth_width
    # depth = np.pad(pre_depth_np, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    imgPath_ = os.path.join(imagesDir_, id_ + ".png")
    img_ = cv2.imread(imgPath_)
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    sobelX_ = cv2.Sobel(img_, cv2.CV_64F, 1, 0, ksize=3)
    sobelY_ = cv2.Sobel(img_, cv2.CV_64F, 0, 1, ksize=3)
    h_, w_ = depth_.shape
    c2w_ = np.linalg.inv(w2c_)

    # 当前像素点对应的3D点，由其深度 转换为 相机坐标系下的3D点（depth_scale 根据深度图和位姿的尺度进行调整）
    cameraXYZMap_ = depth2xyz(depth_, K_, depthScale_=1.0)  # H W 3，xyz

    Z_ = 10
    colors_ = []
    point3ds_ = []
    normals_ = []
    sobelScores_ = []
    # 遍历每个像素点，每100个选1个
    for i_ in range(0, h_, num1_ if Z_ >= depthClip_ else num2_):
        for j_ in range(0, w_, num1_ if Z_ >= depthClip_ else num2_):
            y_ = i_
            x_ = j_
            Z_ = depth_[i_, j_]  # 对应的绝对深度值

            if (ZLimit_ <= 0) or (Z_ >= ZLimit_):
                continue

            color_ = img_[y_, x_]
            colors_.append(color_)

            xyz_ = cameraXYZMap_[y_, x_]  # 取出相机坐标系下 对应像素的 点云 Pc
            point3dGT_ = pc_cam_to_pc_world(xyz_, c2w_)  # 相机下的点云转换为世界坐标系下的点云 Pw
            point3ds_.append(point3dGT_)

            # 使用法线表征纹理的丰富度
            neighbors_ = cameraXYZMap_[max(y_ - 1, 0):min(y_ + 2, h_),
                         max(x_ - 1, 0):min(x_ + 2, w_)]  # 取出以当前3D点为中心的3x3邻居点
            normal_, eigenvector_, _ = cv2.PCACompute2(neighbors_.reshape(-1, 3), mean=None)
            normals_.append(normal_)

            # 当前像素的Sobel梯度作为纹理的度量
            sobelXI_ = sobelX_[y_, x_, :]
            sobelYI_ = sobelY_[y_, x_, :]
            sobelScore_ = np.linalg.norm(sobelXI_) + np.linalg.norm(sobelYI_)
            sobelScores_.append(sobelScore_)
    return np.array(colors_), np.array(point3ds_), np.array(normals_).squeeze(axis=1), np.array(sobelScores_)


def create_gt_points3D_by_depth_multi(trainDir_, K_, outputDir_, depthLimits_=(40, 10, 1000, 200), filterType_=0, fileName_='points3D.bin'):
    imagesDir_ = f"{trainDir_}/images"
    depthDir_ = f"{trainDir_}/depth"
    posePath_ = os.path.join(trainDir_, "gt_as_col_pose.txt")

    dw2c_ = read_images_text_from_GTAsColPose(posePath_, isInv_=False, isCamera_=False)  # gt pose (Twc)

    colors_ = []
    point3ds_ = []
    normals_ = []
    sobelScores_ = []

    dc2wKeysList_ = list(dw2c_.keys())
    traverseStep_ = 10
    dc2wLen_ = len(dc2wKeysList_)
    dc2wIdxLast_ = (dc2wLen_ // int(traverseStep_)) * int(traverseStep_)

    depthLimitsSub_ = depthLimits_[:4]
    for dc2wIdx_ in range(0, dc2wLen_, traverseStep_):
        dc2wEnd_ = dc2wIdx_ + traverseStep_ if dc2wIdx_ != dc2wIdxLast_ else dc2wLen_

        with ProcessPoolExecutor() as executor:
            futures_ = [executor.submit(process_image, id_, dw2c_[id_], depthDir_, imagesDir_, K_, depthLimitsSub_) for
                        id_
                        in dc2wKeysList_[dc2wIdx_:dc2wEnd_]]
            for future_ in as_completed(futures_):
                resultColors_, resultPoint3ds_, resultNormals_, resultSobelScores_ = future_.result()
                colors_.append(resultColors_)
                point3ds_.append(resultPoint3ds_)
                normals_.append(resultNormals_)
                sobelScores_.append(resultSobelScores_)

    # 保存点云
    point3ds_ = np.concatenate(point3ds_, axis=0)  # N 3
    colors_ = np.concatenate(colors_, axis=0)  # N 3
    normals_ = np.concatenate(normals_, axis=0)  # N 3
    sobelScores_ = np.concatenate(sobelScores_, axis=0)  # N,

    colors_ = colors_ / 255.0

    pointCloud_ = o3d.geometry.PointCloud()
    pointCloud_.points = o3d.utility.Vector3dVector(point3ds_)
    pointCloud_.colors = o3d.utility.Vector3dVector(colors_)
    pointCloud_.normals = o3d.utility.Vector3dVector(normals_)

    ptLimitMin_, ptLimitMax_ = depthLimits_[4], depthLimits_[5]

    if ptLimitMin_ != -1 and ptLimitMax_ != -1 and ptLimitMin_ < ptLimitMax_:
        if ptLimitMin_ <= point3ds_.shape[0] <= ptLimitMax_:
            pass
        elif point3ds_.shape[0] < ptLimitMin_:
            raise Exception
        else:
            downVoxelSizeL_, downVoxelSizeR_ = 0.01, 0.5
            downVoxelSizeM_ = -1
            canFlg = True
            while True:
                pcdDownMin_ = pointCloud_.voxel_down_sample(voxel_size=downVoxelSizeR_)
                ptNumsMin_ = np.array(pcdDownMin_.points).shape[0]
                if ptNumsMin_ > ptLimitMax_:
                    if downVoxelSizeR_ < 1.:
                        downVoxelSizeR_ *= 2
                    else:
                        print('point clouds are so more')
                        canFlg = False
                        break
                else:
                    break

            while True:
                pcdDownMax_ = pointCloud_.voxel_down_sample(voxel_size=downVoxelSizeL_)
                ptNumsMax_ = np.array(pcdDownMax_.points).shape[0]
                if ptNumsMax_ < ptLimitMin_:
                    if downVoxelSizeL_ > 0.01:
                        downVoxelSizeL_ /= 2.
                    else:
                        print('point clouds are so sparse')
                        canFlg = False
                        break
                else:
                    break

            if not canFlg:
                raise Exception

            while downVoxelSizeL_ < downVoxelSizeR_:
                downVoxelSizeM_ = (downVoxelSizeL_ + downVoxelSizeR_) / 2.
                pcdDown_ = pointCloud_.voxel_down_sample(voxel_size=downVoxelSizeM_)
                ptNumsT_ = np.array(pcdDown_.points).shape[0]

                if ptLimitMin_ <= ptNumsT_ <= ptLimitMax_:
                    break

                if ptNumsT_ < ptLimitMin_:
                    downVoxelSizeR_ = downVoxelSizeM_
                else:
                    downVoxelSizeL_ = downVoxelSizeM_

            pcdDown_ = pointCloud_.voxel_down_sample(voxel_size=downVoxelSizeM_)
            points_ = np.hstack([np.asarray(pcdDown_.points), np.asarray(pcdDown_.colors)])
    else:
        if filterType_ == 0:
            points_ = np.hstack([np.asarray(point3ds_), np.asarray(colors_)])
            pcdDown_ = pointCloud_
        else:
            if filterType_ == 1:
                downVoxelSize_ = 0.5
                pcdDown_ = pointCloud_.voxel_down_sample(voxel_size=downVoxelSize_)
            elif filterType_ == 2:
                pcdDown_ = voxel_random_filter(pointCloud_, 0.1)
            elif filterType_ == 3:
                randomIdx_, randomIdxTexture_ = voxel_texture_filter(pointCloud_, sobelScores_, 0.2)
                # pcdDown_ = pointCloud_.select_by_index(randomIdx_)
                pcdDown_ = pointCloud_.select_by_index(randomIdxTexture_)
                # print(f"dst num1 {len(pcdDown_.points)}, dst num2 {len(pcdDownTexture_.points)}")
            else:
                raise Exception
            points_ = np.hstack([np.asarray(pcdDown_.points), np.asarray(pcdDown_.colors)])

    outputPath_ = os.path.join(outputDir_, fileName_)
    ext_ = fileName_.split('.')[-1]
    if ext_ == 'txt':
        np.savetxt(outputPath_, points_, fmt="%.6f %.6f %.6f %.6f %.6f %.6f")
    elif ext_ == 'ply':
        o3d.io.write_point_cloud(outputPath_, pcdDown_)
    elif ext_ == 'bin':
        ptNums_ = points_.shape[0]
        points3D_ = {}
        points_[:, 3:6] = 255. * points_[:, 3:6]
        for ptId_ in range(ptNums_):
            points3D_[ptId_] = Point3D(
                id=ptId_,
                xyz=points_[ptId_, :3],
                rgb=points_[ptId_, 3:6].astype(int),
                error=np.array([0]),
                image_ids=np.array([]),
                point2D_idxs=np.array([])
            )
        write_points3D_binary(points3D_, outputPath_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--createDir", type=str, default=None)
    parser.add_argument("--useGivenIntrinsicParams", type=bool, default=None)
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--isInv",
                        default=None)  # 是否要对读取的外参信息求逆，colmap使用的是相机到世界的外参，若已有的外参信息是世界到相机的外参（例如gt_as_col_pose.txt），则需要对其进行求逆
    parser.add_argument("--BOWPath", type=str, default=None)
    parser.add_argument("--camModel", type=str, default="SIMPLE_PINHOLE")
    parser.add_argument("--filterType", type=int, default=None)
    parser.add_argument("--zLimits", type=str, default='1000,4,40,200')
    args = parser.parse_args()

    zLimits = []
    zElms = args.zLimits.split(',')
    zLimits = tuple([int(zElm) for zElm in zElms])

    dataPath = os.path.join(args.createDir, "train")
    extrinsicPath = os.path.join(dataPath, "gt_as_col_pose.txt")
    imagesPath = os.path.join(dataPath, 'images')
    depthPath = os.path.join(dataPath, 'depth')

    sparsePath = os.path.join(args.createDir, "sparse")
    databasePath = os.path.join(sparsePath, "database.db")
    outputPath = os.path.join(sparsePath, '0')
    # camerasPath = os.path.join(args.createDir, 'created/sparse/cameras.txt')
    # inputPath = os.path.join(args.createDir, 'created/sparse')

    camModel = args.camModel

    # 参数准备
    if not os.path.exists(sparsePath):
        os.makedirs(sparsePath, exist_ok=True)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath, exist_ok=True)

    assert os.path.exists(imagesPath) and os.path.exists(extrinsicPath) and os.path.exists(depthPath), \
        "please check indeeded file exists"
    imgNameTmp = os.listdir(imagesPath)[0]
    imgTmp = cv2.imread(os.path.join(imagesPath, imgNameTmp))
    resoluation = [imgTmp.shape[1], imgTmp.shape[0]]

    isInv = False if not args.isInv else True
    extrinsicParams = read_images_text_from_GTAsColPose(extrinsicPath, isInv)
    write_images_binary(extrinsicParams, os.path.join(outputPath, 'images.bin'))

    # 特征提取
    feature_extract = (subprocess.check_output(
        [
            'colmap', 'feature_extractor', '--database_path', databasePath, '--image_path', imagesPath,
            '--ImageReader.single_camera', '1', '--ImageReader.camera_model', camModel],
        universal_newlines=True))

    # 内参导入
    if not args.useGivenIntrinsicParams:
        pass
    else:
        assert args.fov is not None
        intrinsicParams = get_intrinsic_params(resoluation, args.fov, camModel)
        cameras = dict()
        cameras[1] = Camera(id=1, model=intrinsicParams[0],
                            width=intrinsicParams[1], height=intrinsicParams[2],
                            params=intrinsicParams[3:])
        camTodatabase(databasePath, cameras)

    db = COLMAPDatabase.connect(databasePath)
    db.create_tables()
    rows = db.execute("SELECT * FROM cameras")
    camera_id, model, width, height, params, prior = next(rows)
    paramsArr = blob_to_array(params, np.float64)
    if args.camModel == 'PINHOLE':
        K = np.array([
            [paramsArr[0], 0, paramsArr[2]],
            [0, paramsArr[1], paramsArr[3]],
            [0, 0, 1]
        ])
    else:
        K = np.array([
            [paramsArr[0], 0, paramsArr[1]],
            [0, paramsArr[0], paramsArr[2]],
            [0, 0, 1]
        ])

    cameras = {}
    cameras[camera_id] = Camera(id=camera_id, model=CAMERA_MODEL_IDS[model].model_name, width=width, height=height,
                                params=paramsArr.tolist())
    write_cameras_binary(cameras, os.path.join(outputPath, 'cameras.bin'))

    # 特征匹配（若不进行三角化，这一步可以不做）
    # # 穷举匹配法，理论上最好，但是在图像重复性过多时效果可能会变差
    # feature_matcher = (subprocess.check_output(
    #     ['colmap', 'exhaustive_matcher', '--database_path', databasePath],
    #     universal_newlines=True))

    # 带回环的序列匹配方式
    if args.BOWPath is not None:
        feature_matcher = (subprocess.check_output(
            ['colmap', 'sequential_matcher', '--database_path', databasePath,
             '--SequentialMatching.quadratic_overlap', '0', '--SequentialMatching.loop_detection', '1',
             '--SequentialMatching.vocab_tree_path', args.BOWPath]
        ))

    # 基于深度图生成点云
    assert args.filterType is not None
    create_gt_points3D_by_depth_multi(dataPath, K, outputPath, depthLimits_=zLimits, filterType_=args.filterType,
                                      fileName_='points3D.ply')

    # 修改文件权限
    add_write_permission_to_files_chmod(args.createDir)