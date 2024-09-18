# This script is based on an original implementation by True Price.
# 用于手动从database.db中读取相机参数并更改为cameras.txt中的相机参数
# 参考：https://www.cnblogs.com/li-minghao/p/11865794.html
import os
import argparse
import math
import collections

import subprocess
import cv2
import stat
import struct
import sys
import numpy as np
import sqlite3
import shutil

sys.path.append('.')
from .baseTools.pose_process import get_intrinsic_params
from .baseTools.colmap_read_write_model import (Camera, camTodatabase, COLMAPDatabase, CAMERA_MODEL_IDS,
                                                write_cameras_text, write_images_text, read_images_text_from_GTAsColPose,
                                                blob_to_array)
from .baseTools.file_process import set_write_permission, add_write_permission_to_files_chmod, remove_directory_chmod


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--createDir", type=str, default=None)
    parser.add_argument("--useGivenIntrinsicParams", type=bool, default=None)
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--isInv", default=None)  # 是否要对读取的外参信息求逆，colmap使用的是相机到世界的外参，若已有的外参信息是世界到相机的外参（例如gt_as_col_pose.txt），则需要对其进行求逆
    parser.add_argument("--BOWPath", type=str, default=None)
    parser.add_argument("--camModel", type=str, default="SIMPLE_PINHOLE")
    args = parser.parse_args()

    dataPath = os.path.join(args.createDir, "train")
    extrinsicPath = os.path.join(dataPath, "gt_as_col_pose.txt")
    imagesPath = os.path.join(dataPath, 'images')

    sparsePath = os.path.join(args.createDir, "sparse")
    databasePath = os.path.join(sparsePath, "database.db")
    outputPath = os.path.join(sparsePath, '0')
    inputPath = os.path.join(args.createDir, 'created/sparse')

    camModel = args.camModel

    # 参数准备
    if not os.path.exists(sparsePath):
        os.makedirs(sparsePath, exist_ok=True)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath, exist_ok=True)

    assert os.path.exists(imagesPath) and os.path.exists(extrinsicPath), "please check indeeded file exists"
    imgNameTmp = os.listdir(imagesPath)[0]
    imgTmp = cv2.imread(os.path.join(imagesPath, imgNameTmp))
    resoluation = [imgTmp.shape[1], imgTmp.shape[0]]

    isInv = False if not args.isInv else True
    extrinsicParams = read_images_text_from_GTAsColPose(extrinsicPath, isInv)

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

    # 特征匹配
    # # 穷举匹配法，理论上最好，但是在图像重复性过多时效果可能会变差
    # feature_matcher = (subprocess.check_output(
    #     ['colmap', 'exhaustive_matcher', '--database_path', databasePath],
    #     universal_newlines=True))

    # 带回环的序列匹配方式
    assert args.BOWPath is not None and os.path.exists(args.BOWPath), "please check indeeded file exists"
    feature_matcher = (subprocess.check_output(
        ['colmap', 'sequential_matcher', '--database_path', databasePath,
         '--SequentialMatching.quadratic_overlap', '0', '--SequentialMatching.loop_detection', '1',
         '--SequentialMatching.vocab_tree_path', args.BOWPath]
    ))

    # 三角测量
    if not os.path.exists(inputPath):
        os.makedirs(inputPath, exist_ok=True)
    write_images_text(extrinsicParams, os.path.join(inputPath, 'images.txt'))
    with open(os.path.join(inputPath, "points3D.txt"), "w") as file:
        file.write('')
    db = COLMAPDatabase.connect(databasePath)
    db.create_tables()
    rows = db.execute("SELECT * FROM cameras")
    camera_id, model, width, height, params, prior = next(rows)
    paramsArr = blob_to_array(params, np.float64)

    cameras = {}
    cameras[camera_id] = Camera(id=camera_id, model=CAMERA_MODEL_IDS[model].model_name, width=width, height=height,
                                params=paramsArr.tolist())
    write_cameras_text(cameras, os.path.join(inputPath, 'cameras.txt'))

    point_triangulator = (subprocess.check_output(
        [
            'colmap', 'point_triangulator', '--database_path', databasePath, '--image_path', imagesPath,
            '--input_path', inputPath, '--output_path', outputPath,
            # '--Mapper.tri_min_angle', '0.00001'
        ],
        universal_newlines=True))

    remove_directory_chmod(os.path.dirname(inputPath))

    # 修改文件权限
    add_write_permission_to_files_chmod(args.createDir)
