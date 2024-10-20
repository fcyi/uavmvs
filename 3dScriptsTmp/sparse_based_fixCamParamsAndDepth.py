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


IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1
CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""
CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""
CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)
CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""
CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""
CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""
CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"
CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)
ExtrinsicParams = collections.namedtuple(
    "Image", ["id", "params", "camId", "name"]  # params = [qw, qx, qy, qz, tx, ty, tz]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


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


def voxel_random_filter(cloud_, leafSize_):
    pointCloud_ = np.asarray(cloud_.points)  # N 3
    # 1、计算边界点
    xMin_, yMin_, zMin_ = np.amin(pointCloud_, axis=0)  # 按列寻找点云位置的最小值
    xMax_, yMax_, zMax_ = np.amax(pointCloud_, axis=0)
    # 2、计算每个维度上体素格网的个数
    Dx_ = (xMax_ - xMin_) // leafSize_ + 1
    Dy_ = (yMax_ - yMin_) // leafSize_ + 1
    Dz_ = (zMax_ - zMin_) // leafSize_ + 1
    print("Dx * Dy * Dz is {} * {} * {}".format(Dx_, Dy_, Dz_))
    # 3、计算每个点的格网idx
    h_ = list()
    for i_ in range(len(pointCloud_)):
        # 分别在x, y, z方向上格网的idx
        hx_ = (pointCloud_[i_][0] - xMin_) // leafSize_
        hy_ = (pointCloud_[i_][1] - yMin_) // leafSize_
        hz_ = (pointCloud_[i_][2] - zMin_) // leafSize_
        h_.append(hx_ + hy_ * Dx_ + hz_ * Dx_ * Dy_)   # 该点所在格网 映射到1D的idx
    h_ = np.array(h_)

    # 4、体素格网内随机筛选点
    hIndice_ = np.argsort(h_)
    hSorted_ = h_[hIndice_]
    #################################
    # h_         3 1 2 1     每个3D点的格网idx，其位置也为对应3D点在点云中的位置索引
    # hIndice_  1 3 2 0     升序排列时，每个3D点的格网idx在h中的索引
    # hSorted_  1 1 2 3     升序排列后的 每个3D点的格网idx
    #################################
    randomIdx_ = []
    begin_ = 0
    # 遍历每个3D点的格网idx
    for i_ in range(len(hSorted_) - 1):
        # 当前3D点和后一个3D点的格网idx 相同，则跳过
        if hSorted_[i_] == hSorted_[i_ + 1]:
            continue
        # 当前3D点和后一个3D点的格网idx 不相同，则在当前3D点所在格网内随机选择一个3D点
        else:
            # begin_: 在同一个格网内的第一个3D点的格网idx的 在hSorted_/hIndice_的位置，i：在同一个格网内的最后一个3D点的格网idx 在hSorted_/hIndice_的位置
            pointIdx_ = hIndice_[begin_: i_ + 1]  # 同一格网内 所有3D点的格网idx在h中的位置，也是这些3D点在点云中的索引
            randomIdx_.append(random.choice(pointIdx_))  # 在同一格网内 随机选择一个3D点
            begin_ = i_ + 1
    filteredPoints_ = (cloud_.select_by_index(randomIdx_))
    return filteredPoints_


def voxel_texture_filter(cloud_, sobelScores_, leafSize_):
    pointCloud_ = np.asarray(cloud_.points)  # N 3
    # colors_ = np.asarray(cloud_.colors) / 255.0 # N 3
    normals_ = np.asarray(cloud_.normals)
    # 1、计算边界点
    xMin_, yMin_, zMin_ = np.amin(pointCloud_, axis=0)
    xMax_, yMax_, zMax_ = np.amax(pointCloud_, axis=0)
    # 2、计算每个维度上体素格网的个数
    Dx_ = (xMax_ - xMin_) // leafSize_ + 1
    Dy_ = (yMax_ - yMin_) // leafSize_ + 1
    Dz_ = (zMax_ - zMin_) // leafSize_ + 1
    print("Dx * Dy * Dz is {} * {} * {}".format(Dx_, Dy_, Dz_))
    # 3、计算每个3D点的格网idx
    h_ = list()
    for i_ in range(len(pointCloud_)):
        hx_ = (pointCloud_[i_][0] - xMin_) // leafSize_
        hy_ = (pointCloud_[i_][1] - yMin_) // leafSize_
        hz_ = (pointCloud_[i_][2] - zMin_) // leafSize_
        h_.append(hx_ + hy_ * Dx_ + hz_ * Dx_ * Dy_)
    h_ = np.array(h_)

    # 4、根据纹理度量保留3D点
    hIndice_ = np.argsort(h_)
    hSorted_ = h_[hIndice_]
    #################################
    # h_        9 1 7 1 1     每个3D点的格网idx，其位置也为对应3D点在点云中的位置索引
    # hIndice_  1 3 4 2 0     升序排列时，每个3D点的格网idx在h中的索引
    # hSorted_  1 1 1 7 9     升序排列后的 每个3D点的格网idx
    #################################
    begin_ = 0
    meanNormals_ = np.mean(np.std(normals_, axis=0))
    stdNormals_ = np.std(np.std(normals_, axis=0))   # np.std(normals_, axis=0) 1, 3 每个3D点的法线向量在xyz维度的标准差，该区域的法向量分布较为离散,即表面法线变化剧烈,说明纹理丰富
    meanSobelScores_ = np.mean(sobelScores_)
    stdSobelScores_ = np.std(sobelScores_)
    pointIdxScores_ = {}
    textureScores_ = []
    for i_ in range(len(hIndice_) - 1):
        # 当前3D点和后一个3D点的格网idx 相同，则跳过
        if hSorted_[i_] == hSorted_[i_ + 1]:
            continue
        # 当前3D点和后一个3D点的格网idx 不相同，则在当前3D点所在格网内
        else:
            # begin_: 在同一个格网内的第一个3D点的格网idx的 在hIndice_的位置，i：在同一个格网内的最后一个3D点的格网idx 在hIndice_的位置
            pointIdx_ = hIndice_[begin_: i_ + 1]  # 同一格网内 所有3D点的格网idx在h中的位置，也是这些3D点在点云中的位置

            # 计算每个格网纹理度量得分
            normalsInVoxel_ = normals_[pointIdx_]
            sobelScoresInVoxel_ = sobelScores_[pointIdx_]
            avgSobelScore_ = (np.mean(sobelScoresInVoxel_) - meanSobelScores_) / stdSobelScores_
            avgNormalScore_ = (np.mean(np.std(normalsInVoxel_, axis=0)) - meanNormals_) / stdNormals_
            textureScore_ = 0.5 * avgSobelScore_ + 0.5 * avgNormalScore_
            textureScores_.append(textureScore_)

            pointIdxScores_[h_[hIndice_[begin_]]] = [pointIdx_, textureScore_]
            begin_ = i_ + 1

    # 计算所有包含3D点云的格网的纹理度量得分
    meanTextureScore_ = np.mean(textureScores_)
    stdTextureScore_ = np.std(textureScores_)
    maxTextureScore_ = np.max(textureScores_)
    minTextureScore_ = np.min(textureScores_)

    oriPointIdx_ = []
    randomIdx_ = []
    randomIdxTexture_ = []
    # 根据每个格网与所有格网的纹理度量得分比值，确定该格网内保留的点云数量
    for pointIdx_, textureScore_ in pointIdxScores_.values():
        oriPointIdx_.append(pointIdx_)

        randomIdx_.append(random.choice(pointIdx_))

        textureWeight_ = (textureScore_ - minTextureScore_) / (maxTextureScore_ - minTextureScore_)
        numPoints_ = max(1, int(len(pointIdx_) * textureWeight_))
        # numPoints_ = max(1, np.clip(int(len(pointIdx_) * texture_weight), 0, len(pointIdx_)//2)) # 该clip方案实际并不会影响结果，因为计算的保留点数结果 < 总个数的一半
        randomIdxTexture_.extend(random.sample(list(pointIdx_), numPoints_))  # 在同一格网内 随机选取numPoints_个3D点

    print("oriPointIdx_: {}, randomIdx_: {}, randomIdxTexture_: {}".format(len(oriPointIdx_), len(randomIdx_), len(randomIdxTexture_)))

    return randomIdx_, randomIdxTexture_

    # filteredPoints_ = (cloud_.select_by_index(randomIdx_))
    # filteredPointsTexture_ = (cloud_.select_by_index(randomIdxTexture_))
    # return filteredPoints_, filteredPointsTexture_


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


def pose_inverse(pose_):
    qvec_ = np.copy(pose_[:4])
    qvecNorm_ = np.linalg.norm(qvec_)
    if abs(qvecNorm_ - 1.0) > 1e-6:
        print(f"Warning: Quaternion {qvec_} is not normalized.")
        qvec_ = qvec_ / qvecNorm_
    qvec_ = qvec_.tolist()
    RTmp_ = qvec2rotmat(qvec_[:])
    TTmp_ = np.eye(4, 4)
    TTmp_[:3, :3] = RTmp_
    TTmp_[:3, 3] = np.copy(pose_[4:])
    TInvTmp_ = np.linalg.inv(TTmp_)
    RInvTmp_ = TInvTmp_[:3, :3]
    tInvTmp_ = TInvTmp_[:3, 3]
    qvecInv_ = rotmat2qvec(RInvTmp_)
    qvecInvNorm_ = np.linalg.norm(qvecInv_)
    if abs(qvecInvNorm_ - 1.0) > 1e-6:
        print(f"Warning: Quaternion {qvec_} is not normalized.")
        qvecInv_ = qvecInv_ / qvecInvNorm_
    poseNew_ = qvecInv_.tolist() + tInvTmp_.tolist()
    return poseNew_


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
        qvecNorm_ = np.linalg.norm(qvec_)
        if abs(qvecNorm_ - 1.0) > 1e-6:
            print(f"Warning: Quaternion {qvec_} is not normalized.")
            qvec_ = qvec_ / qvecNorm_
        qvec_ = qvec_.tolist()
        RTmp_ = qvec2rotmat(qvec_[:])
        poseRes_ = np.eye(4, 4)
        poseRes_[:3, :3] = RTmp_
        poseRes_[:3, 3] = np.copy(poseSrcTmp_[4:])
    else:
        RTmp_ = poseSrcTmp_[:3, :3]
        tTmp_ = poseSrcTmp_[:3, 3]
        qvec_ = rotmat2qvec(RTmp_)
        qvecNorm_ = np.linalg.norm(qvec_)
        if abs(qvecNorm_ - 1.0) > 1e-6:
            print(f"Warning: Quaternion {qvec_} is not normalized.")
            qvec_ = qvec_ / qvecNorm_
        poseRes_ = qvec_.tolist() + tTmp_.tolist()
    return poseRes_


def read_images_text_from_GTAsColPose(inputPath_, isInv_=False, isCamera_=True):
    assert os.path.exists(inputPath_)
    extrinsicParams_ = dict()

    with open(inputPath_, 'r') as fid_:
        fcot_ = 1
        camId_ = 1
        for line_ in fid_:
            if len(line_) <= 0 or line_[0] == '#':
                continue
            line_ = line_.strip()
            elements_ = line_.split()

            paramsTmp_ = np.array(tuple(map(float, [elements_[7], *elements_[4:7], *elements_[1:4]])))
            if isInv_:
                params_ = pose_inverse(paramsTmp_)
            else:
                params_ = paramsTmp_.tolist()
            if isCamera_:
                fileName = elements_[0] + '.png'
                extrinsicParams_[fcot_] = Image(
                    id=fcot_,
                    qvec=np.array(tuple(map(float, params_[:4]))),
                    tvec=np.array(tuple(map(float, params_[4:]))),
                    camera_id=camId_,
                    name=fileName,
                    xys=np.array([]), point3D_ids=np.array([]))
                fcot_ += 1
            else:
                extrinsicParams_[elements_[0]] = pose7_2_poseM(params_)
    return extrinsicParams_


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


def write_cameras_text(cameras, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "# Number of cameras: {}\n".format(len(cameras))
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")


def write_next_bytes(fid_, data_, formatCharSequence_, endianCharacter_="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param formatCharSequence_: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endianCharacter_: Any of {@, =, <, >, !}
    """
    if isinstance(data_, (list, tuple)):
        bytes_ = struct.pack(endianCharacter_ + formatCharSequence_, *data_)
    else:
        bytes_ = struct.pack(endianCharacter_ + formatCharSequence_, data_)
    fid_.write(bytes_)


def write_images_binary(images_, pathToModelFile_):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(pathToModelFile_, "wb") as fid_:
        write_next_bytes(fid_, len(images_), "Q")
        for _, img_ in images_.items():
            write_next_bytes(fid_, img_.id, "i")
            write_next_bytes(fid_, img_.qvec.tolist(), "dddd")
            write_next_bytes(fid_, img_.tvec.tolist(), "ddd")
            write_next_bytes(fid_, img_.camera_id, "i")
            for char_ in img_.name:
                write_next_bytes(fid_, char_.encode("utf-8"), "c")
            write_next_bytes(fid_, b"\x00", "c")
            write_next_bytes(fid_, len(img_.point3D_ids), "Q")
            for xy_, p3dId_ in zip(img_.xys, img_.point3D_ids):
                write_next_bytes(fid_, [*xy_, p3dId_], "ddq")


def write_points3D_binary(points3D_, pathToModelFile_):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(pathToModelFile_, "wb") as fid_:
        write_next_bytes(fid_, len(points3D_), "Q")
        for _, pt_ in points3D_.items():
            write_next_bytes(fid_, pt_.id, "Q")
            write_next_bytes(fid_, pt_.xyz.tolist(), "ddd")
            write_next_bytes(fid_, pt_.rgb.tolist(), "BBB")
            write_next_bytes(fid_, pt_.error, "d")
            trackLength_ = pt_.image_ids.shape[0]
            write_next_bytes(fid_, trackLength_, "Q")
            for imageId_, point2DId_ in zip(pt_.image_ids, pt_.point2D_idxs):
                write_next_bytes(fid_, [imageId_, point2DId_], "ii")


def write_cameras_binary(cameras_, pathToModelFile_):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(pathToModelFile_, "wb") as fid_:
        write_next_bytes(fid_, len(cameras_), "Q")
        for _, cam_ in cameras_.items():
            print(cam_.model)
            modelId_ = CAMERA_MODEL_NAMES[cam_.model].model_id
            cameraProperties_ = [cam_.id, modelId_, cam_.width, cam_.height]
            write_next_bytes(fid_, cameraProperties_, "iiQQ")
            for p_ in cam_.params:
                write_next_bytes(fid_, float(p_), "d")
    return cameras_


def array_to_blob(array_):
    if IS_PYTHON3:
        return array_.tostring()
    else:
        return np.getbuffer(array_)


def blob_to_array(blob_, dtype_, shape_=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob_, dtype=dtype_).reshape(*shape_)
    else:
        return np.frombuffer(blob_, dtype=dtype_).reshape(*shape_)


class COLMAPDatabase(sqlite3.Connection):
    """
    COLMAPDatabase 是一个自定义类，它扩展了 SQLite 数据库的功能，例如与 COLMAP 相关的数据操作。
    """
    @staticmethod
    def connect(database_path):
        # 使用 sqlite3.connect(database_path, factory=COLMAPDatabase) 语句可以连接到一个 SQLite 数据库，
        # 并且可以通过指定 factory 参数来使用自定义的数据库类（例如 COLMAPDatabase）
        # 如果指定的数据库文件路径不存在，sqlite3.connect() 将会抛出错误。确保路径正确。
        # 如果没有足够的权限访问数据库文件，连接也会失败。这可能发生在某些系统或目录中。
        # 确保 COLMAPDatabase 类中的方法和属性与您想要操作的数据结构相匹配。
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)
        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model_, width_, height_, params_, cameraId_):
        params_ = np.asarray(params_, np.float64)
        cursor_ = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=True WHERE camera_id=?",
            (model_, width_, height_, array_to_blob(params_), cameraId_))
        return cursor_.lastrowid


def camTodatabase(databasePath_, camerasInfo_):
    camModelDict_ = {'SIMPLE_PINHOLE': 0,
                    'PINHOLE': 1,
                    'SIMPLE_RADIAL': 2,
                    'RADIAL': 3,
                    'OPENCV': 4,
                    'FULL_OPENCV': 5,
                    'SIMPLE_RADIAL_FISHEYE': 6,
                    'RADIAL_FISHEYE': 7,
                    'OPENCV_FISHEYE': 8,
                    'FOV': 9,
                    'THIN_PRISM_FISHEYE': 10}
    if not os.path.exists(databasePath_):
        print("ERROR: database path dosen't exist -- please check database.db.")
        return

    if type(camerasInfo_).__name__ == 'str' and not os.path.exists(camerasInfo_):
        print("ERROR: cameras path dosen't exist -- please check cameras.txt.")
        return

    # Open the database.
    db_ = COLMAPDatabase.connect(databasePath_)
    idList_ = list()
    modelList_ = list()
    widthList_ = list()
    heightList_ = list()
    paramsList_ = list()

    if type(camerasInfo_).__name__ == 'str':
        # Update real cameras from .txt
        with open(camerasInfo_, "r") as cam_:
            lines_ = cam_.readlines()
            for i_ in range(0, len(lines_), 1):
                if lines_[i_][0] != '#':
                    strLists_ = lines_[i_].split()
                    cameraId_ = int(strLists_[0])
                    cameraModel_ = camModelDict_[strLists_[1]]  # SelectCameraModel
                    width_ = int(strLists_[2])
                    height_ = int(strLists_[3])
                    paramstr_ = np.array(strLists_[4:12])
                    params_ = paramstr_.astype(np.float64)
                    idList_.append(cameraId_)
                    modelList_.append(cameraModel_)
                    widthList_.append(width_)
                    heightList_.append(height_)
                    paramsList_.append(params_)
                    cameraIdTmp_ = db_.update_camera(cameraModel_, width_, height_, params_, cameraId_)
    elif type(camerasInfo_).__name__ == 'dict':
        for cameraId_, cameraV_ in camerasInfo_.items():
            cameraModel_ = camModelDict_[cameraV_.model]  # SelectCameraModel
            width_ = int(cameraV_.width)
            height_ = int(cameraV_.height)
            paramstr_ = np.array(cameraV_.params)
            params_ = paramstr_.astype(np.float64)
            idList_.append(cameraId_)
            modelList_.append(cameraModel_)
            widthList_.append(width_)
            heightList_.append(height_)
            paramsList_.append(params_)
            cameraIdTmp_ = db_.update_camera(cameraModel_, width_, height_, params_, cameraId_)

    # Commit the data to the file.
    db_.commit()
    # Read and check cameras.
    rows_ = db_.execute("SELECT * FROM cameras")
    for i_ in range(0, len(idList_), 1):
        cameraId_, model_, width_, height_, params_, prior_ = next(rows_)
        params_ = blob_to_array(params_, np.float64)
        assert cameraId_ == idList_[i_]
        assert model_ == modelList_[i_] and width_ == widthList_[i_] and height_ == heightList_[i_]
        assert np.allclose(params_, paramsList_[i_])

    # Close database.db.
    db_.close()


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


def create_gt_points3D_by_depth_multi(trainDir_, K_, outputDir_, depthLimits_=(40, 10, 1000, 200, -1, -1), filterType_=0, fileName_='points3D.bin'):
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
    dc2wIdxLast_ = (dc2wLen_ // int(traverseStep_))*int(traverseStep_)

    depthLimitsSub_ = depthLimits_[:4]
    for dc2wIdx_ in range(0, dc2wLen_, traverseStep_):
        dc2wEnd_ = dc2wIdx_+traverseStep_ if dc2wIdx_ != dc2wIdxLast_ else dc2wLen_

        with ProcessPoolExecutor() as executor:
            futures_ = [executor.submit(process_image, id_, dw2c_[id_], depthDir_, imagesDir_, K_, depthLimitsSub_) for id_
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
    print("========================================================")
    print(ptLimitMin_, ptLimitMax_, point3ds_.shape[0])
    if ptLimitMin_ != -1 and ptLimitMax_ != -1 and ptLimitMin_ < ptLimitMax_:
        if ptLimitMin_ <= point3ds_.shape[0] <= ptLimitMax_:
            points_ = np.hstack([np.asarray(point3ds_), np.asarray(colors_)])
            pcdDown_ = pointCloud_
        elif point3ds_.shape[0] < ptLimitMin_:
            raise Exception
        else:
            downVoxelSizeL_, downVoxelSizeR_ = 0.4, 0.5
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
            print('---')
            print(downVoxelSizeR_)

            while True:
                pcdDownMax_ = pointCloud_.voxel_down_sample(voxel_size=downVoxelSizeL_)
                ptNumsMax_ = np.array(pcdDownMax_.points).shape[0]
                if ptNumsMax_ < ptLimitMin_:
                    if downVoxelSizeL_ > 0.02:
                        downVoxelSizeL_ /= 2.
                    else:
                        print('point clouds are so sparse')
                        canFlg = False
                        break
                else:
                    break
            print('---')
            print(downVoxelSizeL_)

            if not canFlg:
                raise Exception

            while downVoxelSizeL_ < downVoxelSizeR_:
                downVoxelSizeM_ = (downVoxelSizeL_ + downVoxelSizeR_) / 2.
                pcdDown_ = pointCloud_.voxel_down_sample(voxel_size=downVoxelSizeM_)
                ptNumsT_ = np.array(pcdDown_.points).shape[0]
                print(downVoxelSizeL_, downVoxelSizeR_, downVoxelSizeM_, ptNumsT_)

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


def add_write_permission_to_files_chmod(directory_):
    try:
        # 使用 subprocess 调用 chmod 命令，不带通配符
        subprocess.run(['chmod', '-R', '777', directory_], check=True)
        print(f"Permissions changed to 777 for all files and directories in {directory_}.")
    except subprocess.CalledProcessError as e_:
        print(f"Error occurred: {e_}")


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
    parser.add_argument("--zLimits", type=str, default='1000,10,40,200,-1,-1')
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
    print(imagesPath, extrinsicPath, depthPath)
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
    create_gt_points3D_by_depth_multi(dataPath, K, outputPath, depthLimits_=zLimits, filterType_=args.filterType, fileName_='points3D.bin')

    # 修改文件权限
    add_write_permission_to_files_chmod(args.createDir)
