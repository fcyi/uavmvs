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


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


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


def read_images_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            binary_image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                binary_image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = binary_image_name.decode("utf-8")
            num_points2D = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def write_images_text(images_, path_):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    if len(images_) == 0:
        meanObservations_ = 0
    else:
        meanObservations_ = sum(
            (len(img_.point3D_ids) for _, img_ in images_.items())
        ) / len(images_)
    HEADER_ = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + "# Number of images: {}, mean observations per image: {}\n".format(
            len(images_), meanObservations_
        )
    )

    with open(path_, "w") as fid_:
        fid_.write(HEADER_)
        for _, img_ in images_.items():
            imageHeader_ = [
                img_.id,
                *img_.qvec,
                *img_.tvec,
                img_.camera_id,
                img_.name,
            ]
            firstLine_ = " ".join(map(str, imageHeader_))
            fid_.write(firstLine_ + "\n")

            pointsStrings_ = []
            for xy_, point3DId_ in zip(img_.xys, img_.point3D_ids):
                pointsStrings_.append(" ".join(map(str, [*xy_, point3DId_])))
            fid_.write(" ".join(pointsStrings_) + "\n")


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
        fy_ = height_ / 2. / math.tan(math.radians(fov_ / 2.))  # 注意，此处不是height_ / 2. / math.tan(math.radians(fov_ / 2.))，因为使用的是pinhole相机模型
        return [camModel_, width_, height_, fx_, fy_, cx_, cy_]
    else:
        raise Exception


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


def write_intrinsic_params(intrinsicMatrixs_, outputDir_, camModel_, fileName_="cameras.txt"):
    intrinsicMatrixsNum_ = len(intrinsicMatrixs_)

    if not os.path.exists(outputDir_):
        os.makedirs(outputDir_, exist_ok=True)

    outputPath_ = os.path.join(outputDir_, fileName_)

    cameras_ = dict()
    for inMIdx_, inM_ in enumerate(intrinsicMatrixs_):
        cameras_[inMIdx_+1] = Camera(id=inMIdx_+1, model=camModel_, width=inM_[0], height=inM_[1], params=inM_[2:])

    write_cameras_text(cameras_, outputPath_)


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


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


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

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=True WHERE camera_id=?",
            (model, width, height, array_to_blob(params), camera_id))
        return cursor.lastrowid


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


def set_write_permission(filePath_):
    # 检查文件权限
    fileStat_ = os.stat(filePath_)

    # 判断是否为只读（即没有写入权限）
    if not fileStat_.st_mode & stat.S_IWUSR & stat.S_IRUSR & stat.S_IRGRP & stat.S_IROTH:
        # 赋予写入权限
        os.chmod(filePath_, fileStat_.st_mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        print(f"{filePath_} 已成功赋予写入权限。")
    else:
        pass


def add_write_permission_to_files(directory_):
    # 遍历指定目录
    for root_, dirs_, files_ in os.walk(directory_):
        # 更改当前目录的权限
        currentPermissions_ = stat.S_IMODE(os.stat(root_).st_mode)
        os.chmod(root_, currentPermissions_ | stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 777 权限
        for file_ in files_:
            filePath_ = os.path.join(root_, file_)
            # 获取当前文件的权限
            currentPermissions_ = stat.S_IMODE(os.stat(filePath_).st_mode)
            # 添加写入权限
            newPermissions_ = currentPermissions_ | stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH  # 给拥有者添加写入权限
            os.chmod(filePath_, newPermissions_)  # 修改权限
        for dir_ in dirs_:
            dirPath_ = os.path.join(root_, dir_)
            # 获取当前文件的权限
            currentPermissions_ = stat.S_IMODE(os.stat(dirPath_).st_mode)
            # 添加写入权限
            newPermissions_ = currentPermissions_ | stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH  # 给拥有者添加写入权限
            os.chmod(dirPath_, newPermissions_)  # 修改权限


def add_write_permission_to_files_chmod(directory_):
    try:
        # 使用 subprocess 调用 chmod 命令，不带通配符
        subprocess.run(['chmod', '-R', '777', directory_], check=True)
        print(f"Permissions changed to 777 for all files and directories in {directory_}.")
    except subprocess.CalledProcessError as e_:
        print(f"Error occurred: {e_}")


def remove_directory_chmod(directory_):
    try:
        if os.path.exists(directory_) and os.path.isdir(directory_):
            shutil.rmtree(directory_)  # 删除整个文件夹及其内容
            print(f"'{directory_}' have beed removed. ")
    except subprocess.CalledProcessError as e_:
        print(f"Error occurred: {e_}")


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
