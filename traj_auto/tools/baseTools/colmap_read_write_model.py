# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import os
import collections
import numpy as np
import struct
import argparse

import sqlite3
import sys
from pose_process import qvec2rotmat, rotmat2qvec, pose_inverse, pose7_2_poseM


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


def read_next_bytes(fid_, numBytes_, formatCharSequence_, endianCharacter_="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid_:
    :param numBytes_: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param formatCharSequence_: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endianCharacter_: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data_ = fid_.read(numBytes_)
    return struct.unpack(endianCharacter_ + formatCharSequence_, data_)


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


def read_cameras_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params,
                )
    return cameras


def read_cameras_binary(pathToModelFile_):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(pathToModelFile_, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, numBytes_=24, formatCharSequence_="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                numBytes_=8 * num_params,
                formatCharSequence_="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


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


def write_cameras_binary(cameras_, pathToModelFile_):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(pathToModelFile_, "wb") as fid_:
        write_next_bytes(fid_, len(cameras_), "Q")
        for _, cam_ in cameras_.items():
            modelId_ = CAMERA_MODEL_NAMES[cam_.model].model_id
            cameraProperties_ = [cam_.id, modelId_, cam_.width, cam_.height]
            write_next_bytes(fid_, cameraProperties_, "iiQQ")
            for p_ in cam_.params:
                write_next_bytes(fid_, float(p_), "d")
    return cameras_


def read_images_text(path_):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images_ = {}
    with open(path_, "r") as fid_:
        while True:
            line_ = fid_.readline()
            if not line_:
                break
            line_ = line_.strip()
            if len(line_) > 0 and line_[0] != "#":
                elems_ = line_.split()
                imageId_ = int(elems_[0])
                qvec_ = np.array(tuple(map(float, elems_[1:5])))
                tvec_ = np.array(tuple(map(float, elems_[5:8])))
                cameraId_ = int(elems_[8])
                imageName_ = elems_[9]
                elems_ = fid_.readline().split()
                xys_ = np.column_stack(
                    [
                        tuple(map(float, elems_[0::3])),
                        tuple(map(float, elems_[1::3])),
                    ]
                )
                point3DIds_ = np.array(tuple(map(int, elems_[2::3])))
                images_[imageId_] = Image(
                    id=imageId_,
                    qvec=qvec_,
                    tvec=tvec_,
                    camera_id=cameraId_,
                    name=imageName_,
                    xys=xys_,
                    point3D_ids=point3DIds_,
                )
    return images_


def read_images_binary(pathToModelFile_):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(pathToModelFile_, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, numBytes_=64, formatCharSequence_="idddddddi"
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
                fid, numBytes_=8, formatCharSequence_="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                fid,
                numBytes_=24 * num_points2D,
                formatCharSequence_="ddq" * num_points2D,
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


def read_points3D_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3DId_ = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3DId_] = Point3D(
                    id=point3DId_,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs,
                )
    return points3D


def read_points3D_binary(pathToModelFile_):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D_ = {}
    with open(pathToModelFile_, "rb") as fid_:
        numPoints_ = read_next_bytes(fid_, 8, "Q")[0]
        for _ in range(numPoints_):
            binaryPointLineProperties_ = read_next_bytes(
                fid_, numBytes_=43, formatCharSequence_="QdddBBBd"
            )
            point3DId_ = binaryPointLineProperties_[0]
            xyz_ = np.array(binaryPointLineProperties_[1:4])
            rgb_ = np.array(binaryPointLineProperties_[4:7])
            error_ = np.array(binaryPointLineProperties_[7])
            trackLength_ = read_next_bytes(
                fid_, numBytes_=8, formatCharSequence_="Q"
            )[0]
            trackElems_ = read_next_bytes(
                fid_,
                numBytes_=8 * trackLength_,
                formatCharSequence_="ii" * trackLength_,
            )
            imageIds_ = np.array(tuple(map(int, trackElems_[0::2])))
            point2DIdxs_ = np.array(tuple(map(int, trackElems_[1::2])))
            points3D_[point3DId_] = Point3D(
                id=point3DId_,
                xyz=xyz_,
                rgb=rgb_,
                error=error_,
                image_ids=imageIds_,
                point2D_idxs=point2DIdxs_,
            )
    return points3D_


def write_points3D_text(points3D, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum(
            (len(pt.image_ids) for _, pt in points3D.items())
        ) / len(points3D)
    HEADER = (
        "# 3D point list with one line of data per point:\n"
        + "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        + "# Number of points: {}, mean track length: {}\n".format(
            len(points3D), mean_track_length
        )
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, pt in points3D.items():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")


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


def detect_model_format(path, ext):
    if (
        os.path.isfile(os.path.join(path, "cameras" + ext))
        and os.path.isfile(os.path.join(path, "images" + ext))
        and os.path.isfile(os.path.join(path, "points3D" + ext))
    ):
        print("Detected model format: '" + ext + "'")
        return True

    return False


def read_model(path, ext=""):
    # try to detect the extension automatically
    if ext == "":
        if detect_model_format(path, ".bin"):
            ext = ".bin"
        elif detect_model_format(path, ".txt"):
            ext = ".txt"
        else:
            print("Provide model format: '.bin' or '.txt'")
            return

    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3D_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def write_model(cameras, images, points3D, path, ext=".bin"):
    if ext == ".txt":
        write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
        write_images_text(images, os.path.join(path, "images" + ext))
        write_points3D_text(points3D, os.path.join(path, "points3D") + ext)
    else:
        write_cameras_binary(cameras, os.path.join(path, "cameras" + ext))
        write_images_binary(images, os.path.join(path, "images" + ext))
        write_points3D_binary(points3D, os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


# ======================================================= new ==========================================================
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


def remove_images_based_fileNames(inputPath_, removeNames_, outputPath_=None):
    fileName_ = os.path.basename(inputPath_)
    ext_ = fileName_.split('.')[-1]

    images_ = read_images_binary(inputPath_) if ext_ == 'bin' else read_images_text(inputPath_)

    for imageKey_, imageValue_ in images_.items():
        imageNameFull_ = imageValue_.name
        imageName_ = imageNameFull_.split('.')[0]
        if imageName_ in removeNames_:
            del images_[imageKey_]

    outputPathTmp_ = outputPath_ if outputPath_ is not None else inputPath_
    extO_ = os.path.basename(outputPathTmp_).split('.')[-1]
    if extO_ == 'bin':
        write_images_binary(images_, outputPathTmp_)
    else:
        write_images_text(images_, outputPathTmp_)


# ======================================================================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Read and write COLMAP binary and text models"
    )
    parser.add_argument("--input_model", help="path to input model folder")
    parser.add_argument(
        "--input_format",
        choices=[".bin", ".txt"],
        help="input model format",
        default="",
    )
    parser.add_argument("--output_model", help="path to output model folder")
    parser.add_argument(
        "--output_format",
        choices=[".bin", ".txt"],
        help="outut model format",
        default=".txt",
    )
    args = parser.parse_args()

    cameras, images, points3D = read_model(
        path=args.input_model, ext=args.input_format
    )

    print("num_cameras:", len(cameras))
    print("num_images:", len(images))
    print("num_points3D:", len(points3D))

    if args.output_model is not None:
        write_model(
            cameras,
            images,
            points3D,
            path=args.output_model,
            ext=args.output_format,
        )


if __name__ == "__main__":
    main()
