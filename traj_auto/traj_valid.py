import numpy as np
import os
import copy

import sys
sys.path.append("./pcp_tools")

from pcp_tools.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary

from pcp_tools.dataset_readers_sim import sceneLoadTypeCallbacks
from pcp_tools.camera_utils import cameraList_from_camInfos_sim
from read_write_model import rotmat2qvec

import tools.base_tools as btls


def pos_to_rotation(position_vector, look_at):
    # Z-axis (camera is pointing in the positive Z direction)
    z_axis = -(position_vector - look_at).copy()
    # z_axis = position_vector.copy()
    z_axis /= np.linalg.norm(z_axis)  # Normalize
    # Y-axis (up direction)
    y_axis = look_at * -1
    # X-axis (right direction)
    x_axis = np.cross(z_axis, y_axis)
    x_axis /= np.linalg.norm(x_axis)
    # Recompute the Y-axis to ensure orthogonality
    y_axis = np.cross(z_axis, x_axis)

    # Assemble the rotation matrix
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))

    return rotation_matrix


def generate_circle_trajectory(camera_infos_):
    # H_表示y轴到重建目标主轴之间的变化，新的生成圆形轨迹的方式，非常好用，不需要考虑欧拉角
    # atest = np.pi / 4
    # ctest = np.cos(atest)
    # stest = np.sin(atest)
    # # 绕x轴顺时针旋转
    # H_ = np.array([[1, 0, 0],
    #                [0, ctest, -stest],
    #                [0, stest, ctest]])
    H_ = np.array([[1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]])

    view_idx = 1
    view = copy.deepcopy(camera_infos_[view_idx])
    reference_position = view.T.squeeze()

    num_steps = 30
    angle_step = 2 * np.pi / num_steps
    trajectory = []
    for step in range(num_steps):
        angle = step * angle_step

        radius = np.linalg.norm(reference_position)

        dx = radius * np.cos(angle)
        dz = radius * np.sin(angle)
        # dy = radius * 0.1 * np.cos(angle + np.pi) * 0
        dy = -5

        C_ = np.array([dx, dy, dz])
        look_at_ = np.array([0, 1, 0])

        C = H_ @ C_
        look_at = H_ @ look_at_
        rotation_matrix = pos_to_rotation(C, look_at)

        rotation_matrix = rotation_matrix.T
        translation = - rotation_matrix @ C

        trajectory.append((rotation_matrix, translation))
    return trajectory


def generate_leave_trajectory(camera_infos_):
    view_idx = 1
    view = copy.deepcopy(camera_infos_[view_idx])

    # 沿着y轴行进一段距离，只改变平移量，而不改变相机姿态
    num_steps = 15
    trajectory = []
    Rp_, T_ = view.R, view.T
    T_ = -Rp_ @ T_
    for step in range(num_steps):
        T_ = T_ * np.array([1, 1, 1.05])
        Rtmp_ = Rp_.T
        Ttmp_ = -Rtmp_@T_
        trajectory.append((Rtmp_, Ttmp_.copy()))
    return trajectory


def generate_test_trajectory(camera_info):
    viewNum = 10
    trajectory = []

    for idx in range(viewNum):
        cam = camera_info[idx]
        view = np.eye(4)
        view[:3] = np.concatenate([cam.R.T, cam.T[:, None]], 1)
        view = np.linalg.inv(view)
        view[:, 1:3] *= -1
        view = np.linalg.inv(view)
        trajectory.append((view[:3, :3], view[:3, 3]))
    return trajectory


if __name__ == '__main__':
    # traj type
    traj_name = 'leave'

    # sparse reconstruction info(read in cfg_args.txt)
    source_path = "/home/hongqingde/devdata/workspace_gitmp/input/board_new"
    images = "images"
    eval = False
    # read scene info(include camera params and point) of each frame
    assert os.path.exists(source_path), "path is not exist"
    scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, images, eval)

    train_cameras = cameraList_from_camInfos_sim(scene_info.train_cameras)
    test_cameras = cameraList_from_camInfos_sim(scene_info.test_cameras)
    cameras_extent = scene_info.nerf_normalization['radius']

    cam_infos = train_cameras if not eval else test_cameras

    if traj_name == 'circle':
        poses_valid_ = generate_circle_trajectory(cam_infos)
    elif traj_name == 'leave':
        poses_valid_ = generate_leave_trajectory(cam_infos)
    elif traj_name == 'test':
        poses_valid_ = generate_test_trajectory(cam_infos)
    else:
        poses_valid_ = []

    # circle -- rotation(相机坐标系), translation(相机坐标系)
    # leave  -- rotation(相机坐标系)， translation(相机坐标系)
    poses_valid_list = []
    for traj_pose in poses_valid_:
        R, T = traj_pose
        qv = rotmat2qvec(R).tolist()
        tv = T.tolist()
        poses_valid_list.append(qv+tv)

    srcPath = os.path.join(source_path, "sparse/0", "images.bin")
    dstPath = os.path.join("/home/hongqingde/devdata/workspace_gitmp/input/board_new/traj_test", "images.bin")
    btls.images_bin_write(srcPath, poses_valid_list, dstPath)

