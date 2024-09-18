import cv2
import open3d as o3d
import os
import os.path
import numpy as np
import read_write_binary
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import random
import torch
import torch.nn.functional as F
import pycolmap
import math


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def resize_torch(width, height, width_, height_, pre_depth_torch):
    pad_height = height - height_
    pad_width = width - width_

    # 使用 pad 函数对 pre_depth_torch 进行填充
    pre_depth_torch_padded = F.pad(pre_depth_torch, (0, pad_width, 0, pad_height))
    return pre_depth_torch_padded


def get_camera_params(camera):
    if camera.model == 'CameraModelId.SIMPLE_RADIAL' or camera.model == 'CameraModelId.SIMPLE_PINHOLE':
        camera_fx = camera.params[0]
        camera_fy = camera.params[0]
        camera_cx = camera.params[1]
        camera_cy = camera.params[2]
    elif camera.model == 'CameraModelId.PINHOLE':
        camera_fx = camera.params[0]
        camera_fy = camera.params[1]
        camera_cx = camera.params[2]
        camera_cy = camera.params[3]
    else:
        raise ValueError('Unsupported camera model: {}'.format(camera.model))
    return camera_fx, camera_fy, camera_cx, camera_cy


def voxel_texture_filter(cloud, sobel_scores, leaf_size):
    point_cloud = np.asarray(cloud.points)  # N 3
    # colors = np.asarray(cloud.colors) / 255.0 # N 3
    normals = np.asarray(cloud.normals)
    # 1、计算边界点
    x_min, y_min, z_min = np.amin(point_cloud, axis=0)
    x_max, y_max, z_max = np.amax(point_cloud, axis=0)
    # 2、计算每个维度上体素格网的个数
    Dx = (x_max - x_min) // leaf_size + 1
    Dy = (y_max - y_min) // leaf_size + 1
    Dz = (z_max - z_min) // leaf_size + 1
    print("Dx * Dy * Dz is {} * {} * {}".format(Dx, Dy, Dz))
    # 3、计算每个3D点的格网idx
    h = list()
    for i in range(len(point_cloud)):
        hx = (point_cloud[i][0] - x_min) // leaf_size
        hy = (point_cloud[i][1] - y_min) // leaf_size
        hz = (point_cloud[i][2] - z_min) // leaf_size
        h.append(hx + hy * Dx + hz * Dx * Dy)
    h = np.array(h)

    # 4、根据纹理度量保留3D点
    h_indice = np.argsort(h)
    h_sorted = h[h_indice]
    #################################
    # h         9 1 7 1 1     每个3D点的格网idx，其位置也为对应3D点在点云中的位置索引
    # h_indice  1 3 4 2 0     升序排列时，每个3D点的格网idx在h中的索引
    # h_sorted  1 1 1 7 9     升序排列后的 每个3D点的格网idx
    #################################
    begin = 0
    mean_normals = np.mean(np.std(normals, axis=0))
    std_normals = np.std(np.std(normals, axis=0))   # np.std(normals, axis=0) 1, 3 每个3D点的法线向量在xyz维度的标准差，该区域的法向量分布较为离散,即表面法线变化剧烈,说明纹理丰富
    mean_sobel_scores = np.mean(sobel_scores)
    std_sobel_scores = np.std(sobel_scores)
    point_idx_scores = {}
    texture_scores = []
    for i in range(len(h_indice) - 1):
        # 当前3D点和后一个3D点的格网idx 相同，则跳过
        if h_sorted[i] == h_sorted[i + 1]:
            continue
        # 当前3D点和后一个3D点的格网idx 不相同，则在当前3D点所在格网内
        else:
            # begin: 在同一个格网内的第一个3D点的格网idx的 在h_indice的位置，i：在同一个格网内的最后一个3D点的格网idx 在h_indice的位置
            point_idx = h_indice[begin: i + 1]  # 同一格网内 所有3D点的格网idx在h中的位置，也是这些3D点在点云中的位置

            # 计算每个格网纹理度量得分
            normals_in_voxel = normals[point_idx]
            sobel_scores_in_voxel = sobel_scores[point_idx]
            avg_sobel_score = (np.mean(sobel_scores_in_voxel) - mean_sobel_scores) / std_sobel_scores
            avg_normal_score = (np.mean(np.std(normals_in_voxel, axis=0)) - mean_normals) / std_normals
            texture_score = 0.5 * avg_sobel_score + 0.5 * avg_normal_score
            texture_scores.append(texture_score)

            point_idx_scores[h[h_indice[begin]]] = [point_idx, texture_score]
            begin = i + 1

    # 计算所有包含3D点云的格网的纹理度量得分
    mean_texture_score = np.mean(texture_scores)
    std_texture_score = np.std(texture_scores)
    max_texture_score = np.max(texture_scores)
    min_texture_score = np.min(texture_scores)

    ori_point_idx = []
    random_idx = []
    random_idx_texture = []
    # 根据每个格网与所有格网的纹理度量得分比值，确定该格网内保留的点云数量
    for point_idx, texture_score in point_idx_scores.values():
        ori_point_idx.append(point_idx)

        random_idx.append(random.choice(point_idx))

        texture_weight = (texture_score - min_texture_score) / (max_texture_score - min_texture_score)
        num_points = max(1, int(len(point_idx) * texture_weight))
        # num_points = max(1, np.clip(int(len(point_idx) * texture_weight), 0, len(point_idx)//2)) # 该clip方案实际并不会影响结果，因为计算的保留点数结果 < 总个数的一半
        random_idx_texture.extend(random.sample(list(point_idx), num_points))  # 在同一格网内 随机选取num_points个3D点

    print("ori_point_idx: {}, random_idx: {}, random_idx_texture: {}".format(len(ori_point_idx), len(random_idx), len(random_idx_texture)))

    return random_idx, random_idx_texture

    # filtered_points = (cloud.select_by_index(random_idx))
    # filtered_points_texture = (cloud.select_by_index(random_idx_texture))
    # return filtered_points, filtered_points_texture


def voxel_random_filter(cloud, leaf_size):
    point_cloud = np.asarray(cloud.points)  # N 3
    # 1、计算边界点
    x_min, y_min, z_min = np.amin(point_cloud, axis=0)  # 按列寻找点云位置的最小值
    x_max, y_max, z_max = np.amax(point_cloud, axis=0)
    # 2、计算每个维度上体素格网的个数
    Dx = (x_max - x_min) // leaf_size + 1
    Dy = (y_max - y_min) // leaf_size + 1
    Dz = (z_max - z_min) // leaf_size + 1
    print("Dx * Dy * Dz is {} * {} * {}".format(Dx, Dy, Dz))
    # 3、计算每个点的格网idx
    h = list()
    for i in range(len(point_cloud)):
        # 分别在x, y, z方向上格网的idx
        hx = (point_cloud[i][0] - x_min) // leaf_size
        hy = (point_cloud[i][1] - y_min) // leaf_size
        hz = (point_cloud[i][2] - z_min) // leaf_size
        h.append(hx + hy * Dx + hz * Dx * Dy)   # 该点所在格网 映射到1D的idx
    h = np.array(h)

    # 4、体素格网内随机筛选点
    h_indice = np.argsort(h)
    h_sorted = h[h_indice]
    #################################
    # h         3 1 2 1     每个3D点的格网idx，其位置也为对应3D点在点云中的位置索引
    # h_indice  1 3 2 0     升序排列时，每个3D点的格网idx在h中的索引
    # h_sorted  1 1 2 3     升序排列后的 每个3D点的格网idx
    #################################
    random_idx = []
    begin = 0
    # 遍历每个3D点的格网idx
    for i in range(len(h_sorted) - 1):
        # 当前3D点和后一个3D点的格网idx 相同，则跳过
        if h_sorted[i] == h_sorted[i + 1]:
            continue
        # 当前3D点和后一个3D点的格网idx 不相同，则在当前3D点所在格网内随机选择一个3D点
        else:
            # begin: 在同一个格网内的第一个3D点的格网idx的 在h_sorted/h_indice的位置，i：在同一个格网内的最后一个3D点的格网idx 在h_sorted/h_indice的位置
            point_idx = h_indice[begin: i + 1]  # 同一格网内 所有3D点的格网idx在h中的位置，也是这些3D点在点云中的索引
            random_idx.append(random.choice(point_idx)) # 在同一格网内 随机选择一个3D点
            begin = i + 1
    filtered_points = (cloud.select_by_index(random_idx))
    return filtered_points


def create_gt_points3D_by_depth(train_dir, num1=40, num2=4):
    images_dir = f"{train_dir}/images"
    depth_dir = f"{train_dir}/depth"
    pose_path = os.path.join(train_dir, "gt_as_col_pose.txt")
    # calar
    # K = np.array([[554.256, 0, 960],
    #               [0, 554.256, 540],
    #               [0, 0, 1]])
    # K = np.array([[554.2562584220407, 0, 960],
    #               [0, 554.2562584220407, 540],
    #               [0, 0, 1]])
    K = np.array([[555.4534568, 0, 960],
                  [0, 555.4534568, 540],
                  [0, 0, 1]])
    dc2w = load_tum_file(pose_path) # gt pose (Twc)
    colors = []
    point3ds = []
    normals = []
    sobel_scores = []

    pbar = tqdm(total=len(dc2w))
    num_1  = 0
    num_2 = 0
    # 遍历每张图片
    for id, c2w in dc2w.items():
        pbar.update(1)

        depth_path = os.path.join(depth_dir, id + ".npy")
        depth = np.load(depth_path)
        print("max depth = {}, min depth = {}".format(np.max(depth), np.min(depth)))

        img_path = os.path.join(images_dir, id + ".png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        h, w = depth.shape

        w2c = np.linalg.inv(c2w)

        # 当前像素点对应的3D点，由其深度 转换为 相机坐标系下的3D点（depth_scale 根据深度图和位姿的尺度进行调整）
        camera_xyz_map = depth2xyz(depth, K, depth_scale=1.0)   # H W 3，xyz

        Z = 10
        # 遍历每个像素点，每100个选1个
        for i in range(0, h, num1 if Z >= 1500 else num2):
            for j in range(0, w, num1 if Z >= 1500 else num2):
                x = num1 if Z >= 1500 else num2
                if x == num1:
                    num_1 += 1
                elif x == num2:
                    num_2 += 1

                y = i
                x = j
                Z = depth[i, j] # 对应的绝对深度值

                color = img[y, x]
                colors.append(color)

                xyz = camera_xyz_map[y, x]  # 取出相机坐标系下 对应像素的 点云 Pc
                point3d_gt = pc_cam_to_pc_world(xyz, w2c)  # 相机下的点云转换为世界坐标系下的点云 Pw
                point3ds.append(point3d_gt)

                # 使用法线表征纹理的丰富度
                neighbors = camera_xyz_map[max(y - 1, 0):min(y + 2, h), max(x - 1, 0):min(x + 2, w)]    #  取出以当前3D点为中心的3x3邻居点
                mean, eigenvector, _ = cv2.PCACompute2(neighbors.reshape(-1, 3), mean=None)
                normal = mean   # 1 3
                normals.append(normal)

                # 当前像素的Sobel梯度作为纹理的度量
                sobel_x_i = sobel_x[y, x, :]
                sobel_y_i = sobel_y[y, x, :]
                sobel_score = np.linalg.norm(sobel_x_i) + np.linalg.norm(sobel_y_i)
                # sobel_score = np.sqrt(sobel_x_i ** 2 + sobel_y_i ** 2)
                sobel_scores.append(sobel_score)

    # 保存点云
    print("\n{}, {}".format(num_1, num_2))
    point3ds = np.array(point3ds)   # N 3
    colors = np.array(colors)       # N 3
    normals = np.array(normals).squeeze(axis=1)     # N 3
    sobel_scores = np.array(sobel_scores)

    points = np.hstack([np.asarray(point3ds), np.asarray(colors)])
    output_path = os.path.join(train_dir, "gt_sparse_1/0/points3D_gt.txt")
    np.savetxt(output_path, points, fmt="%.6f %.6f %.6f %.6f %.6f %.6f")

    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(point3ds)
    # point_cloud.colors = o3d.utility.Vector3dVector(colors)
    # point_cloud.normals = o3d.utility.Vector3dVector(normals)
    # print(f"src num {len(point_cloud.points)}")
    #
    # # pcd_down = voxel_random_filter(point_cloud, 0.2)
    # random_idx, random_idx_texture = voxel_texture_filter(point_cloud, sobel_scores, 0.2)
    # pcd_down = point_cloud.select_by_index(random_idx)
    # pcd_down_texture = point_cloud.select_by_index(random_idx_texture)
    # print(f"dst num1 {len(pcd_down.points)}, dst num2 {len(pcd_down_texture.points)}")
    #
    # points = np.hstack([np.asarray(pcd_down.points), np.asarray(pcd_down.colors)])
    # output_path = os.path.join(train_dir, "gt_sparse_1/0/points3D_gt_4_02.txt")
    # np.savetxt(output_path, points, fmt="%.6f %.6f %.6f %.6f %.6f %.6f")
    #
    # points = np.hstack([np.asarray(pcd_down_texture.points), np.asarray(pcd_down_texture.colors)])
    # output_path = os.path.join(train_dir, "gt_sparse_1/0/points3D_gt_4_02_texture.txt")
    # np.savetxt(output_path, points, fmt="%.6f %.6f %.6f %.6f %.6f %.6f")

    # output_path = f"{train_dir}/gt_sparse_1/0/points3D_gt.ply"
    # o3d.io.write_point_cloud(output_path, pcd_down)


def create_points3D_by_depth(train_dir, num1=40, num2=3):
    # train_dir = "/mnt/data/data/data/pure/building2/train"
    images_dir = f"{train_dir}/images"
    depth_dir = f"{train_dir}/mvs/stereo/depth_maps"
    # depth_dir = f"/media/liuzhi/b4608ade-d2e0-430d-a40b-f29a8b22cb8c/3DGS_code/PENet_3dgs/results/val_output"
    pose_path = os.path.join(train_dir, "colmap_pose.txt")

    # reconstruction = pycolmap.Reconstruction(f'{train_dir}/sparse')
    # camera = next(iter(reconstruction.cameras.values()))
    # camera_fx, camera_fy, camera_cx, camera_cy = get_camera_params(camera)
    # calar
    # K = np.array([[554.0724944218656, 0, 960],
    #               [0, 554.0724944218656, 540],
    #               [0, 0, 1]])
    #
    dc2w = load_tum_file(pose_path) # gt pose (Twc)

    colors = []
    point3ds = []
    normals = []
    sobel_scores = []

    pbar = tqdm(total=len(dc2w))
    num_1 = 0
    num_2 = 0

    # 遍历每张图片
    for id, c2w in dc2w.items():
        pbar.update(1)

        # depth_path = os.path.join(depth_dir, id + ".npy")
        # depth = np.load(depth_path)
        col_depthf = os.path.join(depth_dir, id + ".png.geometric.bin")
        if not os.path.exists(col_depthf):
            print("skip visualize for the file does not exists: {}".format(col_depthf))
            pass
        pre_depth_np = read_array(col_depthf)
        target_height, target_width = 1080, 1920
        depth_height, depth_width = pre_depth_np.shape[:2]
        pad_height = target_height - depth_height
        pad_width = target_width - depth_width
        depth = np.pad(pre_depth_np, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
        # print("{}, {} \t{}, {}".format(depth_height, depth_width, depth.shape[0], depth.shape[1]))

        # pre_depth_np = read_array(col_depthf)
        # pre_depth_torch = torch.from_numpy(pre_depth_np).unsqueeze(0)   # 1 1078 1918
        # pre_depth_torch_padded = resize_torch(1080, 1920, pre_depth_torch.shape[1], pre_depth_torch.shape[2], pre_depth_torch)
        # depth = np.squeeze(pre_depth_torch_padded.numpy())
        # print("{}, {} \t{}, {}".format(pre_depth_torch.shape[1], pre_depth_torch.shape[2], pre_depth_torch_padded.shape[1], pre_depth_torch_padded.shape[2]))

        img_path = os.path.join(images_dir, id + ".png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        h, w = depth.shape[:2]


        w2c = np.linalg.inv(c2w)

        # 当前像素点对应的3D点，由其深度 转换为 相机坐标系下的3D点（depth_scale 根据深度图和位姿的尺度进行调整）
        camera_xyz_map = depth2xyz(depth, K, depth_scale=1.0)  # H W 3，xyz

        Z = 10
        # 遍历每个像素点，每100个选1个
        for i in range(0, h, num1 if Z >= 999 else num2):
            for j in range(0, w, num1 if Z >= 999 else num2):
                x = num1 if Z >= 999 else num2
                if x == num1:
                    num_1 += 1
                elif x == num2:
                    num_2 += 1

                y = i
                x = j
                Z = depth[i, j]  # 对应的绝对深度值

                color = img[y, x]
                colors.append(color)

                xyz = camera_xyz_map[y, x]  # 取出相机坐标系下 对应像素的 点云 Pc
                point3d_gt = pc_cam_to_pc_world(xyz, w2c)  # 相机下的点云转换为世界坐标系下的点云 Pw
                point3ds.append(point3d_gt)

                # 使用法线表征纹理的丰富度
                neighbors = camera_xyz_map[max(y - 1, 0):min(y + 2, h),
                            max(x - 1, 0):min(x + 2, w)]  # 取出以当前3D点为中心的3x3邻居点
                mean, eigenvector, _ = cv2.PCACompute2(neighbors.reshape(-1, 3), mean=None)
                normal = mean  # 1 3
                normals.append(normal)

                # 当前像素的Sobel梯度作为纹理的度量
                sobel_x_i = sobel_x[y, x, :]
                sobel_y_i = sobel_y[y, x, :]
                sobel_score = np.linalg.norm(sobel_x_i) + np.linalg.norm(sobel_y_i)
                # sobel_score = np.sqrt(sobel_x_i ** 2 + sobel_y_i ** 2)
                sobel_scores.append(sobel_score)

    # 保存点云
    print("\n{}, {}".format(num_1, num_2))
    point3ds = np.array(point3ds)  # N 3
    colors = np.array(colors)  # N 3
    normals = np.array(normals).squeeze(axis=1)  # N 3
    sobel_scores = np.array(sobel_scores)

    points = np.hstack([np.asarray(point3ds), np.asarray(colors)])
    output_path = os.path.join(train_dir, "points3D_sp_1.txt")
    np.savetxt(output_path, points, fmt="%.6f %.6f %.6f %.6f %.6f %.6f")


def pose7_to_matrix44(pose):
    # pose：Twc的 Tx, Ty, Tz, Qx, Qy, Qz, Qw
    translation = pose[:3]
    quaternion = pose[3:]
    rotation = R.from_quat([quaternion[0], quaternion[1], quaternion[2], quaternion[3]]).as_matrix()
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation
    pose_matrix[:3, 3] = translation

    return pose_matrix


def qvec2rotmat(qvec):
    # 四元数qvec=[w, x, y, z] 转 旋转矩阵
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def load_tum_file(file):
    lines = open(file).readlines()
    d = {}
    # all_pose = []
    for line in lines:
        ws = line.strip().split(' ')
        # Twc的 Tx, Ty, Tz, Qx, Qy, Qz, Qw
        pose7 = np.array([float(ws[1]), float(ws[2]), float(ws[3]), float(ws[4]), float(ws[5]), float(ws[6]),  float(ws[7])])
        mat44 = pose7_to_matrix44(pose7)

        d[ws[0]] = mat44
        # all_pose.append(mat44)
    return d


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def read_colmap_camera(intrinsics, extrinsics, width_real=1920, height_real=1080):
    poses = {}
    positions = {}
    for idx, key in enumerate(extrinsics):
        # 获取当前相机的外参和内参
        extr = extrinsics[key]  # 当前相机的外参类Imgae对象
        intr = intrinsics[extr.camera_id]  # 根据外参中的camera_id找到对应的内参类对象
        height = intr.height
        width = intr.width
        # 这里的内参是原图大小估计的内参，而读取的图片是下采样后的，所以需根据视场角计算实际的内参
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)

        fx = fov2focal(FovX, width_real)
        fy = fov2focal(FovY, height_real)
        cx = width_real / 2
        cy = height_real / 2
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])

        uid = intr.id  # 相机的唯一标识符

        # R_cw = qvec2rotmat(extr.qvec)
        # t_cw = np.array(extr.tvec)
        # T_cw = np.eye(4)
        # T_cw[:3, :3] = R_cw
        # T_cw[:3, 3] = t_cw

        # Tcw的 Tx, Ty, Tz, Qx, Qy, Qz, Qw
        pose7 = np.array([float(extr.tvec[0]), float(extr.tvec[1]), float(extr.tvec[2]), float(extr.qvec[1]), float(extr.qvec[2]), float(extr.qvec[3]), float(extr.qvec[0])])
        mat44 = pose7_to_matrix44(pose7)
        T_wc = np.linalg.inv(mat44)
        t_cw = np.array([float(extr.tvec[0]), float(extr.tvec[1]), float(extr.tvec[2])])

        poses[extr.name] = [K, T_wc]
        positions[extr.name] = t_cw
    return poses, positions


def depth2xyz(depth_map, K, flatten=False, depth_scale=1.0):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h, w = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]
    z = depth_map / depth_scale
    x = (w - cx) * z / fx
    y = (h - cy) * z / fy
    xyz = np.dstack((x, y, z)) if flatten == False else np.dstack((x, y, z)).reshape(-1, 3)
    # xyz=cv2.rgbd.depthTo3d(depth_map,depth_cam_matrix)
    return xyz


def pc_cam_to_pc_world(pc, extrinsic):
    """
        pc          相机坐标系下的一个点云 1, 3
        extrinsic   相机位姿，Tcw世界到相机 (4, 4)
    """
    extr_inv = np.linalg.inv(extrinsic)  # Twc
    R = extr_inv[:3, :3]
    T = extr_inv[:3, 3]
    pc = (R @ pc.T).T + T   # Rwc * Pc + Twc = Pw
    return pc


def save_sparse_as_ply():
    sfm_dir = "/mnt/data/data/bk/town2_2024_05_27/sparse"
    sfm = read_write_binary.read_model(sfm_dir)
    p3ds = []
    for id, pt in sfm[2].items():
        p3ds.append(pt.xyz)

    p3ds = np.array(p3ds)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(p3ds)

    output_path_limit2 = f"/mnt/data/data/bk/town2_2024_05_27/point_cloud/iteration_30000/sfm.ply"
    o3d.io.write_point_cloud(output_path_limit2, point_cloud)


# 半径离群点剔除
def remove_outliers_radius(pcd, radius=0.05, min_neighbors=5):
    cl, ind = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    inlier_cloud = pcd.select_by_index(ind)
    return inlier_cloud


def pcu_demo():
    import point_cloud_utils as pcu
    sfm_pcu1 = np.asarray(inlier_sfm_pcd.points)
    gt_pcu1 = np.asarray(gt_pcd.points)
    chamfer_dist = pcu.chamfer_distance(sfm_pcu1, gt_pcu1)
    print(f"CD: {chamfer_dist}")

    hausdorff_a_to_b = pcu.one_sided_hausdorff_distance(sfm_pcu1, gt_pcu1)
    hausdorff_b_to_a = pcu.one_sided_hausdorff_distance(gt_pcu1, sfm_pcu1)
    hausdorff_dist = pcu.hausdorff_distance(sfm_pcu1, gt_pcu1)
    print(f"hausdorff_a_to_b: {hausdorff_a_to_b}")
    print(f"hausdorff_b_to_a: {hausdorff_b_to_a}")
    print(f"hausdorff_dist: {hausdorff_dist}")


def compute_nearest_neighbor_distances(pcd):
    """
    Compute the nearest neighbor distances for all points in a point cloud.

    Parameters:
    pcd (open3d.geometry.PointCloud): The input point cloud.

    Returns:
    np.ndarray: An array of nearest neighbor distances.
    """
    # 构建k-d树
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    distances = []
    for i in range(len(pcd.points)):
        # 查询第i个点的最近邻
        [_, idx, d] = kdtree.search_knn_vector_3d(pcd.points[i], 2)
        # 最近邻距离是d[1]，因为d[0]是该点自身的距离
        distances.append(np.sqrt(d[1]))

    return np.array(distances)


def compute_statistics(distances):
    """
    Compute the mean and median of the distances.

    Parameters:
    distances (np.ndarray): An array of distances.

    Returns:
    tuple: The mean and median of the distances.
    """
    mean_distance = np.mean(distances)
    median_distance = np.median(distances)
    return mean_distance, median_distance


def demo_eval_two_point_cloud():
    # 输入输出文件
    # sfm_ply_file = "/mnt/data/data/bk/town2_2024_05_27/point_cloud/iteration_30000/point_cloud.ply"
    sfm_ply_file = "/mnt/data/data/bk/town2_2024_05_27/point_cloud/iteration_30000/sfm.ply"
    gt_ply_file = "/mnt/data/data/bk/town2_2024_05_27/point_cloud/iteration_30000/pc200.ply"
    sfm_pcd = o3d.io.read_point_cloud(sfm_ply_file)
    gt_pcd = o3d.io.read_point_cloud(gt_ply_file)

    # 输入变换矩阵，是从gt转colmap的
    scale = 0.027
    r_a = [[-1.0, -0.004, -0.007],
           [-0.008, 0.693, 0.721],
           [0.002, 0.721, -0.693]]
    t_a = [4.919, -3.641, 4.673]
    scale = 1.0/(scale + 0.00000000001)

    # 先变换后进行尺度缩放
    # 推导colmap到gt的变换矩阵
    R = np.array(r_a)
    T = np.array(t_a)
    T44 = np.eye(4)
    T44[:3, :3] = R
    T44[:3, 3] = T
    T44_inv = np.linalg.inv(T44)
    sfm_pcd.transform(T44_inv)
    center = np.array([0, 0, 0], dtype=float)
    sfm_pcd.scale(scale, center)
    sfm_pcd.paint_uniform_color([0, 0, 1])

    # 先尺度缩放，再进行矩阵转换
    # center = np.array([0, 0, 0], dtype=float)
    # sfm_pcd.scale(scale, center)
    # R = np.array(r_a)
    # T = np.array(t_a)*scale
    # T44 = np.eye(4)
    # T44[:3, :3] = R
    # T44[:3, 3] = T
    # T44_inv = np.linalg.inv(T44)
    # sfm_pcd.transform(T44_inv)
    # sfm_pcd.paint_uniform_color([0, 0, 1])
    sfm2gt_dists = np.asarray(sfm_pcd.compute_point_cloud_distance(gt_pcd))
    gt2sfm_dists = np.asarray(gt_pcd.compute_point_cloud_distance(sfm_pcd))
    cd = np.mean(sfm2gt_dists) + np.mean(gt2sfm_dists)
    mean, median, min_value, max_value, std_dev = summary(sfm2gt_dists, "Points3d Dist Distribution", "points3d_dist_distribution.png", max_xlim=10, bins_num=100)
    # mean, median, min_value, max_value, std_dev = summary(gt2sfm_dists, "Points3d Dist Distribution", "points3d_dist_distribution.png", max_xlim=10, bins_num=100)
    # print("********** {}: ".format(title))
    print("mean: {:.3f}".format(mean))
    print("mid : {:.3f}".format(median))
    print("min : {:.3f}".format(min_value))
    print("max : {:.3f}".format(max_value))
    print("std : {:.3f}".format(std_dev))

    print(f"CD: {cd}")
    # 计算准确率
    p25 = np.sum(sfm2gt_dists < 0.25) / (len(sfm2gt_dists)+0.0000001)
    print(f"P(0.25): {p25}")
    p50 = np.sum(sfm2gt_dists < 0.50) / (len(sfm2gt_dists)+0.0000001)
    print(f"P(0.5): {p50}")
    p100 = np.sum(sfm2gt_dists < 1.00) / (len(sfm2gt_dists)+0.0000001)
    print(f"P(1.0): {p100}")
    p200 = np.sum(sfm2gt_dists < 2.00) / (len(sfm2gt_dists)+0.0000001)
    print(f"P(2.0): {p200}")
    p400 = np.sum(sfm2gt_dists < 4.00) / (len(sfm2gt_dists)+0.0000001)
    print(f"P(4.0): {p400}")
    # 计算离群点率
    # 计算召回率，也就是CR
    r100 = np.sum(gt2sfm_dists < 1) / (len(gt2sfm_dists)+0.0000001)
    print(f"R(1): {r100}")
    r200 = np.sum(gt2sfm_dists < 2) / (len(gt2sfm_dists)+0.0000001)
    print(f"R(2): {r200}")
    r400 = np.sum(gt2sfm_dists < 4) / (len(gt2sfm_dists)+0.0000001)
    print(f"R(4): {r400}")
    r800 = np.sum(gt2sfm_dists < 8) / (len(gt2sfm_dists)+0.0000001)
    print(f"R(8): {r800}")
    r1600 = np.sum(gt2sfm_dists < 16) / (len(gt2sfm_dists)+0.0000001)
    print(f"R(16): {r1600}")
    # o3d.visualization.draw_geometries([sfm_pcd, gt_pcd])

    # 计算hausdorff距离，意义不大
    # 剔除离群点后再计算
    # inlier_dists = compute_nearest_neighbor_distances(sfm_pcd)
    # # mean_distance = np.mean(inlier_dists)
    # inlier_median_distance = np.median(inlier_dists)
    # print(f"inlier_median_distance: {inlier_median_distance}")
    # inlier_sfm_pcd = remove_outliers_radius(sfm_pcd, radius=inlier_median_distance*5, min_neighbors=10)
    # o3d.visualization.draw_geometries([inlier_sfm_pcd, gt_pcd])
    # eval_two_point_cloud(inlier_sfm_pcd, gt_pcd)


if __name__ == '__main__':
    # train_dir = "/media/liuzhi/b4608ade-d2e0-430d-a40b-f29a8b22cb8c/Dataset/3DGS_Dataset/town-train"
    # create_points3D_by_depth(train_dir)
    train_dir = "/media/liuzhi/b4608ade-d2e0-430d-a40b-f29a8b22cb8c/Dataset/3DGS_Dataset/building_test_50"
    create_gt_points3D_by_depth(train_dir)

    # 远程数据集下：
    # points3D_gt: town 4_01, building 6_01
    # points3D_gt_: town 2_01, building 4_01
    # points3D_gt_4_02: town 40_4_02, building 50_6_02 (building天空区域可以再稀疏一些)
    # train_dir = "/data2/lpl/data/pure/town1/train"
    # create_point_cloud_gt_by_depth(train_dir)
    # train_dir = "/data2/lpl/data/pure/town2/train"
    # create_point_cloud_gt_by_depth(train_dir)
    # train_dir = "/data2/lpl/data/pure/building1/train"
    # create_point_cloud_gt_by_depth(train_dir, 50, 6)
    # train_dir = "/data2/lpl/data/pure/building2/train"
    # create_point_cloud_gt_by_depth(train_dir, 50, 6)
    # train_dir = "/data2/lpl/data/pure/building3/train"
    # create_point_cloud_gt_by_depth(train_dir, 50, 6)
    print(1)
