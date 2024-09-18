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
from process_points3D import load_tum_file, depth2xyz, pc_cam_to_pc_world, read_array, resize_torch, voxel_random_filter, voxel_texture_filter, read_colmap_camera
from read_write_binary import read_cameras_binary, read_images_binary
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_image(id, c2w, depth_dir, images_dir, K, num1=40, num2=4):
    depth_path = os.path.join(depth_dir, id + ".npy")
    depth = np.load(depth_path)
    depth = np.clip(depth, 0, 1000)
    # col_depthf = os.path.join(depth_dir, id + ".png.geometric.bin")
    # if not os.path.exists(col_depthf):
    #     print("skip visualize for the file does not exists: {}".format(col_depthf))
    #     pass
    # pre_depth_np = read_array(col_depthf)
    # target_height, target_width = 1080, 1920
    # depth_height, depth_width = pre_depth_np.shape[:2]
    # pad_height = target_height - depth_height
    # pad_width = target_width - depth_width
    # depth = np.pad(pre_depth_np, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    img_path = os.path.join(images_dir, id + ".png")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    h, w = depth.shape

    w2c = np.linalg.inv(c2w)

    # 当前像素点对应的3D点，由其深度 转换为 相机坐标系下的3D点（depth_scale 根据深度图和位姿的尺度进行调整）
    camera_xyz_map = depth2xyz(depth, K, depth_scale=1.0)  # H W 3，xyz

    Z = 10
    colors = []
    point3ds = []
    normals = []
    sobel_scores = []
    # 遍历每个像素点，每100个选1个
    for i in range(0, h, num1 if Z >= 1000 else num2):
        for j in range(0, w, num1 if Z >= 1000 else num2):
            y = i
            x = j
            Z = depth[i, j]  # 对应的绝对深度值

            color = img[y, x]
            colors.append(color)

            xyz = camera_xyz_map[y, x]  # 取出相机坐标系下 对应像素的 点云 Pc
            point3d_gt = pc_cam_to_pc_world(xyz, w2c)  # 相机下的点云转换为世界坐标系下的点云 Pw
            point3ds.append(point3d_gt)

            # 使用法线表征纹理的丰富度
            neighbors = camera_xyz_map[max(y - 1, 0):min(y + 2, h), max(x - 1, 0):min(x + 2, w)]  # 取出以当前3D点为中心的3x3邻居点
            mean, eigenvector, _ = cv2.PCACompute2(neighbors.reshape(-1, 3), mean=None)
            normal = mean  # 1 3
            normals.append(normal)

            # 当前像素的Sobel梯度作为纹理的度量
            sobel_x_i = sobel_x[y, x, :]
            sobel_y_i = sobel_y[y, x, :]
            sobel_score = np.linalg.norm(sobel_x_i) + np.linalg.norm(sobel_y_i)
            sobel_scores.append(sobel_score)
    return np.array(colors), np.array(point3ds), np.array(normals).squeeze(axis=1), np.array(sobel_scores)


def create_gt_points3D_by_depth(train_dir, num1=40, num2=4):
    images_dir = f"{train_dir}/images"
    depth_dir = f"{train_dir}/depth"
    pose_path = os.path.join(train_dir, "gt_as_col_pose.txt")
    # calar
    K = np.array([[554.256, 0, 960],
                  [0, 554.256, 540],
                  [0, 0, 1]])
    # K = np.array([[960, 0, 960],
    #               [0, 960, 540],
    #               [0, 0, 1]])
    dc2w = load_tum_file(pose_path)  # gt pose (Twc)
    colors = []
    point3ds = []
    normals = []
    sobel_scores = []

    pbar = tqdm(total=len(dc2w))
    num_1 = 0
    num_2 = 0
    # 遍历每张图片
    co = 0
    for id, c2w in dc2w.items():
        pbar.update(1)

        depth_path = os.path.join(depth_dir, id + ".npy")
        depth = np.load(depth_path)
        depth = np.clip(depth, 0, 1000)
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
        for i in range(0, h, num1 if Z >= 900 else num2):
            for j in range(0, w, num1 if Z >= 900 else num2):
                x = num1 if Z >= 900 else num2
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
        co += 1
        if co >= 6:
            break
    # 保存点云
    print("\n{}, {}".format(num_1, num_2))
    point3ds = np.array(point3ds)   # N 3
    colors = np.array(colors)       # N 3
    normals = np.array(normals).squeeze(axis=1)     # N 3
    sobel_scores = np.array(sobel_scores)

    points = np.hstack([np.asarray(point3ds), np.asarray(colors)])
    output_path = os.path.join(train_dir, "sparse_gt_1/0/points3D_gt.txt")
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


def create_gt_points3D_by_depth_multi(train_dir, num1=40, num2=4):
    images_dir = f"{train_dir}/images"
    depth_dir = f"{train_dir}/depth"

    pose_path = os.path.join(train_dir, "gt_as_col_pose.txt")
    # Fov=120: 554.2562584220407
    # colmap: 555.610765
    K = np.array([[554.2562584220407, 0, 960],
                  [0, 554.2562584220407, 540],
                  [0, 0, 1]])
    dc2w = load_tum_file(pose_path)  # gt pose (Twc)

    colors = []
    point3ds = []
    normals = []
    sobel_scores = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, id, c2w, depth_dir, images_dir, K, num1, num2) for id, c2w in dc2w.items()]
        for future in as_completed(futures):
            result_colors, result_point3ds, result_normals, result_sobel_scores = future.result()
            colors.append(result_colors)
            point3ds.append(result_point3ds)
            normals.append(result_normals)
            sobel_scores.append(result_sobel_scores)

    # 保存点云
    print(len(point3ds))
    point3ds = np.concatenate(point3ds, axis=0)  # N 3
    colors = np.concatenate(colors, axis=0)  # N 3
    normals = np.concatenate(normals, axis=0)  # N 3
    sobel_scores = np.concatenate(sobel_scores, axis=0)

    points = np.hstack([np.asarray(point3ds), np.asarray(colors)])
    output_path = os.path.join(train_dir, "sparse_gt_1/0/points3D_gt_4.txt")
    np.savetxt(output_path, points, fmt="%.6f %.6f %.6f %.6f %.6f %.6f")

    print(len(point3ds))

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point3ds)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)
    print(f"src num {len(point_cloud.points)}")

    # pcd_down = voxel_random_filter(point_cloud, 0.1)
    # print(f"dst num {len(pcd_down.points)}")

    random_idx, random_idx_texture = voxel_texture_filter(point_cloud, sobel_scores, 0.2)
    pcd_down = point_cloud.select_by_index(random_idx)
    pcd_down_texture = point_cloud.select_by_index(random_idx_texture)
    print(f"dst num1 {len(pcd_down.points)}, dst num2 {len(pcd_down_texture.points)}")

    # points = np.hstack([np.asarray(pcd_down.points), np.asarray(pcd_down.colors)])
    # output_path = os.path.join(train_dir, "gt_sparse_1/0/points3D_gt_3_03_clip.txt")
    # np.savetxt(output_path, points, fmt="%.6f %.6f %.6f %.6f %.6f %.6f")

    points = np.hstack([np.asarray(pcd_down_texture.points), np.asarray(pcd_down_texture.colors)])
    output_path = os.path.join(train_dir, "sparse_gt_1/0/points3D_gt_4_02_texture.txt")
    np.savetxt(output_path, points, fmt="%.6f %.6f %.6f %.6f %.6f %.6f")


if __name__ == '__main__':
    # train_dir = "/media/liuzhi/b4608ade-d2e0-430d-a40b-f29a8b22cb8c/Dataset/3DGS_Dataset/town-train"
    # create_points3D_by_depth(train_dir)
    train_dir = "/home/liuzhi/Disk_data/dataset_simulator/urban/block2_sim/train"
    # create_gt_points3D_by_depth(train_dir)
    create_gt_points3D_by_depth_multi(train_dir)


    # 远程数据集下：
    # points3D_gt: town 4_01, building 20_6_01
    # points3D_gt_: town 2_01, building 20_4_01
    # points3D_gt_4_02: town 40_4_02, building 50_6_02 (building天空区域可以再稀疏一些)
    # points3D_gt_3_03: town 40_3_03, building 60_5_03 (building天空区域可以再稀疏一些)
    print(1)