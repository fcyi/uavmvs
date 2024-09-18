import os
import os.path
import imageio
import numpy as np
import read_write_binary as im
from scipy.spatial.transform import Rotation as R

def create_empty_point3Dbin(point3D_bin_file):
    point3D = {}
    im.write_points3D_binary(point3D, point3D_bin_file)

def generate_gtimages_txt2bin(source_path, gt_Tcw_pose_path):
    data = im.read_images_binary(os.path.join(source_path, "sparse", "images.bin"))
    image = data[1]

    new_data = {}
    # 打开Tcw格式的gt位姿文件
    with open(gt_Tcw_pose_path, "r") as f:
        lines = f.readlines()
        i = 1
        for line in lines:
            line = line.strip()
            n, tx, ty, tz, qx, qy, qz, qw = line.split(" ")

            image_name = "{}.png".format(n)
            qvec = [float(i) for i in [qw, qx, qy, qz]]
            tvec = [float(i) for i in [tx, ty, tz]]
            image = image._replace(id=i, qvec=np.array(qvec), tvec=np.array(tvec), name=image_name, point3D_ids=[], xys=[])
            new_data[i] = image
            i += 1
    print(len(new_data))

    if not os.path.exists(os.path.join(source_path, "sparse_gt/created")):
        os.mkdir(os.path.join(source_path, "sparse_gt/created"))
    im.write_images_binary(new_data, os.path.join(source_path, "sparse_gt/created", "images.bin"))

def gt_pose_Twc_2_Tcw(gt_as_col_pose_path, inv_gt_pose_path):
    assert os.path.exists(gt_as_col_pose_path), 'Lack: gt_as_col_pose.txt'

    lines = open(gt_as_col_pose_path).readlines()
    with open(inv_gt_pose_path, "w") as f:
        for line in lines:
            ws = line.strip().split(' ')
            name = ws[0]
            # Twc: tx ty tz qx qy qz qw
            pose_wc = np.array([float(ws[1]), float(ws[2]), float(ws[3]), float(ws[4]), float(ws[5]), float(ws[6]), float(ws[7])])
            twc = pose_wc[:3]  # tx ty tz
            qwc = pose_wc[3:]  # qx qy qz qw
            qwc_norm = np.linalg.norm(qwc)
            if abs(qwc_norm - 1.0) > 1e-6:
                print(f"Warning: Quaternion {qwc} is not normalized.")
                qwc = qwc / qwc_norm
            Rwc = R.from_quat([qwc[0], qwc[1], qwc[2], qwc[3]]).as_matrix()

            Twc = np.eye(4)
            Twc[:3, :3] = Rwc
            Twc[:3, 3] = twc

            Tcw = np.linalg.inv(Twc)
            Rcw = R.from_matrix(Tcw[:3, :3])
            qcw = Rcw.as_quat() # qx qy qz qw
            qcw_norm = np.linalg.norm(qcw)
            if abs(qcw_norm - 1.0) > 1e-6:
                print(f"Warning: Quaternion {qcw} is not normalized.")
                qcw = qcw / qcw_norm
            tcw = Tcw[:3, 3]

            # tcw_ = -np.dot(Tcw[:3, :3], twc) # 验证 tcw = - Rcw * twc
            # Tcw: tx ty tz qw qx qy qz
            new_line = "{} {} {} {} {} {} {} {}\n".format(name, tcw[0], tcw[1], tcw[2], qcw[0], qcw[1], qcw[2], qcw[3])
            f.write(new_line)
    f.close()

def convert_cameras_bin2txt(camera_bin_file):
    cameras = im.read_cameras_binary(camera_bin_file)

    filename = camera_bin_file.split('/')[-1].split('.')[0]
    output_path = os.path.join(os.path.dirname(camera_bin_file), filename+'.txt')
    im.write_cameras_text(cameras, output_path)

def convert_cameras_txt2bin(camera_txt_file):
    cameras = im.read_cameras_text(camera_txt_file)

    filename = camera_txt_file.split('/')[-1].split('.')[0]
    output_path = os.path.join(os.path.dirname(camera_txt_file), filename+'.bin')
    im.write_cameras_binary(cameras, output_path)

def convert_points3D_bin2txt(points3D_bin_file):
    points3D = im.read_points3D_binary(points3D_bin_file)
    point3ds = []
    colors = []
    for idx, point3D in points3D.items():
        point3ds.append(point3D.xyz)
        colors.append(point3D.rgb)
    point3ds = np.array(point3ds)  # N 3
    colors = np.array(colors)  # N 3
    points = np.hstack([np.asarray(point3ds), np.asarray(colors)])

    filename = points3D_bin_file.split('/')[-1].split('.')[0]
    output_path = os.path.join(os.path.dirname(points3D_bin_file), filename+".txt")
    np.savetxt(output_path, points, fmt="%.6f %.6f %.6f %.6f %.6f %.6f")

if __name__ == "__main__":
    path = "/home/liuzhi/Disk_data/dataset_simulator/urban/block2_sim/train"

    # 使用gt位姿进行三角化，生成稠密点云
    # 1. 使用 colmap 进行 feature extract 和 feature match，生成 database文件；并进行稀疏重建

    # 2. 生成内参文件cameras.bin（如果没gt内参，则跑稀疏重建后使用sparse中的；如果有gt内参，则读取sparse的cameras.bin，替换fx）
    # cameras = im.read_cameras_binary(os.path.join(path, "sparse/cameras.bin"))
    # camera_param = cameras[1].params
    # camera_param[0] = 554.2562584220407339
    # im.write_cameras_binary(cameras, os.path.join(path, "gt_sparse_1/created/cameras.bin"))
    # debug中验证是否更改正确
    # cameras_ = im.read_cameras_binary(os.path.join(path, "gt_sparse_1/created/cameras.bin"))

    # 或将bin文件转化为txt文件，更改fx后，再转为bin文件
    # convert_cameras_bin2txt(os.path.join(path, "sparse/cameras.bin")) # 转换为txt文件
    # convert_cameras_txt2bin(os.path.join(path, "sparse/cameras_gt.txt"))  # 更改fx值（ar仿真模拟器的fx = 554.2562584220407339）后，重命名为_gt.txt，再转为bin文件
    # # 验证
    # cameras_ = im.read_cameras_binary(os.path.join(path, "sparse/cameras_gt.bin"))

    # 3. 将相机位姿由Twc转换成Tcw，即从gt_as_col_pose.txt生成train_gt_pose.txt.
    # test_path = os.path.join(path, "../test")
    # gt_pose_Twc_2_Tcw(os.path.join(path, "gt_as_col_pose.txt"), os.path.join(path, "train_gt_pose.txt"))
    # gt_pose_Twc_2_Tcw(os.path.join(test_path, "gt_as_col_pose.txt"), os.path.join(test_path, "test_gt_pose.txt"))

    # 4. 从Tcw的gt位姿生成的images.bin
    # generate_gtimages_txt2bin(path, os.path.join(path, "train_gt_pose.txt"))
    #
    # # 5. 创建空的 points3D.bin
    # create_empty_point3Dbin(os.path.join(path, "sparse_gt/created/points3D.bin"))

    # 6. 终端使用下面的命令通过 三角化生成稀疏点云（先创建文件夹sparse_gt/0）
    # colmap point_triangulator --database_path database.db --image_path images --input_path sparse_gt/created --output_path sparse_gt/0

    # for idx, scene in enumerate(['building1-train', 'building2-train', 'building3-train', 'town-train', 'town2-train']):
    #     print('---------------------------------------------------------------------------------')
    #     one_cmd = f"cd /media/liuzhi/b4608ade-d2e0-430d-a40b-f29a8b22cb8c/Dataset/3DGS_Dataset/gt/{scene} && colmap point_triangulator --database_path database.db --image_path images --input_path created/sparse --output_path triangulated/sparse"
    #     print(one_cmd)
    #     os.system(one_cmd)

    # 测试：检查文件内容
    # cameras = im.read_cameras_binary(os.path.join(path, "gt_sparse_1/created/cameras.bin"))

    # images = im.read_images_binary(os.path.join(path, "sparse/images.bin"))
    #
    # points3D1 = im.read_points3D_binary(os.path.join(path, "sparse/points3D.bin"))
    # convert_points3D_bin2txt(os.path.join(path, "sparse/points3D.bin"))
    # points3D2 = im.read_points3D_binary(os.path.join(path, "gt_sparse_1/0/points3D.bin"))
    x = 0