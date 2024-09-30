import os.path

import colmap_read_write_model as crwm
from plyfile import PlyData, PlyElement
import numpy as np
import open3d as o3d


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    tracks = np.vstack(vertices['track']) if 'track' in vertices else np.zeros((positions.shape[0], 1))

    return


def storePly(path, xyz, rgb, tracks):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
             ('track', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, tracks), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def binary_search(arr, target):
    left, right = 0, len(arr)  # 注意这里使用 len(arr)

    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid  # 这里不会跳过可能的目标值

    if left < len(arr) and arr[left] == target:
        return left  # 找到目标值
    return -1  # 未找到目标值


def mvsPly_to_colBin(path_):
    plydata = PlyData.read(path_)
    vertices = plydata['vertex']
    xyz_ = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    rgb_ = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # # points_ = np.hstack([positions, colors])
    # # outPath_ = path_.rsplit('.', 1)[0] + '.txt'
    # # np.savetxt(outPath_, points_, fmt="%.6f %.6f %.6f %.6f %.6f %.6f")
    # xyzRgb_ = np.loadtxt(path_)
    # xyz_ = xyzRgb_[:, :3]
    # rgb_ = xyzRgb_[:, 3:6]

    print(xyz_.shape[0])
    pointCloud_ = o3d.geometry.PointCloud()
    pointCloud_.points = o3d.utility.Vector3dVector(xyz_)
    pointCloud_.colors = o3d.utility.Vector3dVector(rgb_)

    downVoxelSizeL_, downVoxelSizeR_ = 0.01, 0.5  # 注意这里使用 len(arr)
    ptLimitMin_ = 6_000_000
    ptLimitMax_ = 7_000_000
    downVoxelSizeM_ = -1
    canFlg = True

    print("000000000000000000000000000000000000000000000000000")

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

    print("000000000000000000000000000000000000000000000000000")

    while True:
        pcdDownMax_ = pointCloud_.voxel_down_sample(voxel_size=downVoxelSizeL_)
        ptNumsMax_ = np.array(pcdDownMax_.points).shape[0]
        if ptNumsMax_ < ptLimitMin_:
            if downVoxelSizeL_ > 0.001:
                downVoxelSizeL_ /= 2.
            else:
                print('point clouds are so sparse')
                canFlg = False
                break
        else:
            break

    if not canFlg:
        raise Exception

    print("000000000000000000000000000000000000000000000000000")

    while downVoxelSizeL_ < downVoxelSizeR_:
        downVoxelSizeM_ = (downVoxelSizeL_ + downVoxelSizeR_) / 2.
        pcdDown_ = pointCloud_.voxel_down_sample(voxel_size=downVoxelSizeM_)
        ptNumsT_ = np.array(pcdDown_.points).shape[0]
        print(downVoxelSizeL_, downVoxelSizeR_, ptNumsT_)

        if ptLimitMin_ <= ptNumsT_ <= ptLimitMax_:
            break

        if ptNumsT_ < ptLimitMin_:
            downVoxelSizeR_ = downVoxelSizeM_
        else:
            downVoxelSizeL_ = downVoxelSizeM_

    print("000000000000000000000000000000000000000000000000000")

    pcdDown_ = pointCloud_.voxel_down_sample(voxel_size=downVoxelSizeM_)

    xyz_ = []
    rgb_ = []

    xyz_ = np.asarray(pcdDown_.points)
    rgb_ = np.asarray(pcdDown_.colors)
    if rgb_[0][0] < 1 or rgb_[0][1] < 1 or rgb_[0][2] < 1:
        rgb_ = (rgb_ * 255).astype(int)

    ptNums_ = xyz_.shape[0]
    print(ptNums_)
    points3D_ = {}
    for ptId_ in range(ptNums_):
        points3D_[ptId_] = crwm.Point3D(
            id=ptId_,
            xyz=xyz_[ptId_, :3],
            rgb=rgb_[ptId_, :3],
            error=np.array([0]),
            image_ids=np.array([]),
            point2D_idxs=np.array([])
        )
    outPath_ = path_.rsplit('.', 1)[0] + '.bin'
    crwm.write_points3D_binary(points3D_, outPath_)


def test_removeIds(inputPath_, removeNames_, outputPath_=None):
    crwm.remove_images_based_fileNames(inputPath_, removeNames_, outputPath_)


def images_clear_tracks(path_):
    images_ = crwm.read_images_binary(path_)
    imagesClear_ = {}
    for iK_ in images_.keys():
        image_ = images_[iK_]
        imagesClear_[iK_] = crwm.Image(
            id=image_.id,
            qvec=image_.qvec,
            tvec=image_.tvec,
            camera_id=image_.camera_id,
            name=image_.name,
            xys=np.array([]),
            point3D_ids=np.array([])
        )

    outPath_ = path_.rsplit('.', 1)[0] + '_clear.bin'
    crwm.write_images_binary(imagesClear_, outPath_)


def get_mvs_time(timeSL):
    timeL = [0, 0, 0, 0]
    for timeS in timeSL:
        timeE = timeS.split('.')
        timeET = [float(tm_) for tm_ in timeE]
        timeET = timeET[::-1]
        for i in range(len(timeET)):
            timeL[3-i] += timeET[i]

    step3_ = float(int(timeL[3]) // 1000)
    timeL[3] = int(timeL[3]) % 1000

    timeL[2] += step3_
    step2_ = float(int(timeL[2]) // 60)
    timeL[2] = int(timeL[2]) % 60

    timeL[1] += step2_
    step1_ = float(int(timeL[1]) // 60)
    timeL[1] = int(timeL[1]) % 60

    timeL[0] += step1_
    timeL = [str(tt_) for tt_ in timeL]
    print(':'.join(timeL))


if __name__ == '__main__':
    # test_removeIds('/home/hongqingde/devdata/map/paper/block2_sxfx_GM/sparse/0/images.bin', removeNames_=['100695'])
    mvsPly_to_colBin('/home/hongqingde/devdata/workspace_gitmp/input/paper/block2_well_CM/points3D_dense.ply')
    # images_clear_tracks('/home/hongqingde/devdata/workspace_gitmp/input/block2_sxfx_SBT/sparse_mvs/images.bin')
    # get_mvs_time(
    #     [
    #         '2.79',
    #         '122',
    #         '1.952',
    #         '79',
    #         '2.34.864',
    #         '3.10.514',
    #         '3.14.654',
    #         '6.49.276',
    #         '48.807',
    #         '1.48.147',
    #         '17.47.805',
    #         '9.141',
    #         '580'
    #     ]
    # )