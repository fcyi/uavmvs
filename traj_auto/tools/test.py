import copy

import numpy as np
from scipy import interpolate
import math
import cv2
import os

import matplotlib.pyplot as plt
import geo_tools as gtls
import base_tools as btls
import utils as utl


def bspline_test():
    ctr =np.array( [
                    # (3 , 1), (2.5, 4), (0, 1),
                    # (-2.5, 4),
                    # (-3, 0), (-2.5, -4), (0, -1), (2.5, -4), (3, -1)
        # (0, 0), (2, 0), (4, 0), (6, 0), (8, 0), (10, 0),
        # (10, 2),(8, 2), (6, 2), (4, 2), (2, 2), (0, 2),
        # (0, 4), (2, 4), (4, 4), (6, 4), (8, 4), (10, 4)
        (0, 0),
        (5, 5),
        (10, 10)
        #
        # (8, 0), (9, 0), (10, 0),
        # (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9),
        # (10, 10), (9, 10), (8, 10)
        # (10, 1),
        # (0.1, 2), (0, 2),
        # (0, 3),
        # (0, 4), (10, 4),
        # (9, 0), (10, 0),
        # (10, 2), (9, 2),
                    ])

    x=ctr[:,0]
    y=ctr[:,1]

    #x=np.append(x,x[0])
    #y=np.append(y,y[0])

    nums_ = 10
    tck, u = interpolate.splprep([x, y], k=2,s=0)
    u=np.linspace(0,1,num=nums_,endpoint=True)
    out = interpolate.splev(u,tck)

    plt.figure()
    plt.plot(x, y, 'ro', out[0], out[1], 'bo')
    plt.legend(['Points', '插值B样条', '真实数据'],loc='best')
    plt.axis([-5, 16, -5, 11])
    # plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
    plt.title('B样条插值')
    plt.show()


def get_enclosingCircleInfo():
    vertexs = [[12.5, -3.8], [8.88, -3.8], [10.56, -1.2]]
    vertexs_arr = np.array([[ver] for ver in vertexs], dtype=np.float32)
    (x, y), radius = cv2.minEnclosingCircle(vertexs_arr)
    print(radius, x, y)


def img_halve():
    imgPaths = [
        "/home/hongqingde/devdata/map2/hqd/nonTrain/bottle/90_SBD_40W/compare_render_pairs/100001_37.44_0.98_0.016.jpg",
        "/home/hongqingde/devdata/map2/hqd/nonTrain/bottle/90_SBD_40W/compare_render_pairs/100023_17.70_0.86_0.217.jpg",
        "/home/hongqingde/devdata/map2/hqd/nonTrain/bottle/90_SBD_40W/compare_render_pairs/100045_17.52_0.85_0.259.jpg",
        "/home/hongqingde/devdata/map2/hqd/nonTrain/bottle/90_SBD_40W/compare_render_pairs/100067_17.28_0.88_0.174.jpg",
        "/home/hongqingde/devdata/map2/hqd/nonTrain/bottle/90_SBD_160W/compare_render_pairs/100001_35.18_0.97_0.025.jpg",
        "/home/hongqingde/devdata/map2/hqd/nonTrain/bottle/90_SBD_160W/compare_render_pairs/100023_18.47_0.85_0.232.jpg",
        "/home/hongqingde/devdata/map2/hqd/nonTrain/bottle/90_SBD_160W/compare_render_pairs/100045_16.97_0.82_0.296.jpg",
        "/home/hongqingde/devdata/map2/hqd/nonTrain/bottle/90_SBD_160W/compare_render_pairs/100067_16.58_0.85_0.219.jpg"
    ]

    dstDir = "/home/hongqingde/devdata/map2/hqd/nonTrain/bottle/test"

    for imgPath in imgPaths:
        img = cv2.imread(imgPath)
        H, W = img.shape[:2]
        imgSrc = img[:H//2]
        imgDst = img[H//2:]

        preFix = imgPath.split('/')[-3]
        fileName = os.path.basename(imgPath).rsplit('.', maxsplit=1)[0]
        sufFix = os.path.basename(imgPath).rsplit('.', maxsplit=1)[1]
        dstPathSrc = os.path.join(dstDir, preFix+'_'+fileName+'_Src.'+sufFix)
        dstPathDst = os.path.join(dstDir, preFix + '_' + fileName + '_Dst.' + sufFix)
        cv2.imwrite(dstPathSrc, imgSrc)
        cv2.imwrite(dstPathDst, imgDst)


def img_halve2():
    imgPaths = [
        "/home/hongqingde/devdata/map2/hqd/nonTrain/bottle/5_SBD/compare_render_pairs/100326_37.18_0.97_0.066.jpg",
        "/home/hongqingde/devdata/map2/hqd/nonTrain/bottle/5_SBD_20W/compare_render_pairs/100326_39.23_0.97_0.051.jpg",
        "/home/hongqingde/devdata/map2/hqd/nonTrain/bottle/5_SBD_40W/compare_render_pairs/100326_40.86_0.98_0.032.jpg",
        "/home/hongqingde/devdata/map2/hqd/nonTrain/bottle/5_SBD_80W/compare_render_pairs/100326_41.88_0.98_0.022.jpg",
        "/home/hongqingde/devdata/map2/hqd/nonTrain/bottle/5_SBD_160W/compare_render_pairs/100326_42.39_0.99_0.012.jpg"
    ]

    dstDir = "/home/hongqingde/devdata/map2/hqd/nonTrain/bottle/test_part"

    rangeSrc_ = [760, 1134, 450, 680]  # x, y
    for imgPath in imgPaths:
        img = cv2.imread(imgPath)
        H, W = img.shape[:2]

        # imgSrc = img[:H//2, :]
        # plt.figure()
        # plt.imshow(imgSrc)
        # plt.show()

        rangeDst_ = copy.deepcopy(rangeSrc_)
        rangeDst_[2] += H//2
        rangeDst_[3] += H//2
        imgSrc = img[rangeSrc_[2]:rangeSrc_[3], rangeSrc_[0]:rangeSrc_[1]]
        imgDst = img[rangeDst_[2]:rangeDst_[3], rangeDst_[0]:rangeDst_[1]]

        print(imgSrc.shape, imgDst.shape)

        preFix = imgPath.split('/')[-3]
        fileName = os.path.basename(imgPath).rsplit('.', maxsplit=1)[0]
        sufFix = os.path.basename(imgPath).rsplit('.', maxsplit=1)[1]
        dstPathSrc = os.path.join(dstDir, preFix+'_'+fileName+'_Src.'+sufFix)
        dstPathDst = os.path.join(dstDir, preFix + '_' + fileName + '_Dst.' + sufFix)
        cv2.imwrite(dstPathSrc, imgSrc)
        cv2.imwrite(dstPathDst, imgDst)


def pathSmooth():
    data = gtls.tu_polygon_gen(5, 30)
    flyHeight = 20
    sideOverlap = 0.7
    headOverlap = 0.8
    rratio = 0.5
    isCoord = False
    frameW, frameH, focal = 35, 24, 26

    vertexsNorm, verGrav = gtls.polygon_norm(data)

    fovD = 90
    pitchD = -45

    if isCoord:
        sideStep = btls.get_step_base_rep_ratio(flyHeight, frameW, fovD, pitchD, focal, sideOverlap)
        headStep = btls.get_step_base_rep_ratio(flyHeight, frameH, fovD, pitchD, focal, headOverlap)
    else:
        sideStep = btls.get_step_base_rep_ratio(flyHeight, frameH, fovD, pitchD, focal, sideOverlap)
        headStep = btls.get_step_base_rep_ratio(flyHeight, frameW, fovD, pitchD, focal, headOverlap)

    print(sideStep, headStep)
    print(data)

    # extendD = min(-headStep, -sideStep)
    extendD = -5

    trajNodes, vertexsE = gtls.dji_poly_traj_v1(vertexsNorm, sideStep, 0, extendD)
    print(trajNodes)

    # btls.polygon_draw([vertexsNorm, vertexsE], trajNodes)
    # yawTypeTmp_ = 1
    # trajLists = utl.get_traj_by_node_sim(trajNodes, headStep, rratio, yawTypeTmp_)

    trajNodesA = np.array(trajNodes)
    trajNodesX = trajNodesA[:, 0]
    trajNodesY = trajNodesA[:, 1]

    trajNodesN, disSeg = utl.line_traj_node_add_2D(trajNodes)
    trajNodesNA = np.array(trajNodesN)
    trajNodesNAX = trajNodesNA[:, 0]
    trajNodesNAY = trajNodesNA[:, 1]

    # 拐角标识
    wpIndexs_ = utl.wrapPoint_index(trajNodes, trajNodesN)

    # 节点平滑，拐角更新
    trajNodesS, wpIndexSs = utl.process_points(trajNodesN, disSeg, wpIndexs_)
    # 节点修缮
    trajNodesS, wpIndexSs = utl.fix_wrapNodes(trajNodesS, wpIndexSs)

    trajNodesSA = np.array(trajNodesS)
    trajNodesSAX = trajNodesSA[:, 0]
    trajNodesSAY = trajNodesSA[:, 1]

    # 构建拐弯区间
    wrapRanges_ = utl.get_wrapRange(wpIndexSs, len(trajNodesS), True)

    # b样条曲线平滑（整体平滑）
    trajLen_ = gtls.get_traj_len(trajNodesS)
    posNum_ = int(10*trajLen_ / headStep + 0.5)

    tck_, u_ = interpolate.splprep([trajNodesSAX, trajNodesSAY], k=3, s=0)
    u_ = np.linspace(0, 1, num=posNum_, endpoint=True)
    posSm_ = interpolate.splev(u_, tck_)
    #
    # plt.figure()
    # plt.plot(trajNodesX, trajNodesY, 'r-', posSm_[0], posSm_[1], 'b-.')
    # plt.legend(['trajSrc', 'trajBSpline'], loc='best')
    # # plt.axis([-5, 16, -5, 11])
    # # plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
    # plt.title('smooth traj')
    # plt.show()


    # b样条曲线平滑（局部平滑）
    localSmoothNodes_ = []
    ptIdxS_ = 0
    wrIdxS_ = 0
    wrLen_ = len(wrapRanges_)
    trajNodesNum_ = len(trajNodesS)
    while ptIdxS_ < trajNodesNum_:
        if (wrIdxS_ < wrLen_) and (wrapRanges_[wrIdxS_][0] <= ptIdxS_) and (wrapRanges_[wrIdxS_][1] >= ptIdxS_):
            wpNodes_ = []
            wL_, wR_ = wrapRanges_[wrIdxS_][:]
            for wIdx_ in range(wL_, wR_+1):
                wpNodes_.append(trajNodesS[wIdx_])
            print(wL_, wR_)
            wpNodesA_ = np.array(wpNodes_)
            wpNodesAX_ = wpNodesA_[:, 0]
            wpNodesAY_ = wpNodesA_[:, 1]
            localLen_ = gtls.get_traj_len(wpNodes_)
            lposNum_ = int(10 * localLen_ / disSeg + 0.5)
            print('======', lposNum_)

            ltck_, lu_ = interpolate.splprep([wpNodesAX_, wpNodesAY_], k=2, s=0)
            lu_ = np.linspace(0, 1, num=lposNum_, endpoint=True)
            lposSm_ = interpolate.splev(lu_, ltck_)
            lposLen_ = lposSm_[0].shape[0]

            for lposIdx_ in range(lposLen_):
                localSmoothNodes_.append([lposSm_[0][lposIdx_], lposSm_[1][lposIdx_]])

            ptIdxS_ = wrapRanges_[wrIdxS_][1]+1
            wrIdxS_ += 1
        else:
            if (wrIdxS_ < wrLen_) and wrapRanges_[wrIdxS_][1] < ptIdxS_:
                wrIdxS_ += 1
            localSmoothNodes_.append(trajNodesS[ptIdxS_])
            ptIdxS_ += 1
    localSmoothNodesA_ = np.array(localSmoothNodes_)
    localSmoothNodesAX_ = localSmoothNodesA_[:, 0]
    localSmoothNodesAY_ = localSmoothNodesA_[:, 1]

    plt.figure()
    plt.plot(trajNodesX, trajNodesY, 'r-', posSm_[0], posSm_[1], 'b-.', localSmoothNodesAX_, localSmoothNodesAY_, 'g-')
    plt.legend(['trajSrc', 'trajBSpline, k=3', 'localBSpline, k=2'], loc='best')
    # plt.axis([-5, 16, -5, 11])
    # plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
    plt.title('smooth traj')
    plt.show()


    # wPts_ = [trajNodesS[wIdx_] for wIdx_ in wpIndexSs]
    wPts_ = []
    print(len(wrapRanges_))
    for wrapRange_ in wrapRanges_:
        wL_, wR_ = wrapRange_[:]
        for wIdx_ in range(wL_, wR_+1):
            wPts_.append(trajNodesS[wIdx_])

    wPtsA_ = np.array(wPts_)
    wPtsAX_ = wPtsA_[:, 0]
    wPtsAY_ = wPtsA_[:, 1]

    #
    plt.figure()
    # plt.plot(trajNodesX, trajNodesY, 'r-.',trajNodesNAX, trajNodesNAY, 'bo', trajNodesSAX, trajNodesSAY, 'go')
    # plt.legend(['traj', 'same seg', 'smooth nodes'], loc='best')
    plt.plot(trajNodesX, trajNodesY, 'r-.',trajNodesSAX, trajNodesSAY, 'bo', wPtsAX_, wPtsAY_, 'go')
    plt.legend(['traj', 'smooth nodes', 'wrap nodes'], loc='best')
    # plt.axis([-5, 16, -5, 11])
    # plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
    plt.title('same seg split')
    plt.show()


def angle_bisector_test(pts_, inPt_):
    degree_ = gtls.calc_cross_degree_based_point(pts_[0], pts_[1], pts_[2])
    degree0_ = gtls.calc_cross_degree_based_point(pts_[0], pts_[1], inPt_)
    degree1_ = gtls.calc_cross_degree_based_point(inPt_, pts_[1], pts_[2])
    print(degree_, degree_/2., degree0_, degree1_)


def get_angle_bisector_point(pts_, angleIndex_):
    """
    获取三角形内指定角的角平分线的交在angleIndex_对应顶点对应的边上的交点。
    pts_: 包含三个点的元组(x1, y1, x2, y2, x3, y3)
    angleIndex_: 角的索引，1 或 2 或 3
    """
    # 提取三角形的三个顶点
    A_, B_, C_ = pts_[0], pts_[1], pts_[2]
    # x1, y1, x2, y2, x3, y3 = triangle

    # 计算三角形的两边长
    a_ = gtls.distance_p1p2(B_, C_)
    b_ = gtls.distance_p1p2(C_, A_)
    c_ = gtls.distance_p1p2(A_, B_)

    # 计算边长
    # 计算内心坐标
    inPX_ = (a_ * A_[0] + b_ * B_[0] + c_ * C_[0]) / (a_ + b_ + c_)
    inPY_ = (a_ * A_[1] + b_ * B_[1] + c_ * C_[1]) / (a_ + b_ + c_)

    inPt_ = [inPX_, inPY_]  # 内点、内接圆圆心、角平分线交点

    angle_bisector_test([C_, A_, B_], inPt_)
    angle_bisector_test([A_, B_, C_], inPt_)
    angle_bisector_test([B_, C_, A_], inPt_)

    line0_ = gtls.line_from_points(pts_[(angleIndex_+1) % 3], pts_[(angleIndex_+2) % 3])
    line1_ = gtls.line_from_points(pts_[angleIndex_ % 3], inPt_)
    biscPt_ = gtls.intersection_of_lines(line0_, line1_)
    assert biscPt_ is not None, "算法逻辑出现问题，三角形的角平分线与其三条边必定存在交点"

    return biscPt_


def geo_test():
    pts_ = [[0, 0], [1, 0], [0.5, 2]]
    angleIdx = 0
    bp_ = get_angle_bisector_point(pts_, angleIdx)

    pts_.append(pts_[0])
    # pts.append(bp_)
    pts_ = np.array(pts_)

    plt.figure()
    plt.plot(pts_[:, 0], pts_[:, 1], 'b-', [pts_[angleIdx][0], bp_[0]], [pts_[angleIdx][1], bp_[1]], 'g-.')
    # plt.legend(['traj', 'smooth nodes', 'wrap nodes'], loc='best')
    # plt.axis([-5, 16, -5, 11])
    # plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
    plt.title('same seg split')
    plt.show()


if __name__ == "__main__":
    # bspline_test()
    pathSmooth()
    # img_halve()
    # geo_test()
    pass

