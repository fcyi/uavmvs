import cv2
import os
import utils
import numpy as np
import random
import math
import glob

from random import sample
import itertools


if __name__ == '__main__':
    specRatio = 2  # 期望Marker在窗口中的占比为50%，即窗口面积为Marker面积的specRatio倍，默认背景的面积为窗口面积的specRatio倍
    winSpecRatio = math.sqrt(specRatio)

    mkPath = "/home/hongqingde/Downloads/AugumentImageMarker/AIMarker_rename/Pic_0082_100.jpg"
    bgPath = "/home/hongqingde/workspace/datasets/AIDataset/bgs_/bg_11.jpg"
    vCFacFilePath = "/home/hongqingde/workspace/datasets/AIDataset/test/v-Pic_0082_100-bg_11/svp_4.txt"

    mkImg = cv2.imread(mkPath)
    bgImg = cv2.imread(bgPath)

    filename = mkPath.split('/')[-1]
    name0 = filename.split('.')[0]
    filename = bgPath.split('/')[-1]
    name1 = filename.split('.')[0]

    # 获取图像的尺寸
    bgRow, bgCol = bgImg.shape[:2]
    mkRow, mkCol = mkImg.shape[:2]
    winRow, winCol = int(winSpecRatio * mkRow), int(winSpecRatio * mkCol)  # 窗口大小
    nbgRow, nbgCol = specRatio*mkRow, specRatio*mkCol

    if(bgRow < nbgRow) or (bgCol < nbgCol):
        scaleRatio = max(nbgRow/bgRow, nbgCol/bgCol)
        bgImg = utils.simPercent_resize(bgImg, scaleRatio)
        bgRow, bgCol = bgImg.shape[:2]
        nbgRow, nbgCol = min(nbgRow, bgRow), min(nbgCol, bgCol)

    # 选取的背景位置偏移
    rTransBias = random.randint(0, bgRow-nbgRow) if (bgRow != nbgRow) else 0
    cTransBias = random.randint(0, bgCol-nbgCol) if (bgCol != nbgCol) else 0
    bgBou = [cTransBias, cTransBias+nbgCol-1, rTransBias, rTransBias+nbgRow-1]  # 次背景位置
    mkBou = utils.get_bou_in_senceCenter([bgBou[0], bgBou[2], nbgCol, nbgRow], [mkRow, mkCol])  # marker位置（marker放在次背景中心）
    winBou = utils.get_bou_in_senceCenter([bgBou[0], bgBou[2], nbgCol, nbgRow], [winRow, winCol])  # 窗口位置（窗口位于次背景中心）
    mkCenter = [(mkBou[0] + mkBou[1]) // 2, (mkBou[2] + mkBou[3]) // 2]  # marker在次背景中的放置位置中心

    nWinBou = [0, winCol - 1, 0, winRow - 1]
    nWinVer = utils.bou_to_ver(nWinBou)

    rawSence = bgImg.copy()
    rawSence[mkBou[2]:mkBou[3]+1, mkBou[0]:mkBou[1]+1] = mkImg

    senceLimitVer = utils.bou_to_ver([0, bgCol, 0, bgRow])
    winVer = utils.bou_to_ver(winBou)

    nMkBou = [mkBou[0]-winBou[0], mkBou[0]-winBou[0]+mkCol-1, mkBou[2]-winBou[2], mkBou[2]-winBou[2]+mkRow-1]  # 窗口中marker的边界位置
    nMkVer = utils.bou_to_ver(nMkBou)
    nMkCenter = [mkCenter[0] - winBou[0], mkCenter[1] - winBou[2]]
    nMkCenter_ = [(nWinBou[0] + nWinBou[1]) // 2, (nWinBou[2] + nWinBou[3]) // 2]

    showImg = rawSence[winBou[2]:winBou[3]+1, winBou[0]:winBou[1]+1]

    if (nMkCenter[0] != nMkCenter_[0]) or (nMkCenter[1] != nMkCenter_[1]):
        cv2.circle(showImg, tuple(nMkCenter), 4, (255, 0, 0), -1)
        cv2.circle(showImg, tuple(nMkCenter_), 4, (0, 0, 255), -1)
        print("marker中心有所矛盾")
        print(nMkCenter)
        print(nMkCenter_)

    bias_row = np.array([0, 0, 1])
    vCFac_ = utils.load_matrix_from_txt(vCFacFilePath, size=(1, 6)).tolist()
    vCFac = vCFac_[0]
    print(vCFac)

    senceCurVer = utils.bou_to_ver([0, bgCol, 0, bgRow])
    perspecMatrix = np.zeros((3, 3), dtype=np.float32)
    perspecMatrix[[0, 1, 2], [0, 1, 2]] = 1

    # # 随机旋转
    # viewChangeSence = rawSence
    rotation_matrix = cv2.getRotationMatrix2D((mkCenter[0], mkCenter[1]), vCFac[0], 1)  # 获取旋转矩阵
    viewChangeSence = cv2.warpAffine(rawSence, rotation_matrix, (bgCol, bgRow))  # 进行旋转变换
    rotation_matrix_ = np.vstack((rotation_matrix, bias_row))
    perspecMatrix = np.matmul(rotation_matrix_, perspecMatrix)
    Flag, senceCurVer = utils.viewChange_valid(senceCurVer, perspecMatrix, winVer, senceLimitVer)

    # # 随机水平斜切
    # viewChangeSence = rawSence
    shear_matrix_x = np.array([[1, vCFac[1], -vCFac[1] * mkCenter[0]], [0, 1, 0]], dtype=np.float32)  # 获取水平斜切矩阵
    viewChangeSence = cv2.warpAffine(viewChangeSence, shear_matrix_x, (bgCol, bgRow))  # 进行水平斜切变换
    shear_matrix_x_ = np.vstack((shear_matrix_x, bias_row))
    perspecMatrix = np.matmul(shear_matrix_x_, perspecMatrix)
    Flag, senceCurVer = utils.viewChange_valid(senceCurVer, perspecMatrix, winVer, senceLimitVer)

    # 随机垂直斜切
    # viewChangeSence = rawSence
    shear_matrix_y = np.array([[1, 0, 0], [vCFac[2], 1, -vCFac[2] * mkCenter[1]]], dtype=np.float32)  # 获取垂直斜切矩阵
    viewChangeSence = cv2.warpAffine(viewChangeSence, shear_matrix_y, (bgCol, bgRow))  # 进行垂直斜切变换
    shear_matrix_y_ = np.vstack((shear_matrix_y, bias_row))
    perspecMatrix = np.matmul(shear_matrix_y_, perspecMatrix)
    Flag, senceCurVer = utils.viewChange_valid(senceCurVer, perspecMatrix, winVer, senceLimitVer)

    # 随机平移
    # viewChangeSence = rawSence
    translation_matrix = np.array([[1, 0, vCFac[3]], [0, 1, vCFac[4]]], dtype=np.float32)  # 获取平移矩阵
    viewChangeSence = cv2.warpAffine(viewChangeSence, translation_matrix, (bgCol, bgRow))  # 进行平移变换
    translation_matrix_ = np.vstack((translation_matrix, bias_row))
    perspecMatrix = np.matmul(translation_matrix_, perspecMatrix)
    Flag, senceCurVer = utils.viewChange_valid(senceCurVer, perspecMatrix, winVer, senceLimitVer)

    # 随机尺度缩放
    # viewChangeSence = rawSence
    scale_matrix = np.array([[vCFac[5], 0, 0], [0, vCFac[5], 0]], dtype=np.float32)  # 获取垂直斜切矩阵
    viewChangeSence = cv2.warpAffine(viewChangeSence, scale_matrix, (bgCol, bgRow))  # 进行垂直斜切变换
    scale_matrix_ = np.vstack((scale_matrix, bias_row))
    perspecMatrix = np.matmul(scale_matrix_, perspecMatrix)
    Flag, senceCurVer = utils.viewChange_valid(senceCurVer, perspecMatrix, winVer, senceLimitVer)

    # # for i, vcs in enumerate(vSL):
    # #     # res_ = viewChangeSence[winBou[2]:winBou[3], winBou[0]:winBou[1]]
    # #     res = utils.draw_box(vcs, bgBou, (0, 0, 255))
    # #     res = utils.draw_box(res, winBou, (255, 0, 255))
    # #
    # #     cv2.imshow("test {}".format(i), res)

    showImg1 = viewChangeSence[winBou[2]:winBou[3]+1, winBou[0]:winBou[1]+1]

    # 由于后续使用的都是观测窗口中的数据，因此透射变换矩阵、marker的顶点都必须在此基础之上考虑
    nPerspecMatrix = np.zeros((3, 3), dtype=np.float32)
    nPerspecMatrix[[0, 1, 2], [0, 1, 2]] = 1
    nrotation_matrix = cv2.getRotationMatrix2D((nMkCenter[0], nMkCenter[1]), vCFac[0], 1)  # 获取旋转矩阵
    nrotation_matrix_ = np.vstack((nrotation_matrix, bias_row))
    nPerspecMatrix = np.matmul(nrotation_matrix_, nPerspecMatrix)
    nshear_matrix_x = np.array([[1, vCFac[1], vCFac[1] * (-mkCenter[0]+winBou[2])], [0, 1, 0]], dtype=np.float32)  # 获取水平斜切矩阵
    nshear_matrix_x_ = np.vstack((nshear_matrix_x, bias_row))
    nPerspecMatrix = np.matmul(nshear_matrix_x_, nPerspecMatrix)
    nshear_matrix_y = np.array([[1, 0, 0], [vCFac[2], 1, vCFac[2] * (-mkCenter[1]+winBou[0])]], dtype=np.float32)  # 获取垂直斜切矩阵
    nshear_matrix_y_ = np.vstack((nshear_matrix_y, bias_row))
    nPerspecMatrix = np.matmul(nshear_matrix_y_, nPerspecMatrix)
    ntranslation_matrix = np.array([[1, 0, vCFac[3]], [0, 1, vCFac[4]]], dtype=np.float32)  # 获取平移矩阵
    ntranslation_matrix_ = np.vstack((ntranslation_matrix, bias_row))
    nPerspecMatrix = np.matmul(ntranslation_matrix_, nPerspecMatrix)
    nscale_matrix = np.array([[vCFac[5], 0, winBou[0]*(vCFac[5]-1)], [0, vCFac[5], winBou[2]*(vCFac[5]-1)]], dtype=np.float32)  # 获取垂直斜切矩阵
    nscale_matrix_ = np.vstack((nscale_matrix, bias_row))
    nPerspecMatrix = np.matmul(nscale_matrix_, nPerspecMatrix)

    nMkVerP_ = utils.points_trans(np.array(nMkVer), nPerspecMatrix)
    nMkVerP = nMkVerP_.tolist()
    mkVer = utils.bou_to_ver(mkBou)
    mkVerP_ = utils.points_trans(np.array(mkVer), perspecMatrix)
    mkVerP = mkVerP_.tolist()
    nsaveMkVerP = []
    saveMkVerP = []

    for nmvp in nMkVerP:
        if (nmvp[0] < nWinBou[0]) or (nmvp[0] > nWinBou[1]) or (nmvp[1] < nWinBou[2]) or (nmvp[1] > nWinBou[3]):
            nsaveMkVerP.append([-1, -1])
        else:
            nsaveMkVerP.append(nmvp)

    for mvp in mkVerP:
        if (mvp[0] < winBou[0]) or (mvp[0] > winBou[1]) or (mvp[1] < winBou[2]) or (mvp[1] > winBou[3]):
            saveMkVerP.append([-1, -1])
        else:
            saveMkVerP.append([mvp[0]-winBou[0], mvp[1]-winBou[2]])

    for smkv in saveMkVerP:
        if(smkv[0] == -1) or (smkv[1] == -1):
            continue
        else:
            cv2.circle(showImg1, tuple([int(smkv[0]), int(smkv[1])]), 6, (255, 255, 0), -1)

    for nsmkv in nsaveMkVerP:
        if(nsmkv[0] == -1) or (nsmkv[1] == -1):
            continue
        else:
            cv2.circle(showImg1, tuple([int(nsmkv[0]), int(nsmkv[1])]), 6, (0, 255, 0), -1)

    cv2.imshow("nmkCenter", showImg)
    cv2.imshow("nmkVer", showImg1)
    cv2.waitKey()