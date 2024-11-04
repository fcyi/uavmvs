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

    testImgPath = "/home/hongqingde/workspace/datasets/AIDataset/test"

    mksDir = "/home/hongqingde/Downloads/AugumentImageMarker/AIMarker_rename"
    bgDirs = "/home/hongqingde/workspace/datasets/AIDataset/bgs_"
    mkImgs = glob.glob(mksDir+"/*.jpg")
    mkImgs = sorted(mkImgs)
    bgImgs = glob.glob(bgDirs+"/*.jpg")
    bgImgs = sorted(bgImgs)

    senceNums = 36

    changeNums = 5

    random.seed(42)
    mkImgIdxs = sample(range(len(mkImgs)), senceNums)  # 产生不可重复的mkNums个整数，整数属于[0, len(mkImgs)]
    bgImgIdxs = [random.randint(0, len(bgImgs)-1) for _ in range(senceNums)]  # 生成可重复出现的随机整数

    senceIdxs = zip(mkImgIdxs, bgImgIdxs)

    preFix = "v"

    for mki, bgi in senceIdxs:
        mkImg = cv2.imread(mkImgs[mki])
        bgImg = cv2.imread(bgImgs[bgi])

        filename = mkImgs[mki].split('/')[-1]
        name0 = filename.split('.')[0]
        filename = bgImgs[bgi].split('/')[-1]
        name1 = filename.split('.')[0]

        folderName = "{}-{}-{}".format(preFix, name0, name1)
        folderPath = os.path.join(testImgPath, folderName)
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
            print("Folder_{} created successfully".format(folderName))
        else:
            print("当前变换以及场景均已考虑过，请勿重复")
            continue

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
        nMkCenter = [mkCenter[0] - winBou[0], mkCenter[1] - winBou[2]]
        nMkCenter_ = [(nWinBou[0] + nWinBou[1]) // 2, (nWinBou[2] + nWinBou[3]) // 2]

        if(nMkCenter[0] != nMkCenter_[0]) or (nMkCenter[1] != nMkCenter_[1]):
            print("marker中心有所矛盾")

        rawSence = bgImg.copy()
        rawSence[mkBou[2]:mkBou[3]+1, mkBou[0]:mkBou[1]+1] = mkImg

        # cv2.imshow("test", rawSence)
        # cv2.waitKey()

        senceLimitVer = utils.bou_to_ver([0, bgCol, 0, bgRow])
        winVer = utils.bou_to_ver(winBou)

        nMkBou = [mkBou[0]-winBou[0], mkBou[0]-winBou[0]+mkCol-1, mkBou[2]-winBou[2], mkBou[2]-winBou[2]+mkRow-1]  # 窗口中marker的边界位置
        nMkVer = utils.bou_to_ver(nMkBou)

        cv2.imwrite(os.path.join(folderPath, "{}.jpg".format(0)), mkImg)
        cv2.imwrite(os.path.join(folderPath, "{}.jpg".format(1)), rawSence[winBou[2]:winBou[3]+1, winBou[0]:winBou[1]+1])
        perspecMatrix = np.array([[1, 0, nMkBou[0]],
                                          [0, 1, nMkBou[2]],
                                          [0, 0, 1]], dtype=np.float32)
        np.savetxt(os.path.join(folderPath, "H_{}_{}.txt".format(0, 1)), perspecMatrix, fmt="%.4f")
        np.savetxt(os.path.join(folderPath, "kpl_{}.txt".format(1)), np.array(nMkVer, dtype=np.float32), fmt="%.2f")
        np.savetxt(os.path.join(folderPath, "mkCenter_{}.txt".format(1)), np.array(nMkVer, dtype=np.float32), fmt="%.2f")
        np.savetxt(os.path.join(folderPath, "nmkCenter_{}.txt".format(1)), np.array(nMkVer, dtype=np.float32), fmt="%.2f")

        bias_row = np.array([0, 0, 1])
        vCFac = [0, 0, 0, 0, 0, 0]

        for ci in range(changeNums):
            while(1):
                perspecMatrix = np.zeros((3, 3), dtype=np.float32)
                perspecMatrix[[0, 1, 2], [0, 1, 2]] = 1

                senceCurVer = utils.bou_to_ver([0, bgCol, 0, bgRow])

                # 随机旋转
                vCFac[0] = random.randint(-180, 180)  # 随机生成旋转角度
                rotation_matrix = cv2.getRotationMatrix2D((mkCenter[0], mkCenter[1]), vCFac[0], 1)  # 获取旋转矩阵
                viewChangeSence = cv2.warpAffine(rawSence, rotation_matrix, (bgCol, bgRow))  # 进行旋转变换

                rotation_matrix_ = np.vstack((rotation_matrix, bias_row))
                perspecMatrix = np.matmul(rotation_matrix_, perspecMatrix)
                Flag, senceCurVer = utils.viewChange_valid(senceCurVer, rotation_matrix_, winVer, senceLimitVer)
                if not Flag:
                    continue

                # 随机水平斜切
                vCFac[1] = np.random.uniform(-1.0, 1.0)  # 随机生成水平斜切因子
                shear_matrix_x = np.array([[1, vCFac[1], -vCFac[1] * mkCenter[1]], [0, 1, 0]], dtype=np.float32)  # 获取水平斜切矩阵
                viewChangeSence = cv2.warpAffine(viewChangeSence, shear_matrix_x, (bgCol, bgRow))  # 进行水平斜切变换

                shear_matrix_x_ = np.vstack((shear_matrix_x, bias_row))
                perspecMatrix = np.matmul(shear_matrix_x_, perspecMatrix)
                Flag, senceCurVer = utils.viewChange_valid(senceCurVer, shear_matrix_x_, winVer, senceLimitVer)
                if not Flag:
                    continue

                # 随机垂直斜切
                vCFac[2] = np.random.uniform(-1.0, 1.0)  # 随机生成垂直斜切因子
                shear_matrix_y = np.array([[1, 0, 0], [vCFac[2], 1, -vCFac[2] * mkCenter[0]]], dtype=np.float32)  # 获取垂直斜切矩阵
                viewChangeSence = cv2.warpAffine(viewChangeSence, shear_matrix_y, (bgCol, bgRow))  # 进行垂直斜切变换

                shear_matrix_y_ = np.vstack((shear_matrix_y, bias_row))
                perspecMatrix = np.matmul(shear_matrix_y_, perspecMatrix)
                Flag, senceCurVer = utils.viewChange_valid(senceCurVer, shear_matrix_y_, winVer, senceLimitVer)
                if not Flag:
                    continue

                # 随机平移
                vCFac[3] = random.randint(-(winCol // 2), (winCol // 2))  # 随机生成水平平移量
                vCFac[4] = random.randint(-(winRow // 2), (winRow // 2))  # 随机生成垂直平移量
                translation_matrix = np.array([[1, 0, vCFac[3]], [0, 1, vCFac[4]]], dtype=np.float32)  # 获取平移矩阵
                viewChangeSence = cv2.warpAffine(viewChangeSence, translation_matrix, (bgCol, bgRow))  # 进行平移变换

                translation_matrix_ = np.vstack((translation_matrix, bias_row))
                perspecMatrix = np.matmul(translation_matrix_, perspecMatrix)
                Flag, senceCurVer = utils.viewChange_valid(senceCurVer, translation_matrix_, winVer, senceLimitVer)
                if not Flag:
                    continue

                # 随机尺度缩放
                vCFac[5] = np.random.uniform(0.5, 1.2)  # 随机生成缩放因子
                scale_matrix = np.array([[vCFac[5], 0, 0], [0, vCFac[5], 0]], dtype=np.float32)  # 获取垂直斜切矩阵
                viewChangeSence = cv2.warpAffine(viewChangeSence, scale_matrix, (bgCol, bgRow))  # 进行垂直斜切变换

                scale_matrix_ = np.vstack((scale_matrix, bias_row))
                perspecMatrix = np.matmul(scale_matrix_, perspecMatrix)
                Flag, senceCurVer = utils.viewChange_valid(senceCurVer, scale_matrix_, winVer, senceLimitVer)
                if Flag:
                    break

            # for i, vcs in enumerate(vSL):
            #     # res_ = viewChangeSence[winBou[2]:winBou[3], winBou[0]:winBou[1]]
            #     res = utils.draw_box(vcs, bgBou, (0, 0, 255))
            #     res = utils.draw_box(res, winBou, (255, 0, 255))
            #
            #     cv2.imshow("test {}".format(i), res)

            res_ = viewChangeSence[winBou[2]:winBou[3]+1, winBou[0]:winBou[1]+1]
            # cv2.imshow("test", res_)
            cv2.imwrite(os.path.join(folderPath, "{}.jpg".format(ci+2)), res_)

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

            for smvp, nsmvp in zip(saveMkVerP, nsaveMkVerP):
                if(abs(smvp[0] - nsmvp[0]) > 1e-2) or (abs(smvp[1] - nsmvp[1]) > 1e-2):
                    print("note, {} could have error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(ci+2))
                    break

            np.savetxt(os.path.join(folderPath, "H_{}_{}.txt".format(1, ci+2)), nPerspecMatrix, fmt="%.4f")
            np.savetxt(os.path.join(folderPath, "kpl_{}.txt".format(ci+2)), saveMkVerP, fmt="%.2f")
            np.savetxt(os.path.join(folderPath, "kpln_{}.txt".format(ci + 2)), nsaveMkVerP, fmt="%.2f")
            np.savetxt(os.path.join(folderPath, "svp_{}.txt".format(ci + 2)), vCFac, fmt="%.2f")















