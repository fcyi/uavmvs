import cv2
import os
import utils
import numpy as np
import random
import math
import glob


if __name__ == '__main__':
    # 用于产生创建包含增强图像任务所用的参考图像信息的词袋的训练图像
    specRatio = 2  # 期望Marker在窗口中的占比为50%，即窗口面积为Marker面积的specRatio倍，默认背景的面积为窗口面积的specRatio倍
    winSpecRatio = math.sqrt(specRatio)

    testImgPath = "/home/hongqingde/workspace/datasets/AIDataset/marker_voc"

    mksDir = "//home/hongqingde/Downloads/train/IAMarkerHR_mix"
    bgDirs = "/home/hongqingde/workspace/datasets/AIDataset/bgs_room"
    mkImgs = glob.glob(mksDir+"/*.jpeg")
    mkImgs = sorted(mkImgs)
    bgImgs = glob.glob(bgDirs+"/*.jpg")
    bgImgs = sorted(bgImgs)

    changeNums = 10

    # random.seed(42)

    cott = 0

    mkLen, bgLen = len(mkImgs), len(bgImgs)
    for mki in range(mkLen):
        mkImg = cv2.imread(mkImgs[mki])
        if mki > 0:
            break
        for bgi in range(bgLen):
            if bgi > 0:
                break
            bgImg = cv2.imread(bgImgs[bgi])
            filename = mkImgs[mki].split('/')[-1]
            name0 = filename.split('.')[0]
            filename = bgImgs[bgi].split('/')[-1]
            name1 = filename.split('.')[0].split('_')[-1]
            fileName = "{}-{}".format(name0, name1)

            # 获取图像的尺寸
            bgRow, bgCol = bgImg.shape[:2]
            mkRow, mkCol = mkImg.shape[:2]
            winRow, winCol = int(winSpecRatio * mkRow), int(winSpecRatio * mkCol)  # 窗口大小

            nbgRow, nbgCol = specRatio * mkRow, specRatio * mkCol
            if (bgRow < nbgRow) or (bgCol < nbgCol):
                scaleRatio = max(nbgRow / bgRow, nbgCol / bgCol)
                bgImg = utils.simPercent_resize(bgImg, scaleRatio)
                bgRow, bgCol = bgImg.shape[:2]
                nbgRow, nbgCol = min(nbgRow, bgRow), min(nbgCol, bgCol)

            # 选取的背景位置偏移
            rTransBias = random.randint(0, bgRow - nbgRow) if (bgRow != nbgRow) else 0
            cTransBias = random.randint(0, bgCol - nbgCol) if (bgCol != nbgCol) else 0
            bgBou = [cTransBias, cTransBias + nbgCol - 1, rTransBias, rTransBias + nbgRow - 1]  # 偏移后的背景位置

            # 在场景中放置marker
            mkBou = utils.get_bou_in_senceCenter([bgBou[0], bgBou[2], nbgCol, nbgRow],
                                                 [mkRow, mkCol])  # marker位置（marker放在次背景中心）
            winBou = utils.get_bou_in_senceCenter([bgBou[0], bgBou[2], nbgCol, nbgRow],
                                                  [winRow, winCol])  # 窗口位置（窗口位于次背景中心）
            mkCenter = [(mkBou[0] + mkBou[1]) // 2, (mkBou[2] + mkBou[3]) // 2]  # marker在次背景中的放置位置中心

            nWinBou = [0, winCol - 1, 0, winRow - 1]
            nWinVer = utils.bou_to_ver(nWinBou)
            nMkCenter = [mkCenter[0] - winBou[0], mkCenter[1] - winBou[2]]
            nMkCenter_ = [(nWinBou[0] + nWinBou[1]) // 2, (nWinBou[2] + nWinBou[3]) // 2]

            rawSence = bgImg.copy()
            rawSence[mkBou[2]:mkBou[3] + 1, mkBou[0]:mkBou[1] + 1] = mkImg

            senceLimitVer = utils.bou_to_ver([0, bgCol, 0, bgRow])
            winVer = utils.bou_to_ver(winBou)

            nMkBou = [mkBou[0] - winBou[0], mkBou[0] - winBou[0] + mkCol - 1, mkBou[2] - winBou[2],
                      mkBou[2] - winBou[2] + mkRow - 1]  # 窗口中marker的边界位置
            nMkVer = utils.bou_to_ver(nMkBou)

            bias_row = np.array([0, 0, 1])
            vCFac = [0, 0, 0, 0, 0, 0]

            # cfac = ((mkCol // 2)- colTranBias + (winCol // 2)) // mkCol
            # rfac = ((mkRow // 2)- rowTranBias + (winRow // 2)) // mkRow
            # transRation = 1 - (cfac*(1-rfac)+(1-cfac)*rfac+cfac*rfac)  # 平移导致的图像占比，此处不考虑斜切、旋转和缩放的情况
            # marker/win ~ 0.5*[transRation * vCFac[5]_min**2, vCFac[5]_max**2]
            rawSence = cv2.circle(rawSence, (mkCenter[0], mkCenter[1]), 3, (0, 0, 255), -1)
            cv2.imshow("test", rawSence[winBou[2]:winBou[3] + 1, winBou[0]:winBou[1] + 1])
            cv2.waitKey()
            for ci in range(changeNums):
                while (1):
                    perspecMatrix = np.zeros((3, 3), dtype=np.float32)
                    perspecMatrix[[0, 1, 2], [0, 1, 2]] = 1

                    senceCurVer = utils.bou_to_ver([0, bgCol, 0, bgRow])

                    # 随机旋转
                    vCFac[0] = random.randint(-180, 180)  # 随机生成旋转角度
                    rotation_matrix = cv2.getRotationMatrix2D((mkCenter[0], mkCenter[1]), vCFac[0], 1)  # 获取旋转矩阵
                    viewChangeSence = cv2.warpAffine(rawSence, rotation_matrix, (bgCol, bgRow))  # 进行旋转变换

                    rotation_matrix_ = np.vstack((rotation_matrix, bias_row))
                    perspecMatrix = np.matmul(rotation_matrix_, perspecMatrix)
                    Flag, senceCurVer = utils.viewChange_valid(senceCurVer, perspecMatrix, winVer, senceLimitVer)
                    if not Flag:
                        continue

                    # 随机水平斜切
                    vCFac[1] = np.random.uniform(-0.5, 0.5)  # 随机生成水平斜切因子
                    shear_matrix_x = np.array([[1, vCFac[1], -vCFac[1] * mkCenter[1]], [0, 1, 0]],
                                              dtype=np.float32)  # 获取水平斜切矩阵
                    viewChangeSence = cv2.warpAffine(viewChangeSence, shear_matrix_x, (bgCol, bgRow))  # 进行水平斜切变换

                    shear_matrix_x_ = np.vstack((shear_matrix_x, bias_row))
                    perspecMatrix = np.matmul(shear_matrix_x_, perspecMatrix)
                    Flag, senceCurVer = utils.viewChange_valid(senceCurVer, perspecMatrix, winVer, senceLimitVer)
                    if not Flag:
                        continue

                    # 随机垂直斜切
                    vCFac[2] = np.random.uniform(-0.5, 0.5)  # 随机生成垂直斜切因子
                    shear_matrix_y = np.array([[1, 0, 0], [vCFac[2], 1, -vCFac[2]*mkCenter[0]]],
                                              dtype=np.float32)  # 获取垂直斜切矩阵
                    viewChangeSence = cv2.warpAffine(viewChangeSence, shear_matrix_y, (bgCol, bgRow))  # 进行垂直斜切变换

                    shear_matrix_y_ = np.vstack((shear_matrix_y, bias_row))
                    perspecMatrix = np.matmul(shear_matrix_y_, perspecMatrix)
                    Flag, senceCurVer = utils.viewChange_valid(senceCurVer, perspecMatrix, winVer, senceLimitVer)
                    if not Flag:
                        continue

                    # 随机尺度缩放
                    vCFac[5] = np.random.uniform(0.9, 1.1)  # 随机生成缩放因子
                    scale_matrix = np.array([[vCFac[5], 0, 0], [0, vCFac[5], 0]], dtype=np.float32)  # 获取垂直斜切矩阵
                    viewChangeSence = cv2.warpAffine(viewChangeSence, scale_matrix, (bgCol, bgRow))  # 进行垂直斜切变换

                    scale_matrix_ = np.vstack((scale_matrix, bias_row))
                    perspecMatrix = np.matmul(scale_matrix_, perspecMatrix)
                    Flag, senceCurVer = utils.viewChange_valid(senceCurVer, perspecMatrix, winVer, senceLimitVer)
                    if not Flag:
                        continue

                    # 随机平移
                    colTranBias = max((winCol // 2) - (int(mkCol*vCFac[5]) // 2), 0)
                    rowTranBias = max((winRow // 2) - (int(mkRow*vCFac[5]) // 2), 0)
                    vCFac[3] = random.randint(-colTranBias, colTranBias)  # 随机生成水平平移量
                    vCFac[4] = random.randint(-rowTranBias, rowTranBias)  # 随机生成垂直平移量
                    translation_matrix = np.array([[1, 0, vCFac[3]], [0, 1, vCFac[4]]], dtype=np.float32)  # 获取平移矩阵
                    viewChangeSence = cv2.warpAffine(viewChangeSence, translation_matrix, (bgCol, bgRow))  # 进行平移变换

                    translation_matrix_ = np.vstack((translation_matrix, bias_row))
                    perspecMatrix = np.matmul(translation_matrix_, perspecMatrix)
                    Flag, senceCurVer = utils.viewChange_valid(senceCurVer, perspecMatrix, winVer, senceLimitVer)
                    if Flag:
                        break

                res_ = viewChangeSence[winBou[2]:winBou[3] + 1, winBou[0]:winBou[1] + 1]
                # cv2.imwrite(os.path.join(testImgPath, "{}.jpg".format(fileName)), res_)
                cv2.imshow("test", res_)
                cv2.waitKey()

                print(cott)
                cott += 1

