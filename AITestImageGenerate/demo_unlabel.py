import cv2
import os
import utils
import numpy as np
import random
import math
import glob


class transConfig(object):
    def __init__(self, pm, pl, winSpec, isTest=False):
        bgScaleMin = 0.2
        bgScaleMax = 0.9
        markerScaleMin = 0.5
        markerScaleMax = 1.2

        bgRotMin = -45
        bgRotMax = 45
        markerRotMin = -180
        markerRotMax = 180

        bgShearXMin = -0.5
        bgShearXMax = 0.5
        markerShearXMin = -0.125
        markerShearXMax = 0.125

        bgShearYMin = -0.5
        bgShearYMax = 0.5
        markerShearYMin = -0.125
        markerShearYMax = 0.125

        if (pm < pl) and (not isTest):
            self.rotMin, self.rotMax = markerRotMin, markerRotMax
            self.shearXMin, self.shearXMax = markerShearXMin, markerShearXMax
            self.shearYMin, self.shearYMax = markerShearYMin, markerShearYMax
            self.scaleMin, self.scaleMax = markerScaleMin, markerScaleMax
            self.winSpecRatio = winSpec
        else:
            self.rotMin, self.rotMax = bgRotMin, bgRotMax
            self.shearXMin, self.shearXMax = bgShearXMin, bgShearXMax
            self.shearYMin, self.shearYMax = bgShearYMin, bgShearYMax
            self.scaleMin, self.scaleMax = bgScaleMin, bgScaleMax
            self.winSpecRatio = winSpec

    rotMin = 0
    rotMax = 0
    shearXMin = 0
    shearXMax = 0
    shearYMin = 0
    shearYMax = 0
    winSpecRatio = 0


def img_show(img, winVer):
    imgS = img.copy()
    winVerLen = len(winVer)
    for ix in range(0, winVerLen):
        pt1I = ix
        pt2I = (ix+1) % winVerLen

        cv2.line(imgS, winVer[pt1I], winVer[pt2I], (0, 255, 255))

    cv2.imshow('test', imgS)
    cv2.waitKey()

def image_transaltion(pm, pl,
                      bgCol, bgRow,
                      nbgCol, nbgRow,
                      mkCol, mkRow,
                      winCol, winRow,
                      bgImg, mkImg,
                      transCfg, isTest=False):
    bias_row = np.array([0, 0, 1])
    vCFac = [0, 0, 0, 0, 0, 0]
    doLimitMax = 1
    perspecMatrix = np.zeros((3, 3), dtype=np.float32)
    while (1):
        # 选取的背景位置偏移
        rTransBias = random.randint(0, bgRow - nbgRow) if (bgRow != nbgRow) else 0
        cTransBias = random.randint(0, bgCol - nbgCol) if (bgCol != nbgCol) else 0
        bgBou = [cTransBias, cTransBias + nbgCol - 1, rTransBias, rTransBias + nbgRow - 1]  # 偏移后的背景位置

        # 在场景中放置marker，这个位置和仿射变换是否容易成功息息相关，当marker变小时就十分容易导致死循环
        mkBou = utils.get_bou_in_senceCenter([bgBou[0], bgBou[2], nbgCol, nbgRow],
                                             [mkRow, mkCol])  # marker位置（marker放在次背景中心）
        winBou = utils.get_bou_in_senceCenter([bgBou[0], bgBou[2], nbgCol, nbgRow],
                                              [winRow, winCol])  # 窗口位置（窗口放在次背景中心）
        mkCenter = [(mkBou[0] + mkBou[1]) // 2, (mkBou[2] + mkBou[3]) // 2]  # marker中心在背景中的放置位置

        nWinBou = [0, winCol - 1, 0, winRow - 1]
        nWinVer = utils.bou_to_ver(nWinBou)
        nMkCenter = [mkCenter[0] - winBou[0], mkCenter[1] - winBou[2]]  # marker中心相对于观测窗口的位置
        nMkCenter_ = [(nWinBou[0] + nWinBou[1]) // 2, (nWinBou[2] + nWinBou[3]) // 2]  # 不考虑背景的情况下，观测窗口的中心

        # 在背景中放置marker
        rawSence = bgImg.copy()
        if pm < pl:
            rawSence[mkBou[2]:mkBou[3] + 1, mkBou[0]:mkBou[1] + 1] = mkImg

        senceLimitVer = utils.bou_to_ver([0, bgCol, 0, bgRow])  # 可观测场景的四个顶点位置，超过这些顶点所构成的凸多边形部分的内容都会被截掉，且不可恢复
        winVer = utils.bou_to_ver(winBou)  # 窗口在场景中的四个顶点位置

        nMkBou = [mkBou[0] - winBou[0], mkBou[0] - winBou[0] + mkCol - 1, mkBou[2] - winBou[2],
                  mkBou[2] - winBou[2] + mkRow - 1]  # marker相对于观测窗口的边界位置
        nMkVer = utils.bou_to_ver(nMkBou)  # marker相对于观测窗口的四个顶点

        perspecMatrix = np.zeros((3, 3), dtype=np.float32)
        perspecMatrix[[0, 1, 2], [0, 1, 2]] = 1

        senceCurVer = utils.bou_to_ver([0, bgCol, 0, bgRow])  # 当前存在内容的多边形区域的顶点

        # img_show(rawSence, winVer)

        # ==============================此处开始进行仿射变换===========================================
        viewChangeSence = rawSence.copy()
        # 随机旋转
        Flag = False
        doCot = 0
        while not Flag and (doCot < doLimitMax):
            doCot += 1
            vCFac[0] = random.randint(transCfg.rotMin, transCfg.rotMax)  # 随机生成旋转角度
            rotation_matrix = cv2.getRotationMatrix2D((mkCenter[0], mkCenter[1]), vCFac[0], 1)  # 获取旋转矩阵
            viewChangeSence = cv2.warpAffine(viewChangeSence, rotation_matrix, (bgCol, bgRow))  # 进行旋转变换

            rotation_matrix_ = np.vstack((rotation_matrix, bias_row))
            Flag, senceCurVer = utils.viewChange_valid(senceCurVer, rotation_matrix_, winVer, senceLimitVer)
            if Flag:
                perspecMatrix = np.matmul(rotation_matrix_, perspecMatrix)
        if not Flag:
            continue

        # img_show(viewChangeSence, winVer)

        # 随机水平斜切
        Flag = False
        doCot = 0
        while not Flag and (doCot < doLimitMax):
            doCot += 1
            vCFac[1] = np.random.uniform(transCfg.shearXMin, transCfg.shearXMax)  # 随机生成水平斜切因子
            shear_matrix_x = np.array([[1, vCFac[1], -vCFac[1] * mkCenter[1]], [0, 1, 0]],
                                      dtype=np.float32)  # 获取水平斜切矩阵
            viewChangeSence = cv2.warpAffine(viewChangeSence, shear_matrix_x, (bgCol, bgRow))  # 进行水平斜切变换

            shear_matrix_x_ = np.vstack((shear_matrix_x, bias_row))
            Flag, senceCurVer = utils.viewChange_valid(senceCurVer, shear_matrix_x_, winVer, senceLimitVer)
            if Flag:
                perspecMatrix = np.matmul(shear_matrix_x_, perspecMatrix)

        if not Flag:
            continue

        # img_show(viewChangeSence,winVer)

        # 随机垂直斜切
        Flag = False
        doCot = 0
        while not Flag and (doCot < doLimitMax):
            doCot += 1
            vCFac[2] = np.random.uniform(transCfg.shearYMin, transCfg.shearYMax)  # 随机生成垂直斜切因子
            shear_matrix_y = np.array([[1, 0, 0], [vCFac[2], 1, -vCFac[2] * mkCenter[0]]],
                                      dtype=np.float32)  # 获取垂直斜切矩阵
            viewChangeSence = cv2.warpAffine(viewChangeSence, shear_matrix_y, (bgCol, bgRow))  # 进行垂直斜切变换

            shear_matrix_y_ = np.vstack((shear_matrix_y, bias_row))
            Flag, senceCurVer = utils.viewChange_valid(senceCurVer, shear_matrix_y_, winVer, senceLimitVer)
            if Flag:
                perspecMatrix = np.matmul(shear_matrix_y_, perspecMatrix)
        if not Flag:
            continue

        # img_show(viewChangeSence, winVer)

        # 随机尺度缩放
        Flag = False
        doCot = 0
        while not Flag and (doCot < doLimitMax):
            doCot += 1
            vCFac[5] = np.random.uniform(transCfg.scaleMin, transCfg.scaleMax)  # 随机生成缩放因子
            scale_matrix = np.array([[vCFac[5], 0, 0], [0, vCFac[5], 0]], dtype=np.float32)  # 获取垂直斜切矩阵
            viewChangeSence = cv2.warpAffine(viewChangeSence, scale_matrix, (bgCol, bgRow))  # 进行垂直斜切变换

            scale_matrix_ = np.vstack((scale_matrix, bias_row))
            Flag, senceCurVer = utils.viewChange_valid(senceCurVer, scale_matrix_, winVer, senceLimitVer)
            if Flag:
                perspecMatrix = np.matmul(scale_matrix_, perspecMatrix)
        if not Flag:
            continue

        # img_show(viewChangeSence, winVer)

        # 随机平移
        if (pm < pl) and (not isTest):
            # 此处主要考虑marker在进行投射变换后的四个角点位置，期望这些点在通过仿射变换后不会跑到观测窗口外部
            mkCurver = utils.bou_to_ver(mkBou)
            Flag, mkCurver = utils.viewChange_valid_within(mkCurver, perspecMatrix, winVer, senceLimitVer)
            if not Flag:
                continue

            colTranBias = bgCol
            rowTranBias = bgRow

            for curPt in mkCurver:
                colTranBias = min(colTranBias, abs(curPt[0] - winBou[0]))
                colTranBias = min(colTranBias, abs(curPt[0] - winBou[1]))
                rowTranBias = min(rowTranBias, abs(curPt[1] - winBou[2]))
                rowTranBias = min(rowTranBias, abs(curPt[1] - winBou[3]))
        else:
            colTranBias = max((int(winCol * vCFac[5]) // 4), 0)
            rowTranBias = max((int(winRow * vCFac[5]) // 4), 0)

        colTranBias = int(colTranBias)
        rowTranBias = int(rowTranBias)

        Flag = False
        doCot = 0
        while not Flag and (doCot < doLimitMax):
            doCot += 1
            vCFac[3] = random.randint(-colTranBias, colTranBias)  # 随机生成水平平移量
            vCFac[4] = random.randint(-rowTranBias, rowTranBias)  # 随机生成垂直平移量
            translation_matrix = np.array([[1, 0, vCFac[3]], [0, 1, vCFac[4]]], dtype=np.float32)  # 获取平移矩阵
            viewChangeSence = cv2.warpAffine(viewChangeSence, translation_matrix, (bgCol, bgRow))  # 进行平移变换

            translation_matrix_ = np.vstack((translation_matrix, bias_row))
            Flag, senceCurVer = utils.viewChange_valid(senceCurVer, translation_matrix_, winVer, senceLimitVer)

            if Flag:
                perspecMatrix = np.matmul(translation_matrix_, perspecMatrix)
        # img_show(viewChangeSence, winVer)
        if Flag:
            break

    return perspecMatrix, viewChangeSence, winBou, mkBou


if __name__ == '__main__':
    # 用于产生创建包含增强图像任务所用的参考图像信息的词袋的训练图像
    specRatio = 2  # 期望Marker在窗口中的占比为50%，即窗口面积为Marker面积的specRatio倍，默认背景的面积为窗口面积的specRatio倍
    winSpecRatio = math.sqrt(specRatio)

    # 由于背景存在大量的平滑区域，所以建议对其进行收缩
    # 为了让背景部分的特征点尽可能容易出现，旋转角度，斜切程度都应该有所缓和，观测窗口也应该有所增大

    changeNums = 1

    mksDir = "//home/hongqingde/Downloads/train/IAMarkerHR_mix"
    bgDirs = "/home/hongqingde/workspace/datasets/AIDataset/bgs_summary"
    mkImgs = glob.glob(mksDir + "/*.jpeg")
    mkImgs = sorted(mkImgs)
    bgImgs = glob.glob(bgDirs + "/*.jpg")
    bgImgs = sorted(bgImgs)

    mkLen, bgLen = len(mkImgs), len(bgImgs)

    isTest = True  # 生成的是训练集还是测试集

    totalNum = 10000
    ratioName = True
    isRepeat = False

    plList = [0.75]

    for pl in plList:
        if not isTest:
            testImgPath = "/home/hongqingde/workspace/datasets/AIDataset/marker_voc_{}".format(int(pl*100))
            testMaskPath = "/home/hongqingde/workspace/datasets/AIDataset/marker_voc_{}_mask".format(int(pl*100))
            bgPaths = bgImgs
            random.seed(42)
            np.random.seed(42)
        else:
            testImgPath = "/home/hongqingde/workspace/datasets/AIDataset/test_voc_{}".format(int(pl * 100))
            testMaskPath = "/home/hongqingde/workspace/datasets/AIDataset/test_voc_{}_mask".format(int(pl * 100))
            random.seed(60)
            np.random.seed(60)

            assert totalNum > mkLen, "totalNum must be larger than mkLen"
            selectNum = totalNum // mkLen
            if selectNum <= bgLen:
                bgPaths = random.sample(bgImgs, selectNum)
                isRepeat = False
            else:
                bgPaths = random.choices(bgImgs, k=selectNum)
                isRepeat = True

        if not os.path.exists(testImgPath):
            os.makedirs(testImgPath)

        if not os.path.exists(testMaskPath):
            os.makedirs(testMaskPath)

        cott = 0

        for mki in range(mkLen):
            mkImg_ = cv2.imread(mkImgs[mki])

            genNum = 0
            for bgPath in bgPaths:
                mkScal = np.random.uniform(0.8, 1.0)
                mkImg = utils.simPercent_resize(mkImg_, mkScal)

                pm = np.random.uniform(0, 1.0)
                transCfg = transConfig(pm, pl, winSpecRatio, isTest)

                bgImg = cv2.imread(bgPath)
                filename = mkImgs[mki].split('/')[-1]
                name0 = filename.split('.')[0]
                filename = bgPath.split('/')[-1]
                name1 = filename.split('.')[0].split('_')[-1]

                fileName = "{}-{}".format(name0, name1)

                # 获取图像的尺寸
                bgRow, bgCol = bgImg.shape[:2]
                mkRow, mkCol = mkImg.shape[:2]
                rmkRow, rmkCol = mkImg_.shape[:2]
                winRow, winCol = int(transCfg.winSpecRatio * rmkRow), int(transCfg.winSpecRatio * rmkCol)  # 窗口大小
                winRow = max(rmkRow, min(winRow, 1000))
                winCol = max(rmkCol, min(winCol, 1000))

                nbgRow, nbgCol = specRatio * rmkRow, specRatio * rmkCol
                if (bgRow < nbgRow) or (bgCol < nbgCol):
                    scaleRatio = max(nbgRow / bgRow, nbgCol / bgCol)
                    bgImg = utils.simPercent_resize(bgImg, scaleRatio)
                    bgRow, bgCol = bgImg.shape[:2]
                    nbgRow, nbgCol = min(nbgRow, bgRow), min(nbgCol, bgCol)

                # cfac = ((mkCol // 2)- colTranBias + (winCol // 2)) // mkCol
                # rfac = ((mkRow // 2)- rowTranBias + (winRow // 2)) // mkRow
                # transRation = 1 - (cfac*(1-rfac)+(1-cfac)*rfac+cfac*rfac)  # 平移导致的图像占比，此处不考虑斜切、旋转和缩放的情况
                # marker/win ~ 0.5*[transRation * vCFac[5]_min**2, vCFac[5]_max**2]
                for ci in range(changeNums):
                    perspecMatrix, viewChangeSence, winBou, mkBou = image_transaltion(pm, pl, bgCol, bgRow, nbgCol, nbgRow,
                                                                                      mkCol, mkRow,
                                                                                      winCol, winRow,
                                                                                      bgImg, mkImg,
                                                                                      transCfg, isTest)

                    res_ = viewChangeSence[winBou[2]:winBou[3] + 1, winBou[0]:winBou[1] + 1]

                    maskSence = np.zeros(bgImg.shape, np.uint8)
                    if pm < pl:
                        maskSence[mkBou[2]:mkBou[3] + 1, mkBou[0]:mkBou[1] + 1] = 255
                        maskSence = cv2.warpAffine(maskSence, perspecMatrix[0:2, :], (bgCol, bgRow))
                    mask_ = maskSence[winBou[2]:winBou[3] + 1, winBou[0]:winBou[1] + 1]

                    fileNameFix = ""
                    if not ratioName:
                        if not isRepeat:
                            fileNameFix = fileName
                        else:
                            fileNameFix = "{}-{}".format(fileName, genNum)
                    else:
                        chls = 1
                        mskWid, mskHei = mask_.shape[:2]
                        if len(mask_.shape) == 3:
                            chls = mask_.shape[2]
                        nonzero_indices = np.nonzero(mask_)
                        nonzero_count = len(nonzero_indices[0])
                        areaRatio = float(nonzero_count) / float(chls*mskHei*mskWid)
                        areaRatio_ = int(areaRatio*100)
                        if not isRepeat:
                            fileNameFix = "{}-{}".format(fileName, areaRatio_)
                        else:
                            fileNameFix = "{}-{}".format(fileName, genNum, areaRatio_)

                    cv2.imwrite(os.path.join(testImgPath, "{}.jpg".format(fileNameFix)), res_)
                    cv2.imwrite(os.path.join(testMaskPath, "{}.jpg".format(fileNameFix)), mask_)

                    print(cott)
                    cott += 1
                    genNum += 1

