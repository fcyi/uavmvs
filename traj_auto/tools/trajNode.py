import math
import numpy as np
import utils3D
from read_write_model import *
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
import random


class TrajBaseNode(object):
    def __init__(self, xyz, jtDir, trajType=0):
        """
        :param xyz: 轨迹节点的位置，[x, y, z]
        :param jtDir: 飞行期间镜头的朝向，0-沿着行进方向，-1-行进方向左转90度，1-行进方向右转90度
        :param kind: 除最后一个节点外，每两个轨迹节点之间会有一段路线，kind表示节点之后所对应的路线的类型，0表示弧线，1表示直线
        """
        self.xyz = xyz
        self.jtDir = jtDir
        self.tType = trajType


class TypicalNodeList(object):
    def __init__(self, tType=-1, **args):
        """
        :param type: 0-circle, 1-rectangle, 2-ciretangle
        :param args:
        """
        assert tType in [0, 1, 2], 'the type——{} have not been completed'.format(tType)
        self.params = args
        self.tType = tType
        self.nodeLists = []
        needParamsCommon = ['baseHeight', 'regionCenter', 'flyHeight', 'loopNum', 'heightRatio', 'isConnect', 'isRandom', 'isSnake']
        needParamsRec = ['heights', 'widths', 'rtheta']  # 为了防止楼宇的最小外接矩形的边不予坐标轴相平行
        needParamsCir = ['radiuss']
        needParams = needParamsCommon
        if tType == 0:
            needParams += needParamsCir
        elif tType == 1:
            needParams += needParamsRec
        else:
            needParams += needParamsRec + ['rratio']

        self.params_valid_check(needParams)

    def params_valid_check(self, needParams):
        id2name = {0: 'circle', 1: 'rectangle', 2: 'ciretangle'}
        errorParams = []
        paramsKeys = list(self.params.keys())

        for param in paramsKeys:
            if param not in needParams:
                errorParams.append(param)

        self.ret_error_params_info(errorParams, id2name, 0)

        name2default = {'isConnect': False, 'isRandom': False, 'isSnake': False, 'baseHeight': 5}
        name2defaultKeys = list(name2default.keys())
        for needParam in needParams:
            if needParam not in paramsKeys:
                if needParam not in name2defaultKeys:
                    errorParams.append(needParam)
                else:
                    self.params[needParam] = name2default[needParam]

        self.ret_error_params_info(errorParams, id2name, 1)

        assert not (self.params['isconnect'] and self.params['isSnake']), '蛇形轨迹与衔接不能同时开'

    def ret_error_params_info(self, errorParams, id2name, eType):
        errorParamsLen = len(errorParams)
        if errorParamsLen > 0:
            errorParamNames = errorParams[0]
            for i in range(1, errorParamsLen):
                errorParamNames += ', {}'.format(errorParams[i])
            if eType == 0:
                assert errorParamsLen == 0, "param: {} not be needed by traj: {}".format(errorParamNames, id2name[self.tType])
            else:
                assert errorParamsLen == 0, "param: {} should be included by traj: {}".format(errorParamNames, id2name[self.tType])

    def get_nodeList(self):
        random_seed = 42  # 随机种子
        random_generator = random.Random(random_seed)  # 创建带有指定种子的随机数生成器

        baseHeight = self.params['baseHeight']
        regionCenter = self.params['regionCenter']
        flyHeight = self.params['flyHeight']
        loopNum = self.params['loopNum']
        heightRatio = self.params['heightRatio']
        isConnect = self.params['isConnect']
        isRandom = self.params['isRandom']
        isSnake = self.params['isSnake']

        heightRatioLen = len(heightRatio)
        heightRatioAccum = []

        assert (((heightRatioLen+1 == loopNum) and (not isSnake)) or
                ((heightRatioLen+2 == loopNum) and isSnake)), 'heightRation len set meet fault'
        if heightRatioLen > 0:
            heightRatioAccum[0] = heightRatio[0]
            for i in range(1, heightRatioLen):
                heightRatioAccum[i] = heightRatio[i] + heightRatioAccum[i+1]

        heightList = [baseHeight]
        if (loopNum > 2) and isSnake:
            heightRes = flyHeight-baseHeight
            heightList.append(baseHeight)
            heightList += [baseHeight + heightRes*heightRatioAccum[i] for i in range(loopNum - 2)]
        else:
            if loopNum > 1:
                heightRes = flyHeight-baseHeight
                heightList += [baseHeight + heightRes*heightRatioAccum[i] for i in range(loopNum - 1)]
            else:
                if not isSnake:
                    heightList[0] = flyHeight

        if isSnake:
            heightList.append(flyHeight)

        nodeLists = []
        if self.tType == 0:
            radiuss = self.params['radiuss']
            assert len(radiuss) == loopNum, "radiuss set meet error."

            angleLen = 360
            cirAngles = np.linspace(0, 360, angleLen, endpoint=False)
            startIdx = 0 if not isRandom else random_generator.randint(0, 359)
            angleSeg = angleLen // 4

            baseNodeList = []
            for loopIdx in range(loopNum):
                for i in range(4):
                    baseAngle = cirAngles[(startIdx + i * angleSeg) % angleLen]
                    x = radiuss[loopIdx] * np.cos(np.radians(baseAngle))
                    y = radiuss[loopIdx] * np.sin(np.radians(baseAngle))
                    baseNodeList.append([x, y])
            nodeList_ = []
            for loopIdx in range(loopNum):
                heightS = heightList[loopIdx]
                heightE = heightList[loopIdx+1] if isSnake else heightList[loopIdx]
                heightSeg = (heightE-heightS) / 4.
                for i in range(4):
                    nodeList_.append([baseNodeList[loopIdx*4+i][0], baseNodeList[loopIdx*4+i][1], heightS+i*heightSeg])
                nodeLists.append(nodeList_)
                if isConnect and (not isSnake) and (loopIdx+1 != loopNum):
                    nodeLists.append([
                        [baseNodeList[loopIdx*4][0], baseNodeList[loopIdx*4][1], heightS],
                        [baseNodeList[loopIdx*4+4][0], baseNodeList[loopIdx*4+4][1], heightE]
                    ])
        elif self.tType == 1:
            heights, widths = self.params['heights']
            assert (len(heights) == len(widths)) and (len(heights) == loopNum), "heights or widths set meet error"
            baseNodeList = []
            for loopIdx in range(loopNum):
                for i in range(4):
                    y = heights[loopIdx] if i < 2 else 0
                    x = widths[loopIdx] if i % 3 == 0 else 0
                    baseNodeList.append([x, y])


        else:
            pass

        self.nodeLists = nodeLists


class TrajBasetraj(object):
    def __init__(self, nodeList, step, refineStepRatio,
                 regionCenterXYZ=(), isClose=False):
        """
        :param nodeList: 轨迹上所包含的节点列表，若轨迹节点上仅有一个节点，那么只有kind为1才行，
                         此时该轨迹为圆形轨迹，相应的圆心为0，该节点表示圆形轨迹的起始点。
                         因为轨迹涉及到行进方向，其他闭合轨迹难以根据节点列表中点数进行判断。
                         节点只是为了控制轨迹的方向、形状，最终产生的轨迹中不一定会包含节点
        :param regionCenterXYZ: 任何轨迹都是一个相对量，它是基于一定的区域中心考虑的
        :param kind: 目前有两种轨迹，闭合轨迹和非闭合轨迹，用于判断最后一个节点上后是否要跟着一段轨迹
        """
        self.nodeList = nodeList
        self.step = step
        self.refineStepRatio = refineStepRatio
        self.regionCenterXYZ = regionCenterXYZ
        self.isClose = isClose
        self.nodeNum = len(nodeList)
        self.xyzList = []
        self.posList = []

    def get_pos_base_node(self):

        pass

    def get_xyz_base_node(self):
        assert self.nodeNum >= 2, "轨迹节点必须在两个以上，否则产生不了轨迹"

        # 轨迹合法性校验
        # 0-若连续弧线的两端对应的节点所在的路线都非圆弧，那么，这连续圆弧上的节点必须为偶数，否则会出错
        # 1-若最后一个节点为圆弧并且轨迹为闭合形曲线，那么其所在的圆弧节点段上的圆弧节点必须为偶数个，否则会出错
        # 其实，是否会出现这两种错误，可以根据路线是否闭合来划分情况，
        # 而实际上闭合轨迹和非闭合轨迹的唯一区别就在于最后一个点是否需要校验
        isValid = True
        arcPointNum = 0
        checkLen = 0 if self.isClose else -1  # 为了
        checkLen += self.nodeNum
        for idx in range(checkLen):
            if self.nodeList[idx].tType == 0:
                arcPointNum += 1
            if (idx == 0) or (idx == checkLen-1) or (self.nodeList[idx].tType == 1):
                if arcPointNum % 2 == 1:
                    isValid = False
                    break
                arcPointNum = 0

        assert isValid is True, "路线节点设置中类型组成非法"

        pIdx = 0
        resAccum = 0
        while (pIdx < (self.nodeNum-2)) or (pIdx == 0):
            nodeStart = self.nodeList[pIdx]
            trajTmp = []
            if nodeStart.tType == 1:
                pIdx += 1
                nodeEnd = self.nodeList[pIdx]
                trajTmp, resAccum = utils3D.line_traj_xyz(nodeStart.xyz, nodeEnd.xyz,
                                                          resAccum,
                                                          self.step, self.refineStepRatio)
            elif nodeStart.tType == 0:
                pIdx += 1
                nodeMid = self.nodeList[pIdx]
                pIdx += 1
                nodeEnd = self.nodeList[pIdx]
                trajTmp, resAccum = utils3D.circular_traj(nodeStart.xyz, nodeMid.xyz, nodeEnd.xyz,
                                                          resAccum,
                                                          self.step, self.refineStepRatio)

            self.xyzList += trajTmp

        if pIdx == self.nodeNum-1:
            if self.isClose:
                nodeLast0 = self.nodeList[pIdx]
                assert nodeLast0.tType != 0, "轨迹合理性检测不够完善"
                nodeLast1 = self.nodeList[pIdx]
                trajTmp, resAccum = utils3D.line_traj_xyz(nodeLast0.xyz, nodeLast1.xyz,
                                                          resAccum,
                                                          self.step, self.refineStepRatio)
                self.xyzList += trajTmp

        elif pIdx == self.nodeNum-2:
            nodeLast0 = self.nodeList[pIdx]
            nodeLast1 = self.nodeList[pIdx+1]
            nodeLast2 = self.nodeList[0]
            assert nodeLast0.tType == nodeLast1.tType, "轨迹合理性检测不够完善"
            if nodeLast0.tType == 0:
                assert self.isClose, "轨迹合理性检测不够完善"
                trajTmp, resAccum = utils3D.circular_traj(nodeLast0.xyz, nodeLast1.xyz, nodeLast2.xyz,
                                                          resAccum,
                                                          self.step, self.refineStepRatio)
            else:
                trajTmp = []
                trajTmp_, resAccum = utils3D.line_traj_xyz(nodeLast0.xyz, nodeLast1.xyz,
                                                           resAccum,
                                                           self.step, self.refineStepRatio)
                trajTmp += trajTmp_
                trajTmp_, resAccum = utils3D.line_traj_xyz(nodeLast1.xyz, nodeLast2.xyz,
                                                           resAccum,
                                                           self.step, self.refineStepRatio)
                trajTmp += trajTmp_
            self.xyzList += trajTmp


if __name__ == '__main__':

    # trajList, _ = utils3D.line_traj_xyz([0, 0, 0], [0, 0, 1], 0, 0.1, 1)
    # trajList, _ = utils3D.circular_traj([0, 0., 0.], [0., 1., 0.], [0., 1., 2], 0, 0.1)
    # trajList = np.array(trajList).transpose()
    # trajList = trajList.tolist()

    nodeList = [
        TrajBaseNode([0.3, 0, 0], 0, 1),
        TrajBaseNode([0.7, 0, 0], 0, 0),
        TrajBaseNode([0.912, 0.088, 0], 0, 0),
        TrajBaseNode([1, 0.3, 0], 0, 1),
        TrajBaseNode([1, 0.7, 0], 0, 0),
        TrajBaseNode([0.912, 0.912, 0], 0, 0),
        TrajBaseNode([0.7, 1, 0], 0, 1),
        TrajBaseNode([0.3, 1, 0], 0, 0),
        TrajBaseNode([0.088, 0.912, 0], 0, 0),
        TrajBaseNode([0, 0.7, 0], 0, 1),
        TrajBaseNode([0, 0.3, 0], 0, 0),
        TrajBaseNode([0.088, 0.088, 0], 0, 0),
    ]

    trajTest = TrajBasetraj(nodeList, 0.1, 1, isClose=True)
    trajTest.get_xyz_base_node()
    trajList = np.array(trajTest.xyzList).transpose()
    # 创建三维坐标系
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制三维点
    ax.scatter(trajList[0], trajList[1], trajList[2], c='r', marker='o')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 显示图形
    plt.show()

    pass
