import cv2
import re
import numpy as np

from shapely.geometry import Polygon


def simPercent_resize(img, scale_percent):
    # 缩放比例
    height, width = img.shape[:2]
    # 计算缩放后的图像宽度和高度
    new_width = int(width * scale_percent)
    new_height = int(height * scale_percent)

    # 缩放图像
    resized_image = cv2.resize(img, (new_width, new_height))
    return resized_image


def get_bou_in_senceCenter(senceLoc, markerSize):
    """
    获取marker处于sence中心位置时，其上下左右所对应的位置
    :param senceLoc: 左边界位置，上边界位置，宽，高
    :param markerSize: 高，宽
    :return:
    """
    bgLeft, bgTop, nbgCol, nbgRow = senceLoc[:]
    winRow, winCol = markerSize[:]
    rBgWinDif, cBgWinDif = nbgRow - winRow, nbgCol - winCol
    rWinBais, cWinBais = rBgWinDif // 2, cBgWinDif // 2
    winLeft, winRight, winTop, winBottom = bgLeft + cWinBais, bgLeft + cWinBais + winCol-1, bgTop + rWinBais, bgTop + rWinBais + winRow-1  # 窗口位置
    return [winLeft, winRight, winTop, winBottom]


def draw_box(img, Bou, color=(255, 0, 0)):
    """
    在img上绘制图像框
    :param img: 被绘制的图像
    :param Loc: 绘制的图像框的左边界，右边界，上边界，下边界
    :return:
    """
    res = img.copy()
    pts = bou_to_ver(Bou)
    for i in range(4):
        cv2.line(res, pts[i], pts[(i+1)%4], color, 2)

    return res


def bou_to_ver(Bou):
    """将边界转为顶点"""
    return [(Bou[0], Bou[2]), (Bou[1], Bou[2]), (Bou[1], Bou[3]), (Bou[0], Bou[3])]


def get_common_region(pointA, pointB):
    # 创建多边形对象
    polygonA = Polygon(pointA)
    polygonB = Polygon(pointB)

    # 计算两个多边形的交集
    intersection = polygonA.intersection(polygonB)

    # 获取交集的边界点坐标
    x, y = intersection.exterior.coords.xy

    pointC = np.vstack((x, y))
    return pointC.T


def viewChange_valid(rawSenceVer_, viewChangeMatrix, winVer, senceLimitVer):
    """
    验证投射变换是否合法，合法的变换要求变换后的场景不会移出观测窗口，也就是说，观测窗口要能够处于变换后的场景中
    :param rawSenceBou: 场景边界
    :param viewChangeMatrix: 投射变换矩阵
    :param winBou: 观测窗口边界
    :return:
    """

    rawSenceVer = np.array(rawSenceVer_)

    vCSenceVer__ = points_trans(rawSenceVer, viewChangeMatrix)
    vCSenceVer0 = vCSenceVer__.tolist()
    vCSenceVer_ = get_common_region(vCSenceVer0, senceLimitVer)
    vCSenceVer = vCSenceVer_.tolist()

    vCSenceVerL = len(vCSenceVer)
    vCSenceEgs = [[vCSenceVer[i], vCSenceVer[(i+1) % vCSenceVerL]] for i in range(vCSenceVerL)]

    Flag = True

    for wver in winVer:
        if not is_point_within_polygon(wver, vCSenceVer, vCSenceEgs):
            Flag = False
            break

    return Flag, vCSenceVer


def viewChange_valid_within(rawSenceVer_, viewChangeMatrix, winVer, senceLimitVer):
    """
    验证投射变换是否合法，合法的变换要求变换后的场景不会移出观测窗口，也就是说，观测窗口要能够处于变换后的marker外面
    :param rawSenceBou: marker边界
    :param viewChangeMatrix: 投射变换矩阵
    :param winBou: 观测窗口边界
    :return:
    """

    rawSenceVer = np.array(rawSenceVer_)

    vCSenceVer__ = points_trans(rawSenceVer, viewChangeMatrix)
    vCSenceVer0 = vCSenceVer__.tolist()
    vCSenceVer_ = get_common_region(vCSenceVer0, senceLimitVer)
    vCSenceVer = vCSenceVer_.tolist()

    winVerL = len(winVer)
    winEgs = [[winVer[i], winVer[(i+1) % winVerL]] for i in range(winVerL)]

    Flag = True

    for ver in vCSenceVer:
        if not is_point_within_polygon(ver, winVer, winEgs):
            Flag = False
            break

    return Flag, vCSenceVer


def is_point_within_polygon(pt, polygon, edges=[]):
    if len(edges) == 0:
        polygonL = len(polygon)
        edges = [[polygon[i], polygon[(i + 1) % polygonL]] for i in range(polygonL)]
    if (is_poi_within_poly(pt, polygon) is True) or (is_poi_in_edges(pt, edges) is True) or (is_poi_in_vertex(pt, polygon) is True):
        return True
    else:
        return False


def points_trans(points_, H_, isInv=False):
    """
    对给定的二维点集进行H_描述的透射变换或投射逆变换
    :param points_: ndarray点集，要求只有两列，每一行的数据分别表示点的x和y
    :param H_: 投射变换矩阵
    :param isRef: 进行的是投射变换还是逆变换
    :return:
    """
    points_1 = np.c_[points_, np.ones(points_.shape[0])]
    H = H_.copy()
    if isInv:
        H = np.linalg.inv(H_)

    points_2 = np.dot(H, points_1.T)
    points_norm = points_2 / points_2[2, :]
    points = points_norm[0:2, :].T

    return points


def is_ray_intersects_segment(poi, s_poi, e_poi):  # [x,y] [lng,lat]
    # 输入：判断点，边起点，边终点，都是[lng,lat]格式数组
    if s_poi[1] == e_poi[1]:  # 排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if s_poi[1] > poi[1] and e_poi[1] > poi[1]:  # 线段在射线上边
        return False
    if s_poi[1] < poi[1] and e_poi[1] < poi[1]:  # 线段在射线下边
        return False
    if s_poi[1] == poi[1] and e_poi[1] > poi[1]:  # 交点为下端点，对应spoint
        return False
    if e_poi[1] == poi[1] and s_poi[1] > poi[1]:  # 交点为下端点，对应epoint
        return False
    # if s_poi[0]<poi[0] and e_poi[1]<poi[1]: #线段在射线左边
    #   return False

    xseg = e_poi[0] - (e_poi[0] - s_poi[0]) * (e_poi[1] - poi[1]) / (e_poi[1] - s_poi[1])  # 求交
    if xseg < poi[0]:  # 交点在射线起点的左侧
        return False
    return True  # 排除上述情况之后


# 射线法判断点是否在任意多边形内部，但是需要注意的是，这种方法对于点在多边形顶点的情况会有问题。
def is_poi_within_poly(poi, poly):
    # 输入：点，多边形二维数组
    # poly=[[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]],[[w1,t1],……[wk,tk]] 二维数组

    # 可以先判断点是否在外包矩形内
    # if not isPoiWithinBox(poi,mbr=[[0,0],[180,90]]): return False
    # 但算最小外包矩形本身需要循环边，会造成开销，本处略去
    sinsc = 0  # 交点个数

    # 循环每条边的曲线->each polygon 是二维数组[[x1,y1],…[xn,yn]]
    polyLen = len(poly)
    for i in range(polyLen):  # [0,len-1]
        s_poi = poly[i]
        e_poi = poly[(i + 1) % polyLen]
        if is_ray_intersects_segment(poi, s_poi, e_poi):
            sinsc += 1  # 有交点就加1

    return True if sinsc % 2 == 1 else False


def is_poi_in_edges(poi, edges, eps=1e-6):
    # 计算向量pa和向量pb
    for edge in edges:
        a = edge[0]
        b = edge[1]
        vector_pa = [poi[0] - a[0], poi[1] - a[1]]
        vector_pb = [poi[0] - b[0], poi[1] - b[1]]

        # 计算向量的叉乘
        cross_product = vector_pa[0] * vector_pb[1] - vector_pa[1] * vector_pb[0]

        # 判断p是否在ab所组成的线段上
        if abs(cross_product) <= eps and (a[0] <= poi[0] <= b[0] or b[0] <= poi[0] <= a[0]) and (
                a[1] <= poi[1] <= b[1] or b[1] <= poi[1] <= a[1]):
            return True

    return False


def is_poi_in_vertex(poi, vertex, eps=1e-2):
    for pv in vertex:
        if (abs(poi[0]-pv[0]) <= eps) and (abs(poi[1]-pv[1]) <= eps):
            return True

    return False


# 读取指定文件，并将文件内容解析为一个 3x3 的 numpy 数组，每个元素都是浮点数
def load_matrix_from_txt(filePath, size):
    # 使用 open() 函数打开指定的文件，并将文件对象赋值给变量 file,使用 with 关键字可以确保文件在处理完后被正确关闭，不需要手动调用 close() 方法
    with open(filePath) as file:
        s = file.read()  # 使用文件对象的 read() 方法读取整个文件内容
        # 使用正则表达式 '\n| ' 对字符串 s 进行拆分，以换行符或空格作为分隔符，得到一个列表 nrs。[:-1] 则是为了去除列表中最后一个空字符串元素
        nrs = re.split('\n| ', s)[:-1]
        nrs = [nr for nr in nrs if nr != '']
        file.close()
        # 使用 np.array() 方法将 nrs 列表转换为一个 numpy 数组，并使用 reshape() 方法将数组形状调整为 3x3 的矩阵。
        # 最后，使用 astype() 方法将数组元素的数据类型转换为浮点数类型
        return np.array(nrs).reshape(size[0], size[1]).astype(np.float)
