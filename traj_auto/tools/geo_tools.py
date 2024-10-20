import math
import copy
import random

import cv2
import numpy as np

import base_tools as btls


# ==============================================基本计算=================================================================
def get_center_bias(centerx, centery, cols, rows):
    return [centerx-cols//2, centery-rows//2]


def get_angle(arc_length, radius):
    angle = (arc_length / radius)
    return angle


def distance_p1p2(pt1_, pt2_):
    assert len(pt1_) == len(pt2_)
    ptLen_ = len(pt1_)
    dis_ = 0
    for pIdx_ in range(ptLen_):
        dis_ += (pt1_[pIdx_]-pt2_[pIdx_])**2

    return math.sqrt(dis_)


def line_from_points(p1_, p2_):
    """ 从两个点计算直线的系数 A, B, C 使得 Ax + By + C = 0 """
    A_ = p2_[1] - p1_[1]  # y2 - y1
    B_ = p1_[0] - p2_[0]  # x1 - x2
    C_ = A_ * p1_[0] + B_ * p1_[1]  # A*x1 + B*y1
    return A_, B_, -C_  # 返回 A, B, C


def intersection_of_lines(line1_, line2_):
    """ 解两条直线的交点，line1和line2分别是(A1, B1, C1)和(A2, B2, C2) """
    A1_, B1_, C1_ = line1_
    A2_, B2_, C2_ = line2_

    # 计算行列式
    det_ = A1_ * B2_ - A2_ * B1_
    if det_ == 0:
        return None  # 直线平行，没有交点
    else:
        x = (B1_ * C2_ - B2_ * C1_) / det_
        y = (A2_ * C1_ - A1_ * C2_) / det_
        return [x, y]


# p2 = p0 + t * (p1-p0)
def get_mediate_pt(p0_, p1_, t_):
    return [p0_[0]+t_*(p1_[0]-p0_[0]), p0_[1]+t_*(p1_[1]-p0_[1])]


# 叉积计算
#   对于凸包而言，v1 = p2-p1, v2 = p3-p1，通过计算叉乘计算 3 点方向
#   如果叉乘结果 = 0 , 则说明 p1/p2/p3 共线
#   如果叉乘结果 > 0 , 则为顺时针方向
#   如果叉乘结果 < 0 , 则为逆时针方向
def cross_product(v1, v2):
    return v1[0]*v2[1] - v1[1]*v2[0]


def calc_point_inline_with_y(pt1_, pt2_, y_):
    # 计算连接pt1和pt2的线段上，y坐标为y的点
    s_ = pt1_[1] - pt2_[1]
    if not np.isclose(s_, 0):
        x_ = (y_-pt1_[1])*(pt1_[0]-pt2_[0]) / s_ + pt1_[0]
    else:
        return None

    # 判断x是否在pt1、pt2在x轴的投影里，不是的话返回false
    minX_, maxX_ = min(pt1_[0], pt2_[0]), max(pt1_[0], pt2_[0])
    if (x_ < minX_) or (x_ > maxX_):
        return None

    return [x_, y_]


def calc_cross_degree(v1_, v2_):
    print(np.dot(v1_, v2_) / (np.linalg.norm(v1_)*np.linalg.norm(v2_)))
    return np.degrees(np.arccos(np.dot(v1_, v2_) / (np.linalg.norm(v1_)*np.linalg.norm(v2_))))


def calc_cross_degree_based_point(sPt_, mPt_, ePt_):
    v1_ = [sPt_[0]-mPt_[0], sPt_[1]-mPt_[1]]
    v2_ = [ePt_[0]-mPt_[0], ePt_[1]-mPt_[1]]
    return calc_cross_degree(v1_, v2_)


# #########################################仿射变换#################################################################3

def rotate_coordinate(coordinate, a, b, alpha):
    # 将二维点绕着(a, b)顺时针或逆时针旋转alpha度
    # 确保传入的转角信息为角度
    alpha_rad = math.radians(alpha)  # 将角度转换为弧度
    cos_alpha = math.cos(alpha_rad)
    sin_alpha = math.sin(alpha_rad)

    x, y = coordinate  # 拆分坐标元组
    x -= a
    y -= b

    rx = x * cos_alpha + y * sin_alpha
    ry = -x * sin_alpha + y * cos_alpha

    rotated_x = rx + a
    rotated_y = ry + b
    rotated_coord = [rotated_x, rotated_y]  # 组合旋转后的坐标
    return rotated_coord


def rotate_coordinates(coordinates, a, b, alpha):
    # 将二维点绕着(a, b)顺时针或逆时针旋转alpha度
    # 确保传入的转角信息为角度
    rotated_coordinates = []
    alpha_rad = math.radians(alpha)  # 将角度转换为弧度
    cos_alpha = math.cos(alpha_rad)
    sin_alpha = math.sin(alpha_rad)
    for coord in coordinates:
        x, y = coord  # 拆分坐标元组
        x -= a
        y -= b
        rx = x * cos_alpha + y * sin_alpha
        ry = -x * sin_alpha + y * cos_alpha
        rotated_coordinates.append([rx+a, ry+b])  # 组合旋转后的坐标
    return rotated_coordinates


def get_rotate_vec(src_vec, theta, axis):
    # 将3维坐标向量沿着第axis个坐标向量顺时针旋转theta度
    res_vec = [0, 0, 0]
    rad = np.radians(theta)
    crad = np.cos(rad)
    srad = np.sin(rad)
    if axis == 0:
        res_vec[0] = src_vec[0]
        res_vec[1] = crad*src_vec[1]-srad*src_vec[2]
        res_vec[2] = srad*src_vec[1]+crad*src_vec[2]
    elif axis == 1:
        res_vec[0] = crad * src_vec[0] + srad * src_vec[2]
        res_vec[1] = src_vec[1]
        res_vec[2] = -srad * src_vec[0] + crad * src_vec[2]
    elif axis == 2:
        res_vec[0] = crad*src_vec[0]-srad*src_vec[1]
        res_vec[1] = srad*src_vec[0]+crad*src_vec[1]
        res_vec[2] = src_vec[2]
    return res_vec


def get_rotate_vec_fast(src_vec, thetas):
    rad0, rad1, rad2 = np.radians(thetas[0]), np.radians(thetas[1]), np.radians(thetas[2])
    c0, c1, c2 = np.cos(rad0), np.cos(rad1), np.cos(rad2)
    s0, s1, s2 = np.sin(rad0), np.sin(rad1), np.sin(rad2)
    res_vec = [0, 0, 0]
    res_vec[0] = c1*c2*src_vec[0] - c1*s2*src_vec[1] + s1*src_vec[2]
    res_vec[1] = (s0*s1*c2+c0*s2)*src_vec[0] + (c0*c2-s0*s1*s2)*src_vec[1] - s0*c1*src_vec[2]
    res_vec[2] = (s0*s2-c0*s1*c2)*src_vec[0] + (c0*s1*s2+s0*c2)*src_vec[1] + c0*c1*src_vec[2]
    return res_vec


def transform(pt, trans, theta, scale=(1, 1)):
    # 计算点pt绕着点trans逆时针theta度，之后再放大Scale倍
    thetaRad = np.radians(theta)
    sx, sy = scale[:]
    x, y = pt[:]
    tx, ty = trans[:]
    return [
        sx*((x-tx)*math.cos(thetaRad) - (y-ty)*math.sin(thetaRad)) + tx,
        sy*((x-tx)*math.sin(thetaRad) + (y-ty)*math.cos(thetaRad)) + ty
    ]


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=圆心计算++++++++++++++++++++++++++++++++++++++++++++++=
def calculate_circle_center(pts, radius, max_iterations=1000, tolerance=1e-6):
    points = np.array(pts)
    center = np.mean(points, axis=0)  # 初始近似圆心为所有点的平均位置

    for _ in range(max_iterations):
        distances = np.linalg.norm(points - center, axis=1)  # 计算每个点到近似圆心的距离
        difference = distances - radius  # 距离与给定半径之间的差异

        if np.all(np.abs(difference) < tolerance):  # 如果差异小于容忍度，则认为已经找到圆心
            break

        delta = difference / distances  # 计算每个点对于圆心的位移比例
        center += np.mean(delta[:, np.newaxis] * (points - center), axis=0)  # 更新圆心位置

    return center.tolist()


def calculate_circle_intersection(p1, p2, r1, r2):
    x1, y1 = p1[:]
    x2, y2 = p2[:]
    distance = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

    if (distance <= 1e-6) or (distance > (r1 + r2)):
        # 两个圆相离或相合，可能有0个交点或无数个交点
        return None
    elif (distance == r1 + r2) or (distance == math.fabs(r1-r2)):
        # 两个圆相切（外切或内切），有一个交点
        x_intersect = x1 + r1 * (x2 - x1) / distance
        y_intersect = y1 + r1 * (y2 - y1) / distance
        return [[x_intersect, y_intersect]]
    else:
        # distance > (r1 + r2)
        if distance < math.fabs(r1-r2):
            return None
        elif distance > math.fabs(r1-r2):
            # 两个圆相交，有两个交点
            l1 = (r1 ** 2 + distance ** 2 - r2 ** 2) / (2 * distance)
            l2 = math.sqrt(r1**2 - l1**2)
            px_ = x1 + l1*(x2-x1)/distance
            py_ = y1 + l1 * (y2 - y1) / distance
            px1 = px_-l2*(y2-y1)/distance
            py1 = py_+l2*(x2-x1)/distance
            px2 = px_ + l2 * (y2 - y1) / distance
            py2 = py_ - l2 * (x2 - x1) / distance
            return [[px1, py1], [px2, py2]]


def calculate_circle_center_fast(pts, radius):
    # 基于几何方法计算一系列点所在的圆
    # 输入的点最好根据一定的规则排好序，不排序也行，排序后续可以进一步优化
    ptsLen = len(pts)
    assert ptsLen >=3, "输入的点少于3个，圆心求解存在歧义"

    pt1 = pts[0]
    pt2 = pts[ptsLen-1]
    pth = pts[ptsLen//2]
    centerL = calculate_circle_intersection(pt1, pt2, radius, radius)
    assert centerL is not None, "输入的在圆上的点有问题"

    centerLen = len(centerL)
    assert centerLen < 3, "两圆交点算法有问题"

    if centerLen == 1:
        return centerL[0]
    else:
        c0 = centerL[0]
        c1 = centerL[1]
        distance0 = math.sqrt((c0[0]-pth[0])*(c0[0]-pth[0]) + (c0[1]-pth[1])*(c0[1]-pth[1]))
        distance1 = math.sqrt((c1[0]-pth[0])*(c1[0]-pth[0]) + (c1[1]-pth[1])*(c1[1]-pth[1]))
        if distance0 < distance1:
            return c1
        else:
            return c0


# ==========================================================多边形相关==================================================


# 角排序函数
def angle_sort(points):
    # 点集个数必须大于 3 个
    if len(points) < 2:
        return points

    # 求点集的极角
    def polar_angle(p0, p):
        return math.atan2(p[1] - p0[1], p[0] - p0[0])

    # 计算两点之间的距离的平方
    def distance_squared(p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    # 找到最左下方的点作为起始点
    # p0 点作为极坐标的原点
    p0 = min(points, key=lambda p: (p[1], p[0]))
    # 按照极角和距离的平方排序
    # 先按照极角进行排序 如果极角相同 则按照 p0 点到该点的距离排序
    sorted_points = sorted(points, key=lambda p: (polar_angle(p0, p), distance_squared(p0, p)))
    # 返回按照极角进行排序的 Point 集合
    return [p0] + sorted_points[1:]


# Graham 扫描法找凸包
def graham_scan(points):
    assert len(points) >= 3, "求解凸包的点数必须大于3"

    # 进行角排序
    sorted_points = angle_sort(points)
    # 栈数据结构
    stack = []
    for p in sorted_points:
        # 如果栈中的元素大于 2 个开始执行循环, 计算 p 点 在 栈顶两个元素(倒数第二个在前,倒数第一个再后)组成的向量的左侧还是右侧
        while len(stack) >= 2:
            v1 = [stack[-1][0] - stack[-2][0], stack[-1][1] - stack[-2][1]]
            v2 = [p[0] - stack[-2][0], p[1] - stack[-2][1]]
            if cross_product(v1, v2) <= 0:
                # 如果 p 点在栈顶两个元素组成的向量的左侧 则说明该点是凸边中的点 , 栈顶元素不是凸边中的点 , 将栈顶出栈 , 将本元素入栈
                stack.pop()
            else:
                break
        # 向栈中添加新元素
        stack.append(p)
    return stack


def tu_polygon_gen(n, radius):
    # 给定外接圆，随机生成一个n条边的凸多边形
    random.seed(12345)
    samNums = n*10
    seqL = random.sample(range(0, samNums), n)
    seqL = sorted(seqL)

    seqLN = [seqL[i]/samNums for i in range(n)]
    vertexs = []

    for idx in range(n):
        x = radius * math.cos(2 * math.pi * seqLN[idx])
        y = radius * math.sin(2 * math.pi * seqLN[idx])
        vertexs.append([x, y])

    return vertexs


# 多边形外接框
def create_poly_bounds(vertexs, type='sim rectangle'):
    if type == 'sim rectangle':
        # 求取多边形的外接矩形，从而计算航线与多边形边界的交点
        ver_y, ver_x = [], []
        min_x, max_x, min_y, max_y = np.inf, 0, np.inf, 0
        for ver in vertexs:
            ver_y.append(ver[1])
            ver_x.append(ver[0])

            if ver[0] < min_x:
                min_x = ver[0]
            if ver[0] > max_x:
                max_x = ver[0]
            if ver[1] < min_y:
                min_y = ver[1]
            if ver[1] > max_y:
                max_y = ver[1]

        cent = [float(max_x+min_x)/2., float(max_y+min_y)/2.]

        return [
            [min_x, max_y],
            [max_x, max_y],
            [max_x, min_y],
            [min_x, min_y]
        ], cent
    elif type == 'min rectangle':  # 最小外接矩形
        vertexs_arr = np.array([[ver] for ver in vertexs], dtype=np.float32)
        min_rec = cv2.minAreaRect(vertexs_arr)
        min_box = cv2.boxPoints(min_rec).tolist()
        return [
            min_box[0], min_box[3],
            min_box[2], min_box[1]
        ], list(min_rec[0])
    elif type == 'min circle':  # 最小外接圆
        vertexs_arr = np.array([[ver] for ver in vertexs], dtype=np.float32)
        (x, y), radius = cv2.minEnclosingCircle(vertexs_arr)
        return radius, [x, y]
    else:
        raise AssertionError('多边形外接框类型定义无效')


def get_polygon_area(vertexs):
    # 基于向量叉积计算凸多边形面积，凸多边形的点要求按照一定次序存储
    area = 0
    vertexsNum = len(vertexs)
    assert vertexsNum >= 3, "少于3个点是构建不了凸多边形的"

    for i in range(vertexsNum):
        j = (i+1)%vertexsNum
        area += (
            vertexs[i][0]*vertexs[j][1]-vertexs[i][1]*vertexs[j][0]
        )

    area /= 2.

    return abs(area)


def get_polygon_len(vertexs_):
    # 计算多边形的周长
    vertexsNum_ = len(vertexs_)
    assert vertexsNum_ >= 3, "少于3个点是构建不了凸多边形的"

    length_ = get_traj_len(vertexs_, True)
    return length_


def get_polygon_gravity(vertexs):
    # 求解多边形的重心
    gx, gy, area = 0., 0., 0.
    versNum = len(vertexs)
    px, py = vertexs[0][0], vertexs[0][1]
    i = 1
    while i <= versNum:
        idx = i if i != versNum else 0
        sx, sy = vertexs[idx][0], vertexs[idx][1]
        tp = px * sy - sx * py
        area += tp / 2
        gx += (px+sx) * tp
        gy += (py+sy) * tp
        px, py = sx, sy
        i += 1

    gx /= 6
    gy /= 6
    sx, sy = gx/area, gy/area
    return [sx, sy]


def polygon_norm(vertexs):
    verGrav = get_polygon_gravity(vertexs)
    versNorm = []
    for ver in vertexs:
        versNorm.append([ver[0]-verGrav[0], ver[1]-verGrav[1]])

    return versNorm, verGrav


def create_rotate_polygon(vertexs, pt, theta):
    """
     无人机航线规划 ，基于凸多边形地块往复式运动。
     凹多边形有时会出现不能完全覆盖的情况 。 一般处理方法就是将一个凹多边形切割成多个凸多边形 。
     vertexs 拐角点列表
     pt      中心点
     theta   角度

    让多边形绕着中心点（外接矩形的中心点）旋转想要的角度，将得到的新多边形再与纬度线做相交操作，获取到那些交点之后，再将那些交点旋转回来。
    换句话说，变换前它是一个任意多边形，变换后，它还是一个任意多边形，都是满足上面已经预设好的场景的。
    这样的好处显而易见，你不需要修改上面的任何一个函数，也不需要去多写一条两个一次函数求交点的公式。
    把问题化到最简单的场景去，只需要添加变换的代码本文来源全球无人机网
    """
    res = []

    vertexsNum = len(vertexs)
    for i in range(vertexsNum):
        tr = transform(vertexs[i], pt, theta)
        res.append(tr)

    return res


def expand_polygon_d(vertexs_, d, expand_point=False):
    """
    基于向量法，等间隔缩放多边形，适用于凸多边形和凹多边形，但是这个代码感觉有点不精确，此外当任意一个边的长度小于缩放长度的时候，会产生交叉，后续需要判定交叉小边现象的出现，并优化
    vertexs: 多边形的顶点，按一定次序存放
    d: 缩放间隔
    expand_point: True-将多边形的点缩放d个间隔，False-将多边形的边缩放d个间隔
    """
    # 求两条邻边的向量
    vertexs = copy.deepcopy(vertexs_)
    vertexs.append(vertexs[0])
    vertexs.append(vertexs[1])
    vertexsE = []
    xNums = len(vertexs)
    for i in range(xNums):
        if i < (xNums - 2):
            # 求边的长度
            vec1 = [vertexs[i + 1][0] - vertexs[i][0], vertexs[i + 1][1] - vertexs[i][1]]
            vec2 = [vertexs[i + 1][0] - vertexs[i + 2][0], vertexs[i + 1][1] - vertexs[i + 2][1]]
            d1 = (vec1[0] ** 2 + vec1[1] ** 2) ** 0.5
            d2 = (vec2[0] ** 2 + vec2[1] ** 2) ** 0.5
            # 两条边的夹角
            # 内积ab,sinA
            ab = vec1[0] * vec2[0] + vec1[1] * vec2[1]
            cosA = ab / (d1 * d2)

            # x[]
            # 判断凹凸点（叉积）
            P1P3_x = vertexs[i + 2][0] - vertexs[i][0]
            P1P3_y = vertexs[i + 2][1] - vertexs[i][1]
            P1P2_x = vertexs[i + 1][0] - vertexs[i][0]
            P1P2_y = vertexs[i + 1][1] - vertexs[i][1]
            # P=P1P3 x P1P2
            P = (P1P3_y * P1P2_x) - (P1P3_x * P1P2_y)
            # 为凹时
            fac = -1 if P > 0 else 1
            sinA = ((1 - cosA ** 2) ** 0.5)
            # 向量V1,V2的坐标
            if expand_point:
                sinA_half = math.sqrt((1 - cosA) / 2.)
                d = d * sinA_half
            dv = fac * d / sinA  # V1,V2长度相等
            dv1 = dv / d1
            dv2 = dv / d2
            v1_x = dv1 * vec1[0]
            v1_y = dv1 * vec1[1]
            v2_x = dv2 * vec2[0]
            v2_y = dv2 * vec2[1]
            Qi_x = vertexs[i + 1][0] + v1_x + v2_x
            Qi_y = vertexs[i + 1][1] + v1_y + v2_y

            vertexsE.append([Qi_x, Qi_y])

    vertexsE = [vertexsE[-1]] + vertexsE[:-1]
    return vertexsE


# ===========================================================轨迹相关==========================================================
def get_traj_len(vertexs_, isClosed_=False):
    # 计算多边形的周长
    length_ = 0
    vertexsNum_ = len(vertexs_)
    assert vertexsNum_ >= 2, "少于2个点是构建不了路径"

    traverseLen_ = vertexsNum_ if isClosed_ else vertexsNum_-1
    for i_ in range(traverseLen_):
        j_ = (i_ + 1) % vertexsNum_
        length_ += distance_p1p2(vertexs_[i_], vertexs_[j_])
    return length_


def calc_line_num_in_polygon(outRect, step):
    nums = math.floor(distance_p1p2(outRect[0], outRect[3]) // step) + 1
    return nums


def get_polygonLine_area(polyline, sideStep):
    # 计算航线所覆盖的区域面积，此处假设航线上的每一帧图像画面刚好能覆盖旁向间隔
    area = 0
    polylineNums = len(polyline)
    for i in range(0, polylineNums, 2):
        area += distance_p1p2(polyline[i], polyline[i+1])

    return area * sideStep


def line_re_order(lines_):
    linesLen_ = len(lines_)
    assert linesLen_ % 2 == 0, "线段端点需要为偶数"
    linesCp_ = []
    for lidx_ in range(0, linesLen_, 2):
        linesCp_.append(lines_[lidx_+1])
        linesCp_.append(lines_[lidx_])
    return linesCp_


def fix_lines(rVertexs, rVertexsE, routRect, rOutRectE, lines, d, sStep):
    # 判断顶部和底部超过拓张前，旋转后的凸多边形边界的线段数目
    topAboveNums = 0
    botBeyondNums = 0
    vertexsNum = len(rVertexsE)
    linesLen = len(lines)

    routRectSmall = copy.deepcopy(routRect) if d < 0 else copy.deepcopy(rOutRectE)
    routRectBig = copy.deepcopy(rOutRectE) if d < 0 else copy.deepcopy(routRect)
    rVertexsBig = copy.deepcopy(rVertexsE) if d < 0 else copy.deepcopy(rVertexs)
    for lidx in range(linesLen):
        if lines[lidx][1] <= routRectSmall[0][1]:
            break
        topAboveNums += 1
    topAboveNums /= 2

    for lidx in range(linesLen-1, -1, -1):
        if lines[lidx][1] >= routRectSmall[2][1]:
            break
        botBeyondNums += 1
    botBeyondNums /= 2

    linesCp = []

    if topAboveNums > 1:
        lineCot = int((topAboveNums-1) * 2)
        linesCp = copy.deepcopy(lines[lineCot:])
        if (topAboveNums-1) % 2 == 1:
            linesCp = line_re_order(linesCp)
    elif topAboveNums == 1:
        linesCp = copy.deepcopy(lines)
    else:
        linesTmp = []
        if d != 0:
            # 遍历每一个多边形顶点
            newY = lines[0][1]
            YLim = 0.4*(routRectBig[0][1]-routRectSmall[0][1]) + routRectSmall[0][1]
            while newY <= YLim:
                newY += 0.1 * sStep

            if (newY >= routRectBig[0][1]) or (newY <= routRectSmall[0][1]):
                newY = YLim

            for j in range(vertexsNum):
                point = calc_point_inline_with_y(
                    rVertexsBig[j],
                    rVertexsBig[(j + 1) % vertexsNum],
                    newY
                )

                if point:
                    linesTmp.append(point)
        else:
            # 若无外扩行为，则不用进行过多的考虑，轨迹复制，但需要对y坐标进行调整
            linesTmp = copy.deepcopy(lines[:2])
            if linesTmp[0][1] < routRectSmall[0][1]:
                linesTmp[0][1] = routRectSmall[0][1]
                linesTmp[1][1] = routRectSmall[0][1]
            else:
                linesTmp = []
            # else:  # linesTmp[0][1] == routRectSmall[0][1]
            #     linesTmp[0][1] = routRectSmall[0][1] + abs(d / 2.)
            #     linesTmp[1][1] = routRectSmall[0][1] + abs(d / 2.)

        if linesTmp:
            # 保证有两个不重合的交点
            assert len(linesTmp) == 2 and linesTmp[0][0] != linesTmp[1][0], "{}, {}, {}".format(len(linesTmp), linesTmp[0][0], linesTmp[1][0])
            linesCp = [[max(linesTmp[0][0], linesTmp[1][0]), linesTmp[0][1]],
                       [min(linesTmp[0][0], linesTmp[1][0]), linesTmp[0][1]]] + lines
            linesCp = line_re_order(linesCp)
        else:
            linesCp = copy.deepcopy(lines)

    if botBeyondNums > 1:
        lineCot = int((botBeyondNums-1) * 2)
        for _ in range(lineCot):
            linesCp.pop()
    elif botBeyondNums < 1:
        linesTmp = []
        if d != 0:
            # 遍历每一个多边形顶点
            newY = linesCp[-1][1]
            YLim = 0.4 * (routRectBig[2][1] - routRectSmall[2][1]) + routRectSmall[2][1]
            while newY >= YLim:
                newY -= 0.1*sStep

            if (newY <= routRectBig[2][1]) or (newY >= routRectSmall[2][1]):
                newY = YLim

            for j in range(vertexsNum):
                point = calc_point_inline_with_y(
                    rVertexsBig[j],
                    rVertexsBig[(j + 1) % vertexsNum],
                    newY
                )

                if point:
                    linesTmp.append(point)
        else:
            # 若无外扩行为，则不用进行过多的考虑，轨迹复制，但需要对y坐标进行调整
            linesTmp = copy.deepcopy(lines[-1:-3:-1])
            if linesTmp[0][1] > routRectSmall[2][1]:
                linesTmp[0][1] = routRectSmall[2][1]
                linesTmp[1][1] = routRectSmall[2][1]
            else:
                linesTmp = []
            # else:  # linesTmp[0][1] == routRectSmall[0][1]
            #     linesTmp[0][1] = routRectSmall[2][1] - abs(d / 2.)
            #     linesTmp[1][1] = routRectSmall[2][1] - abs(d / 2.)

        if len(linesTmp) > 0:
            # 保证有两个不重合的交点
            assert len(linesTmp) == 2 and linesTmp[0][0] != linesTmp[1][0]
            linescpLen = len(linesCp) / 2
            if linescpLen % 2 == 0:
                linesCp.append([min(linesTmp[0][0], linesTmp[1][0]), linesTmp[0][1]])
                linesCp.append([max(linesTmp[0][0], linesTmp[1][0]), linesTmp[0][1]])
            else:
                linesCp.append([max(linesTmp[0][0], linesTmp[1][0]), linesTmp[0][1]])
                linesCp.append([min(linesTmp[0][0], linesTmp[1][0]), linesTmp[0][1]])

    if abs(linesCp[0][0] - linesCp[1][0]) < d:
        linesCp[0][0] -= d
        linesCp[1][0] += d

    if abs(linesCp[-1][0] - linesCp[-2][0]) < d:
        linescpLen = len(linesCp) / 2
        if linescpLen % 2 == 0:
            linesCp[-1][0] -= d
            linesCp[-2][0] += d
        else:
            linesCp[-1][0] += d
            linesCp[-2][0] -= d

    return linesCp


def dji_poly_traj_v1(vertexs, sideStep, theta=0, d=0, isFixLine=True):
    """
    vertexs: 凸多边形的顶点
    sideOverlap: 旁向重叠率
    theta: 逆时针旋转角度
    d: 缩放间隔，<0，外扩，>0，内缩
    """
    if d != 0:
        vertexsE = expand_polygon_d(vertexs, d, expand_point=False)
    else:
        vertexsE = copy.deepcopy(vertexs)

    vertexsNum = len(vertexsE)
    outRect, pt = create_poly_bounds(vertexsE)

    # 创建变换后的多边形
    rVertexsE = create_rotate_polygon(vertexsE, pt, theta)
    rOutRectE, rpt = create_poly_bounds(rVertexsE)

    nums = calc_line_num_in_polygon(rOutRectE, sideStep)
    lines = []

    # 遍历每一条纬度线
    linecot = 0
    for i in range(nums):
        linesTmp = []
        # 遍历每一个多边形顶点
        for j in range(vertexsNum):
            point = calc_point_inline_with_y(
                rVertexsE[j],
                rVertexsE[(j+1) % vertexsNum],
                rOutRectE[0][1] - i*sideStep
            )

            if point:
                linesTmp.append(point)

        # 去掉交点个数不为2的纬度线
        if len(linesTmp) != 2:
            continue

        # 去掉两个交点重合的维度线
        if linesTmp[0][0] == linesTmp[1][0]:
            continue

        if linecot % 2 == 0:
            lines.append([min(linesTmp[0][0], linesTmp[1][0]), linesTmp[0][1]])
            lines.append([max(linesTmp[0][0], linesTmp[1][0]), linesTmp[0][1]])
        else:
            lines.append([max(linesTmp[0][0], linesTmp[1][0]), linesTmp[0][1]])
            lines.append([min(linesTmp[0][0], linesTmp[1][0]), linesTmp[0][1]])
        linecot += 1

    # 对生成的直线进行后处理
    rvertexs = create_rotate_polygon(vertexs, pt, theta)
    routRect, _ = create_poly_bounds(rvertexs)
    if isFixLine:
        linesCp = fix_lines(rvertexs, rVertexsE, routRect, rOutRectE, lines, d, sideStep)
    else:
        linesCp = lines

    # 最后就可以直接转换来进行绘制
    rLines = create_rotate_polygon(linesCp, pt, -theta)

    return rLines, vertexsE


def cross_lines(trajLine0_, trajLine1_):
    # 用于产生井字路线，确保trajLine0_与trajLine1_为两条相互正交的航线
    assert (len(trajLine0_) >= 4 and len(trajLine1_) >= 4)

    ps0 = [trajLine0_[0], trajLine0_[1], trajLine0_[-2], trajLine0_[-1]]
    ps1 = [trajLine1_[0], trajLine1_[1], trajLine1_[-2], trajLine1_[-1]]

    assert (((ps0[0][0] != ps0[1][0]) or (ps0[0][1] != ps0[1][1])) and
            ((ps0[0][0] != ps0[2][0]) or (ps0[0][1] != ps0[2][1])) and
            ((ps0[0][0] != ps0[3][0]) or (ps0[0][1] != ps0[3][1])) and
            ((ps1[0][0] != ps1[1][0]) or (ps1[0][1] != ps1[1][1])) and
            ((ps1[0][0] != ps1[2][0]) or (ps1[0][1] != ps1[2][1])) and
            ((ps1[0][0] != ps1[3][0]) or (ps1[0][1] != ps1[3][1]))
            )

    eIdx0_, sIdx1_ = 3, 0
    minDis_ = distance_p1p2(ps0[3], ps1[0])
    for idx0_ in range(4):
        for idx1_ in range(4):
            if idx0_ != 3 and idx1_ != 0:
                disTmp_ = distance_p1p2(ps0[idx0_], ps1[idx1_])
                if disTmp_ < minDis_:
                    eIdx0_ = idx0_
                    sIdx1_ = idx1_
                    minDis_ = disTmp_

    crossIdx_ = len(trajLine0_)-1
    trajLine0R_ = copy.deepcopy(trajLine0_)
    trajLine1R_ = copy.deepcopy(trajLine1_)

    if eIdx0_ != 0 and eIdx0_ != 3:
        trajLine0R_ = line_re_order(trajLine0R_)
    if eIdx0_ <= 1:
        trajLine0R_ = trajLine0R_[::-1]

    if sIdx1_ != 0 and sIdx1_ != 3:
        trajLine1R_ = line_re_order(trajLine1R_)
    if sIdx1_ >= 2:
        trajLine1R_ = trajLine1R_[::-1]

    trajLineCross_ = trajLine0R_ + trajLine1R_

    return trajLineCross_, crossIdx_


def well_poly_traj_v1(vertexs_, sideStep_, theta_=0, d_=0, isFixLine=True):
    trajLines0_, vertexsE_ = dji_poly_traj_v1(vertexs_, sideStep_, theta_, d_, isFixLine)
    trajLines1_, _ = dji_poly_traj_v1(vertexs_, sideStep_, theta_+90, d_, isFixLine)
    trajLines_, _ = cross_lines(trajLines0_, trajLines1_)

    return trajLines_, vertexsE_


if __name__ == '__main__':
    # data = tu_polygon_gen(5, 10)
    data = [[30.0, 0.0], [-21.86905882264234, 20.536413177860666], [-29.057494833858932, 7.460696614945645], [-29.763441039434337, -3.7599970069291286], [26.289200401315906, -14.45261022305146]]
    sideStep, headStep = 1.384615384615385, 1.3461538461538458
    traj_line, dataE = dji_poly_traj_v1(data, sideStep, 0, -headStep)
    traj_line1, _ = dji_poly_traj_v1(data, sideStep, 90, -headStep)

    # print(len(traj_line))
    traj_line_sum = traj_line + traj_line1
    print(distance_p1p2(traj_line[-1], traj_line1[0]))
    btls.polygon_draw([data, dataE], traj_line_sum)

    traj_line_sum_, crossIdx = cross_lines(traj_line, traj_line1)
    print(distance_p1p2(traj_line_sum_[crossIdx], traj_line_sum_[crossIdx+1]))
    btls.polygon_draw([data, dataE], traj_line_sum_)


