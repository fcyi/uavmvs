import numpy as np
import random


# 点云随机滤波
def voxel_random_filter(cloud_, leafSize_):
    pointCloud_ = np.asarray(cloud_.points)  # N 3
    # 1、计算边界点
    xMin_, yMin_, zMin_ = np.amin(pointCloud_, axis=0)  # 按列寻找点云位置的最小值
    xMax_, yMax_, zMax_ = np.amax(pointCloud_, axis=0)
    # 2、计算每个维度上体素格网的个数
    Dx_ = (xMax_ - xMin_) // leafSize_ + 1
    Dy_ = (yMax_ - yMin_) // leafSize_ + 1
    Dz_ = (zMax_ - zMin_) // leafSize_ + 1
    print("Dx * Dy * Dz is {} * {} * {}".format(Dx_, Dy_, Dz_))
    # 3、计算每个点的格网idx
    h_ = list()
    for i_ in range(len(pointCloud_)):
        # 分别在x, y, z方向上格网的idx
        hx_ = (pointCloud_[i_][0] - xMin_) // leafSize_
        hy_ = (pointCloud_[i_][1] - yMin_) // leafSize_
        hz_ = (pointCloud_[i_][2] - zMin_) // leafSize_
        h_.append(hx_ + hy_ * Dx_ + hz_ * Dx_ * Dy_)   # 该点所在格网 映射到1D的idx
    h_ = np.array(h_)

    # 4、体素格网内随机筛选点
    hIndice_ = np.argsort(h_)
    hSorted_ = h_[hIndice_]
    #################################
    # h_         3 1 2 1     每个3D点的格网idx，其位置也为对应3D点在点云中的位置索引
    # hIndice_  1 3 2 0     升序排列时，每个3D点的格网idx在h中的索引
    # hSorted_  1 1 2 3     升序排列后的 每个3D点的格网idx
    #################################
    randomIdx_ = []
    begin_ = 0
    # 遍历每个3D点的格网idx
    for i_ in range(len(hSorted_) - 1):
        # 当前3D点和后一个3D点的格网idx 相同，则跳过
        if hSorted_[i_] == hSorted_[i_ + 1]:
            continue
        # 当前3D点和后一个3D点的格网idx 不相同，则在当前3D点所在格网内随机选择一个3D点
        else:
            # begin_: 在同一个格网内的第一个3D点的格网idx的 在hSorted_/hIndice_的位置，i：在同一个格网内的最后一个3D点的格网idx 在hSorted_/hIndice_的位置
            pointIdx_ = hIndice_[begin_: i_ + 1]  # 同一格网内 所有3D点的格网idx在h中的位置，也是这些3D点在点云中的索引
            randomIdx_.append(random.choice(pointIdx_))  # 在同一格网内 随机选择一个3D点
            begin_ = i_ + 1
    filteredPoints_ = (cloud_.select_by_index(randomIdx_))
    return filteredPoints_


# 点云根据纹理进行滤波（纹理通过法向量以及sobel响应幅值表示）
def voxel_texture_filter(cloud_, sobelScores_, leafSize_):
    pointCloud_ = np.asarray(cloud_.points)  # N 3
    # colors_ = np.asarray(cloud_.colors) / 255.0 # N 3
    normals_ = np.asarray(cloud_.normals)
    # 1、计算边界点
    xMin_, yMin_, zMin_ = np.amin(pointCloud_, axis=0)
    xMax_, yMax_, zMax_ = np.amax(pointCloud_, axis=0)
    # 2、计算每个维度上体素格网的个数
    Dx_ = (xMax_ - xMin_) // leafSize_ + 1
    Dy_ = (yMax_ - yMin_) // leafSize_ + 1
    Dz_ = (zMax_ - zMin_) // leafSize_ + 1
    print("Dx * Dy * Dz is {} * {} * {}".format(Dx_, Dy_, Dz_))
    # 3、计算每个3D点的格网idx
    h_ = list()
    for i_ in range(len(pointCloud_)):
        hx_ = (pointCloud_[i_][0] - xMin_) // leafSize_
        hy_ = (pointCloud_[i_][1] - yMin_) // leafSize_
        hz_ = (pointCloud_[i_][2] - zMin_) // leafSize_
        h_.append(hx_ + hy_ * Dx_ + hz_ * Dx_ * Dy_)
    h_ = np.array(h_)

    # 4、根据纹理度量保留3D点
    hIndice_ = np.argsort(h_)
    hSorted_ = h_[hIndice_]
    #################################
    # h_        9 1 7 1 1     每个3D点的格网idx，其位置也为对应3D点在点云中的位置索引
    # hIndice_  1 3 4 2 0     升序排列时，每个3D点的格网idx在h中的索引
    # hSorted_  1 1 1 7 9     升序排列后的 每个3D点的格网idx
    #################################
    begin_ = 0
    meanNormals_ = np.mean(np.std(normals_, axis=0))
    stdNormals_ = np.std(np.std(normals_, axis=0))   # np.std(normals_, axis=0) 1, 3 每个3D点的法线向量在xyz维度的标准差，该区域的法向量分布较为离散,即表面法线变化剧烈,说明纹理丰富
    meanSobelScores_ = np.mean(sobelScores_)
    stdSobelScores_ = np.std(sobelScores_)
    pointIdxScores_ = {}
    textureScores_ = []
    for i_ in range(len(hIndice_) - 1):
        # 当前3D点和后一个3D点的格网idx 相同，则跳过
        if hSorted_[i_] == hSorted_[i_ + 1]:
            continue
        # 当前3D点和后一个3D点的格网idx 不相同，则在当前3D点所在格网内
        else:
            # begin_: 在同一个格网内的第一个3D点的格网idx的 在hIndice_的位置，i：在同一个格网内的最后一个3D点的格网idx 在hIndice_的位置
            pointIdx_ = hIndice_[begin_: i_ + 1]  # 同一格网内 所有3D点的格网idx在h中的位置，也是这些3D点在点云中的位置

            # 计算每个格网纹理度量得分
            normalsInVoxel_ = normals_[pointIdx_]
            sobelScoresInVoxel_ = sobelScores_[pointIdx_]
            avgSobelScore_ = (np.mean(sobelScoresInVoxel_) - meanSobelScores_) / stdSobelScores_
            avgNormalScore_ = (np.mean(np.std(normalsInVoxel_, axis=0)) - meanNormals_) / stdNormals_
            textureScore_ = 0.5 * avgSobelScore_ + 0.5 * avgNormalScore_
            textureScores_.append(textureScore_)

            pointIdxScores_[h_[hIndice_[begin_]]] = [pointIdx_, textureScore_]
            begin_ = i_ + 1

    # 计算所有包含3D点云的格网的纹理度量得分
    meanTextureScore_ = np.mean(textureScores_)
    stdTextureScore_ = np.std(textureScores_)
    maxTextureScore_ = np.max(textureScores_)
    minTextureScore_ = np.min(textureScores_)

    oriPointIdx_ = []
    randomIdx_ = []
    randomIdxTexture_ = []
    # 根据每个格网与所有格网的纹理度量得分比值，确定该格网内保留的点云数量
    for pointIdx_, textureScore_ in pointIdxScores_.values():
        oriPointIdx_.append(pointIdx_)

        randomIdx_.append(random.choice(pointIdx_))

        textureWeight_ = (textureScore_ - minTextureScore_) / (maxTextureScore_ - minTextureScore_)
        numPoints_ = max(1, int(len(pointIdx_) * textureWeight_))
        # numPoints_ = max(1, np.clip(int(len(pointIdx_) * texture_weight), 0, len(pointIdx_)//2)) # 该clip方案实际并不会影响结果，因为计算的保留点数结果 < 总个数的一半
        randomIdxTexture_.extend(random.sample(list(pointIdx_), numPoints_))  # 在同一格网内 随机选取numPoints_个3D点

    print("oriPointIdx_: {}, randomIdx_: {}, randomIdxTexture_: {}".format(len(oriPointIdx_), len(randomIdx_), len(randomIdxTexture_)))

    return randomIdx_, randomIdxTexture_

    # filteredPoints_ = (cloud_.select_by_index(randomIdx_))
    # filteredPointsTexture_ = (cloud_.select_by_index(randomIdxTexture_))
    # return filteredPoints_, filteredPointsTexture_

