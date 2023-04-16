from parameter import gridSize
import numpy as np
import math


def downSample(PC):
    xMax = np.max(PC[:, 0])
    xMin = np.min(PC[:, 0])
    yMax = np.max(PC[:, 1])
    yMin = np.min(PC[:, 1])
    zMax = np.max(PC[:, 2])
    zMin = np.min(PC[:, 2])

    xNum = math.ceil((xMax - xMin) / gridSize)
    yNum = math.ceil((yMax - yMin) / gridSize)
    zNum = math.ceil((zMax - zMin) / gridSize)
    # 计算出每个点的网格索引，然后加入索引数组，将索引数组使用argsort进行排序，返回一个索引，然后使用这个索引来得到每个网格中的点，然后计算每个网格中的点的均值点，即为下采样点
    # 将此下采样点组成的数组返回，以供下一步构建片元
    gridIndex = []
    for i in range(PC.shape[0]):
        xID = math.floor((PC[i, 0] - xMin) / gridSize)
        yID = math.floor((PC[i, 1] - yMin) / gridSize)
        zID = math.floor((PC[i, 2] - zMin) / gridSize)
        ID = xID + yID * xNum + zID * yNum * xNum
        gridIndex.append(ID)
    gridIndex = np.array(gridIndex)

    # 对索引进行排序
    sort = np.argsort(gridIndex)    # sort类型: ndarray
    arrayIndex = gridIndex[sort]
    arrayPoints = PC[sort, :]
    # 计算采样点
    points = []
    xsum = 0
    ysum = 0
    zsum = 0
    numpoint = 0
    for i in range(len(sort) + 1):  # +1 为计算最后一个网格提供一次机会
        if i < len(sort):
            if arrayIndex[i] == arrayIndex[i - 1] or i == 0:
                xsum += arrayPoints[i, 0]
                ysum += arrayPoints[i, 1]
                zsum += arrayPoints[i, 2]
                numpoint += 1
            else:
                xmean = xsum / numpoint
                ymean = ysum / numpoint
                zmean = zsum / numpoint
                point = [xmean, ymean, zmean]
                points.append(point)
                xsum = arrayPoints[i, 0]
                ysum = arrayPoints[i, 1]
                zsum = arrayPoints[i, 2]
                numpoint = 1
        else:
            xmean = xsum / numpoint
            ymean = ysum / numpoint
            zmean = zsum / numpoint
            point = [xmean, ymean, zmean]
            points.append(point)
    points = np.array(points, dtype=float).reshape((-1, 3))
    return points