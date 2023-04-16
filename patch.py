from scipy.spatial import KDTree
from parameter import leafSize, downArea
import numpy as np


def makePatches(DPC, PC, k):
    root = KDTree(PC, leafsize=leafSize)
    pointIndexs = []
    patches = []
    for point in DPC:
        dist, indexs = KDTree.query(root, point, k)
        pointIndexs.append(indexs)

        # 构建片元
        patch = PC[indexs, :]
        patch = patch - point
        patches.append(patch)
    patches = np.array(patches)
    pointIndexs = np.array(pointIndexs)
    return patches, pointIndexs


def getNeighborPatch(DPC):
    root = KDTree(DPC, leafsize=leafSize)
    neigIndexs = []
    for i in range(DPC.shape[0]):
        index = root.query_ball_point(DPC[i], r=downArea)
        neigIndexs.append(index)    # 考虑是否要与自身进行计算
    return neigIndexs


def deCenter(DPC, Patches):
    for i in range(DPC.shape[0]):
        Patches[i] = Patches[i] + DPC[i]


def getDePoints(Patches, maskPatch, pointIndexs, pointNum, k):
    dePC = np.zeros((pointNum, 3))
    Count = np.zeros(pointNum)
    for i in range(Patches.shape[0]):
        if maskPatch[i]:
            for j in range(k):
                dePC[pointIndexs[i][j]] += Patches[i][j]
                Count[pointIndexs[i][j]] += 1
    denoisePC = []
    for i in range(pointNum):
        if Count[i] != 0:  # 去掉没有用到的点
            denoisePC.append(dePC[i] / Count[i])
    denoisePC = np.array(denoisePC).reshape((-1, 3))
    # print("点云点数为", denoisePC.shape)
    return denoisePC
