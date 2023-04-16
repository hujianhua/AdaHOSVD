import numpy as np
from downsample import downSample
# 添加噪声
noiseLevel = 0.05


def computeDistance(A, B):
    '''
    计算两个点云中，最远的两个点的距离
    :param A: shape: m * 3
    :param B: shape: n * 3
    :return DistMat: shape: m * n
    '''
    DistMat = np.zeros((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        DistMat[i, :] = np.linalg.norm(B[:, :] - A[i, :], axis=1).reshape((1, B.shape[0]))
    maxDist = np.sqrt(np.max(DistMat))
    return maxDist


def addNoise(PC):
    """
    1.对点云进行下采样
    2.寻找下采样点云中两个最远距离的点，将他们之间的距离开根
    3.将这个根再乘以传入的标准差
    4.在0-1间随机生成一个正态分布的矩阵
    5.将此矩阵乘以这个3中的结果
    6.将5结果加到点云当中
    PC.shape: n * 3
    dpc.shape: n * 3
    noisePC.shape: n * 3
    """
    # dpc = downSample(PC)
    maxDist = computeDistance(PC, PC)
    sigma = maxDist * noiseLevel
    randMat = np.random.randn(PC.shape[0], PC.shape[1]) * sigma     # 生成一个在0-1间，服从正态分布的矩阵
    noisePC = PC + randMat
    return noisePC