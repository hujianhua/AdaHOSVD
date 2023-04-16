import numpy as np


def MSE(PC, DNPC):
    # 计算PC到DNPC，点对点的平均距离
    PC2DNPC = []
    for i in range(PC.shape[0]):
        dist = np.min(np.linalg.norm(DNPC[:] - PC[i], ord=2, axis=1))
        PC2DNPC.append(dist)
    mean1 = np.mean(PC2DNPC)

    # 计算DNPC到PC，点对点的平均距离
    DNPC2PC = []
    for i in range(DNPC.shape[0]):
        dist = np.min(np.linalg.norm(PC[:] - DNPC[i], ord=2, axis=1))
        DNPC2PC.append(dist)
    mean2 = np.mean(DNPC2PC)

    # 两者距离的均值即为去噪结果的误差均值
    mean = (mean1 + mean2) / 2
    return mean, max(mean1, mean2), mean2


def SNR(PC, Mse):
    snr = 10 * np.log10((np.sum(np.linalg.norm(PC, ord=2, axis=1)) / PC.shape[0]) / Mse)
    return snr


def MCD(PC, dePC):
    # 计算PC到DNPC，点对点的平均距离
    PC2dePC = []
    for i in range(PC.shape[0]):
        dist = np.min(np.linalg.norm(dePC[:] - PC[i], ord=1, axis=1))
        PC2dePC.append(dist)
    mean1 = np.mean(PC2dePC)

    # 计算DNPC到PC，点对点的平均距离
    DNPC2PC = []
    for i in range(dePC.shape[0]):
        dist = np.min(np.linalg.norm(PC[:] - dePC[i], ord=1, axis=1))
        DNPC2PC.append(dist)
    mean2 = np.mean(DNPC2PC)

    # 两者距离的均值即为去噪结果的误差均值
    mcd = (mean1 + mean2) / 2
    return mcd
