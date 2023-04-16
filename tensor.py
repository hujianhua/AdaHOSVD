from denoise import computeSimi, HOSVD, comtranslate
from parameter import simiThreshold, simiNum, translate
import numpy as np


def maketensor(patches, neigIndexs, k):
    # 相似片元数目，相似度阈值
    patchIndexs = []
    RT = []
    tensors = []

    # 1.筛选
    oneindexs = []
    for i in range(patches.shape[0]):
        oneindex = []
        for j in range(len(neigIndexs[i])):
            dist1 = comtranslate(patches[i], patches[j])
            if dist1 < translate:
                oneindex.append(neigIndexs[i][j])
        oneindexs.append(oneindex)

    # 2.计算相似度
    for i in range(patches.shape[0]):
        tensor = []
        patchIndex = []
        tensor.append(patches[i])
        Error = []
        Rt = []
        for j in range(len(oneindexs[i])):
            # 比较本片元与近邻片元
            dist, patch, rt = computeSimi(patches[i], patches[j])

            if dist < simiThreshold:
                patchIndex.append(oneindexs[i][j])
                tensor.append(patch)
                Rt.append(rt)
                Error.append(dist)
        # 对张量进行重新排列，依照dist的大小
        patchIndex = np.array(patchIndex)
        tensor = np.array(tensor).reshape((-1, k, 3))
        Rt = np.array(Rt)

        sorted = np.argsort(Error)
        # print("相似度:", max(Error), min(Error))
        if len(sorted) > simiNum:
            sorted = sorted[:simiNum]
        patchIndex = patchIndex[sorted]
        tensor = tensor[sorted]
        Rt = Rt[sorted]

        tensors.append(tensor)
        RT.append(Rt)
        patchIndexs.append(patchIndex)
        # print("张量构建     " + str((i+1)) + " | " + str(patches.shape[0]) + "维度:" + str(len(tensor)))
    return tensors, RT, patchIndexs


def getDePatches(tensors, patchIndexs, RTs, k):
    CountPatch = np.zeros(len(patchIndexs))  # 统计每个片元在几个张量中
    dePatch = np.zeros((len(patchIndexs), k, 3))  # 将计算得到的均值片元装入去噪片元集dePC
    for i in range(len(patchIndexs)):
        for j in range(len(patchIndexs[i])):
            tensor = tensors[i][j]
            # 反向旋转平移
            # 齐次化
            ones = np.ones((k, 1))
            pat_R = np.hstack((tensor, ones))
            # 反旋转平移矩阵
            ones_R = np.zeros((1, 4))
            ones_R[0, 3] = 1
            in_t = (-1 * np.dot(RTs[i][j][:, :3], RTs[i][j][:, 3])).reshape((3, 1))
            in_Rt = np.hstack((RTs[i][j][:, :3].T, in_t))
            pha_Rt = np.vstack((in_Rt, ones_R))

            tmp = np.dot(pha_Rt, pat_R.T)
            dePatch[patchIndexs[i][j]] += tmp.T[:, :3]
            CountPatch[patchIndexs[i][j]] += 1
    mask = CountPatch > 0
    for i in range(len(patchIndexs)):
        if mask[i] is True:
            dePatch[i] = dePatch[i] / CountPatch[i, 0]
    return dePatch, mask
