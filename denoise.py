import numpy as np
from parameter import coreThre


# 底层功能
def ten2mat(in_C, mode):
    return np.reshape(np.moveaxis(in_C, mode, 0), (in_C.shape[mode], -1), order='F')


def mat2ten(in_mat, in_ten_shape, mode):
    return np.moveaxis(np.reshape(in_mat, in_ten_shape, order='F'), 0, mode)


def comtranslate(patchA, patchB):
    tranvalue = np.mean((patchA - patchB), 0)
    tranerror = np.mean(np.linalg.norm(patchB + tranvalue - patchA, axis=1))
    return tranerror


# 所需小功能
def computeSimi(patchA, patchB):
    vecA = patchA.T

    W = np.eye(vecA.shape[1])
    in_S1 = np.einsum('ij,jk->ik', vecA, W)
    in_S2 = np.einsum('ij,jk->ik', in_S1, patchB)
    in_U, in_sigma, in_vt = np.linalg.svd(in_S2)

    # 计算旋转矩阵
    middle = np.eye(in_U.shape[1])
    middle[-1, -1] = np.linalg.det(np.einsum('ij,jk->ik', in_vt.T, in_U.T))
    in_R2 = np.einsum('ij,jk->ik', in_vt.T, middle)
    in_R = np.einsum('ij,jk->ik', in_R2, in_U.T)

    # 计算平移向量
    q_mean = np.mean(patchB, 0).reshape(1, 3)
    p_mean = np.mean(vecA, 1).reshape(3, 1)
    t = q_mean.T - np.einsum('ij, jk->ik', in_R, p_mean)
    in_Rt = np.hstack((in_R, t))

    # 为片元施加旋转不变性
    in_ones = np.ones((patchB.shape[0], 1))
    vecB = np.hstack((patchB, in_ones)).T
    new_vecB = np.einsum('ij,jk->ik', in_Rt, vecB).T

    # 求相似片元间的误差
    in_dist_cor = patchA - new_vecB
    in_dist = np.linalg.norm(np.linalg.norm(in_dist_cor, axis=1), ord=1) / patchB.shape[0]
    return in_dist, new_vecB, in_Rt


def HOSVD(in_tensor):
    in_U = []  # 特征矩阵
    in_S = []  # 特征值
    in_C = in_tensor  # 核张量
    X = in_tensor
    for dim in range(len(in_tensor.shape)):
        U_mode, S_mode, Vt_mode = np.linalg.svd(ten2mat(in_tensor, dim))
        in_U.append(U_mode)
        in_S.append(S_mode)
        in_C = ten2mat(in_C, dim)
        in_C = np.dot(U_mode.T, in_C)

        # 反向n模态
        shape_ten = list(np.moveaxis(X, dim, 0).shape)
        shape_ten[0] = -1
        shape_ten = tuple(shape_ten)
        in_C = mat2ten(in_C, shape_ten, dim)
        X = in_C
    return in_U, in_S, in_C


def re_HOSVD(in_C, in_U):
    in_ten = in_C
    X = in_C
    for dim in range(len(in_U)):
        in_ten = ten2mat(in_ten, dim)
        in_ten = np.dot(in_U[dim], in_ten)

        # 反n模态
        shape_ten = list(np.moveaxis(X, dim, 0).shape)
        shape_ten[0] = -1
        in_ten = mat2ten(in_ten, shape_ten, dim)
        X = in_ten
    return in_ten


def deTensors(tensors):
    denoiseTensors = []
    for tensor in tensors:
        # 奇异向量，奇异值，核张量
        sinMat, sinValue, corTen = HOSVD(tensor)
        CThreshold = np.zeros(corTen.shape)
        for i in range(corTen.shape[0]):
            for j in range(corTen.shape[1]):
                for k in range(corTen.shape[2]):
                    # CThreshold[i, j, k] = coreThre * np.sqrt(2 * np.log(corTen.shape[0] * corTen.shape[1] * corTen.shape[2])) * np.log((i+1) ** 2 + (j+1) ** 2 + (k+1) ** 2)
                    CThreshold[i, j, k] = coreThre * np.sqrt(2 * np.log(corTen.shape[0] * corTen.shape[1] * corTen.shape[2]) * (i ** 2 + j ** 2 + k ** 2) / (corTen.shape[0] ** 2 + corTen.shape[1] ** 2 + corTen.shape[2] ** 2))
        mask = np.abs(corTen) < CThreshold
        corTen[mask] = 0

        new_ten = re_HOSVD(corTen, sinMat)
        denoiseTensors.append(new_ten)
    return denoiseTensors
