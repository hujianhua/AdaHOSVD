"""
1.读取文件
2.下采样
3.构建片元，并中心化
4.寻找近邻片元，构建张量
5.张量高阶奇异值分解，计算核张量
6.核张量阈值剔除
7.重新计算张量
8.将张量中的片元放回原位
9.将放回原位的片元反向中心化
"""
from readfile import readFile
from parameter import path, noisepath, k
from downsample import downSample
from patch import makePatches, getNeighborPatch, deCenter, getDePoints
from tensor import maketensor, getDePatches
from denoise import deTensors
# import numpy as np
import open3d as o3d
# from addnoise import addNoise
import time
from evaluation import MSE, SNR, MCD


# Test.py 生成了阁楼数据集
# 去除噪声以后，点云呈现不同尺度同时显示的问题
# 已修改：增加可视化，修改gridSize尺寸，显示最大最小xyz值
def main():
    # 读取点云
    oldPC = readFile(path, lable=0)     # shape: n * 3

    # 加载噪声点云
    PC = readFile(noisepath, lable=1)
    # print("添加噪声完成,点数:", PC.shape[0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(PC)
    o3d.visualization.draw_geometries([pcd])

    pointNum = PC.shape[0]      # 点云点数
    print("点云点数为：", pointNum)

    # 下采样，开始计时
    start = time.time()
    DPC = downSample(PC)    # shape: n * 3
    print("下采样完成,确定片元数: "+str(DPC.shape[0]))

    # 构建片元
    print("片元包含点数", k)
    patches, pointIndexs = makePatches(DPC, PC, k)
    print("片元构建完成")

    # 寻找近邻片元
    neigIndexs = getNeighborPatch(DPC)
    print("近邻片元寻找完成")

    # 计算与近邻片元之间的相似性，从而得到张量
    print("开始构建张量")
    tensors, RTs, patchIndexs = maketensor(patches, neigIndexs, k)
    print("张量构建完成")

    # 除噪,需要用到张量，阈值，返回张量
    deTensor = deTensors(tensors)
    print("除噪完成")

    # 放回原位，需要用到张量，patchIndexs，返回片元集,TODO还需要逆变换
    dePatches, maskPatch = getDePatches(deTensor, patchIndexs, RTs, k)
    print("片元已放回原位")

    # 反中心化，需要用到DPC，返回片元集,python函数是引用传递，并非值传递
    deCenter(DPC, dePatches)
    print("片元反中心化完成")

    # 放回原位，需要用到pointIndexs，返回点集
    dePC = getDePoints(dePatches, maskPatch, pointIndexs, pointNum, k)
    print("点云放回原位完成: ", dePC.shape[0])

    # 返回算法使用时间
    end = time.time()
    usingtime = end - start
    print("使用时间: ", usingtime)
    print("处理点数: ", dePC.shape)

    # 算法评估
    cd, mse, ptd= MSE(oldPC, dePC)
    snr = SNR(dePC, mse)
    print("CD: ", cd)
    print("MSE：", mse)
    print("PTD：", ptd)
    print("SNR: ", snr)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dePC)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
