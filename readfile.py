# 读取文件
import numpy as np
import scipy.io as scio
from plyfile import PlyData


def readTXT(path, lable=0):
    fp = open(path, "r")
    all_lines = fp.readlines()
    data = []
    if lable == 0:
        for line in all_lines:
            items = line.split(',')
            # 使用空格将字符串切开
            data.append([float(items[0]), float(items[1]), float(items[2])])
        data = np.array(data)
    elif lable == 1:
        for line in all_lines:
            items = line.split(' ')
            # 使用空格将字符串切开
            data.append([float(items[0]), float(items[1]), float(items[2])])
        data = np.array(data)
    return data


def readMAT(path):
    data = scio.loadmat(path)
    pointCloud = data['U'].T  # shape: n * 4
    return pointCloud


def readPLY(path):
    plydata = PlyData.read(path)
    xlist = plydata['vertex']['x']
    ylist = plydata['vertex']['y']
    zlist = plydata['vertex']['z']
    PC = np.zeros((len(xlist), 3))
    PC[:, 0] = np.array(xlist)
    PC[:, 1] = np.array(ylist)
    PC[:, 2] = np.array(zlist)
    return PC


def readFile(path, format = "txt", lable=0):
    # pointCloud.shape: n * 4
    if format == "txt":
        pointCloud = readTXT(path, lable)
        pointCloud = pointCloud[:, :3]
        return pointCloud
    elif format == "mat":
        pointCloud = readMAT(path)
        pointCloud = pointCloud[:, :3]
        return pointCloud
    elif format == "ply":
        pointCloud = readPLY(path)
        return pointCloud
    else:
        print("还不支持其他格式")