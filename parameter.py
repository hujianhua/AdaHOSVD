path = "../datasets/airplane_0001.txt"  # 点云文件所在位置
noisepath = "../noisedatasets/airplane005.txt"
gridSize = 0.04
leafSize = 32
k = 63
simiNum = 15
downArea = 0.5  # 这是搜索近邻点时的半径,原为100
simiThreshold = 0.2  # 寻找相似片元的自适应阈值
translate = 0.2
coreThre = 0.05   # 核张量剔除元素比例

'''
数据集,    gridSize    k   simiNum downArea    simiThreshold   coreThre
gelou005:   0.6     50      20      5               200         150
bathtub01   0.07    20      10      0.5             0.3         0.8
bottle01    0.07    20      10      0.5             0.3         0.8
bowl\car    0.07    20      10      0.5             0.5         0.8
'''

# path = "../datasets/gelou.txt"  # 点云文件所在位置
# noisepath = "../noisedatasets/gelou05.txt"
# gridSize = 0.04
# leafSize = 32
# k = 63
# simiNum = 15
# downArea = 0.5  # 这是搜索近邻点时的半径,原为100
# simiThreshold = 0.2  # 寻找相似片元的自适应阈值
# translate = 0.2
# coreThre = 0.05   # 核张量剔除元素比例