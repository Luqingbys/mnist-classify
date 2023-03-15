# 有关基于MNIST手写数字识别数据集图像分类问题的解决方案的探索

采取不同的解决方案，探索深度学习界的Hello World————手写数字识别的图像分类问题。目前已经尝试的解决方案包括：

- 全连接层
- 深度卷积网络（ResNet）

## MNIST数据集简介

MNIST数据集来自美国国家标准与技术研究所，采集了250个不同的人的手写数字，每一个样本都是0~9的手写数字，且均为$28\times 28$的灰度图。下载地址：http://yann.lecun.com/exdb/mnist/，图片是以字节的形式进行存储，它包含了四个部分:

- Training set images: train-images-idx3-ubyte.gz (9.9 MB, 解压后 47 MB, 包含 60,000 个样本)
- Training set labels: train-labels-idx1-ubyte.gz (29 KB, 解压后 60 KB, 包含 60,000 个标签)
- Test set images: t10k-images-idx3-ubyte.gz (1.6 MB, 解压后 7.8 MB, 包含 10,000 个样本)
- Test set labels: t10k-labels-idx1-ubyte.gz (5KB, 解压后 10 KB, 包含 10,000 个标签)

未完待续......
