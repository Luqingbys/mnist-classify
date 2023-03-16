'''从MNIST数据集按一定比例随机抽样，保证各个类别样本数量相等，读取成一份.csv文件'''
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import os
import argparse


parser = argparse.ArgumentParser('Sample by certain ratio from initial MNIST dataset.')
parser.add_argument('--path', type=str, default='data/MNIST/png/', help='Select a path which saves the whole png files.')
parser.add_argument('--ratio', type=float, default=0.7, help='Input the ratio the sample from MNIST.')
args = parser.parse_args()


def get_each_nums(path):
    '''返回每一个类别的样本总数'''
    res = dict()
    # path = 'data/MNIST/png/'
    for i in range(10):
        res[str(i)] = len(os.listdir(path+str(i)))
    return res


def sample_from_mnist(ratio, path):
    '''
    按比例抽样成训练集，注意原数据集训练集一共60000份样本，每一个类别的抽取样本数量应为60000×0.1×ratio
    ratio: 比例
    '''
    N = 60000
    each_num = N * 0.1 * ratio
    print(each_num)
    num_dict = get_each_nums(path)
    sample = []
    # sample.append(np.array(['path', 'label']))
    for i in tqdm(range(10)):
        print("Sampling randomly for index {}......".format(i))
        random_list = random.sample(range(num_dict[str(i)]), int(each_num)) # 注意这里要做强制类型转换，不然报错
        for n in random_list:
            sample.append(np.array([str(i) + '/mnist_' + str(n) + '-' + str(i) + '.png', str(i)]))
    sample = np.array(sample)
    np.random.shuffle(sample)
    np.savetxt(f'train_{str(ratio)}.csv', sample, delimiter=',', fmt='%s') # 保存str类型ndarray必须加上fmt=%s
    print('Success to sample MNIST PNG image files!')


if __name__ == '__main__':
    # path = 'data/MNIST/png/'
    sample_from_mnist(ratio=args.ratio, path=args.path)
    # print(get_each_nums(path))