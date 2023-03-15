from tqdm import tqdm
import torch
import torch.nn as nn
from utils.draw import drawCharts


def train(net: nn.Module, dataLoader, optimizer, lossFunc, logger, device='cpu'):
    ''' 
    net: 模型
    dataLoader: 数据读取器
    optimizer: 优化器
    lossFunc: 损失函数
    device: cpu or gpu
    '''
    logger.info('Start trainning......')
    EPOCHS = 10
    # 存储训练过程
    history = {'Test Loss': [], 'Test Accuracy': []}
    for epoch in range(1, EPOCHS + 1):
        # 添加一个进度条，增加可视化效果
        processBar = tqdm(dataLoader, unit='step')
        net.train(True)
        loss = 0
        accuracy = 0
        for step, (trainImgs, labels) in enumerate(processBar):

            trainImgs = trainImgs.to(device)
            labels = labels.to(device)

            net.zero_grad()
            outputs = net(trainImgs)
            loss = lossFunc(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = torch.sum(predictions == labels)/labels.shape[0]
            loss.backward()

            optimizer.step()
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                    (epoch, EPOCHS, loss.item(), accuracy.item()))
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , EPOCHS, loss, accuracy))
        processBar.close()
    
    logger.info('Finish training!')
    drawCharts(history)
    

