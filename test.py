import torch
from tqdm import tqdm


def test(net, dataLoader, loss_func, batch_size, output, device='cpu'):
    net = torch.load(output+'model.pkl')

    # 构造临时变量
    correct, totalLoss = 0, 0
    # 关闭模型的训练状态
    net.train(False)
    torch.no_grad()
    # 对测试集的DataLoader进行迭代
    processBar = tqdm(dataLoader, unit='step')
    for step, (testImgs, labels) in enumerate(processBar):
        # print('测试数据集: ', testImgs.shape)
        testImgs = testImgs.to(device)
        labels = labels.to(device)
        outputs = net(testImgs)
        loss = loss_func(outputs, labels)

        # predictions: (256, )，刚好是当前批量每一个样本的预测结果
        predictions = torch.argmax(outputs, dim=1)
        # print(
        #     f'current batch testing result, loss:{loss}, prediction {predictions.shape} ')
        
        # 存储测试结果
        totalLoss += loss
        cur_correct = torch.sum(predictions == labels)
        correct += cur_correct
        # 将本step结果进行可视化处理
        processBar.set_description("current batch testing result... Test Loss: %.4f, Test Acc: %.4f" % (loss.item(), cur_correct / batch_size))

    # 计算总测试的平均准确率
    testAccuracy = correct/(batch_size * len(dataLoader))
    print('average test accuracy: ', testAccuracy.item())
    # 计算总测试的平均Loss
    testLoss = totalLoss/len(dataLoader)
    print('total test loss: ', testLoss.item())
    # 将本step结果进行可视化处理
    # processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
    #                         (epoch, EPOCHS, loss.item(), accuracy.item(), testLoss.item(), testAccuracy.item()))