import torch


def test(net, dataLoader, loss_func, batch_size, device='cpu'):
    # 构造临时变量
    correct, totalLoss = 0, 0
    # 关闭模型的训练状态
    net.train(False)
    # 对测试集的DataLoader进行迭代
    for testImgs, labels in dataLoader:
        # print('测试数据集: ', testImgs.shape)
        testImgs = testImgs.to(device)
        labels = labels.to(device)
        outputs = net(testImgs)
        loss = loss_func(outputs, labels)

        # predictions: (256, )，刚好是当前批量每一个样本的预测结果
        predictions = torch.argmax(outputs, dim=1)
        print(
            f'current batch testing result, loss:{loss}, prediction {predictions.shape} ')
            
        # 存储测试结果
        totalLoss += loss
        correct += torch.sum(predictions == labels)

    # 计算总测试的平均准确率
    testAccuracy = correct/(batch_size * len(dataLoader))
    print('average test accuracy: ', testAccuracy.item())
    # 计算总测试的平均Loss
    testLoss = totalLoss/len(dataLoader)
    print('total test loss: ', testLoss.item())
    # 将本step结果进行可视化处理
    # processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
    #                         (epoch, EPOCHS, loss.item(), accuracy.item(), testLoss.item(), testAccuracy.item()))