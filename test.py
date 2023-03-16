import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from utils.draw import drawMPR


def test(net, dataLoader, loss_func, batch_size, output, device='cpu'):
    net = torch.load(output+'model.pkl')
    all_labels = []
    all_predicts = []
    eye = np.eye(10)

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
        # print(testImgs.shape)
        # print(labels)
        outputs: torch.Tensor = net(testImgs)
        # print(outputs)
        loss = loss_func(outputs, labels)

        all_predicts.append(outputs.to('cpu').detach().numpy()) # 直接添加当前批量的预测结果，outputs: (batch_size, 10)
        all_labels.append(eye[labels.to('cpu')]) # 通过单位阵eye得到labels的全部独热编码

        # predictions: (256, )，刚好是当前批量每一个样本的预测结果
        predictions = torch.argmax(outputs, dim=1)
        # print(predictions)
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
    # print('predict correctly: ', correct)
    # print('total: ', batch_size*len(dataLoader))
    print('average test accuracy: ', testAccuracy.item())

    # 计算总测试的平均Loss
    testLoss = totalLoss/len(dataLoader)
    print('total test loss: ', testLoss.item())
    # 将本step结果进行可视化处理
    # processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
    #                         (epoch, EPOCHS, loss.item(), accuracy.item(), testLoss.item(), testAccuracy.item()))

    # 计算完整测试集上的精度和召回
    all_predicts: np.ndarray = np.concatenate(all_predicts, axis=0)
    all_labels: np.ndarray = np.concatenate(all_labels, axis=0) # (10000, 10)
    precision_dict = dict()
    recall_dict = dict()
    average_precision_dict = dict()
    for i in range(10):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(all_labels[:, i], all_predicts[:, i])
        average_precision_dict[i] = average_precision_score(all_labels[:, i], all_predicts[:, i])
    precision_dict['macro'], recall_dict['macro'], _ = precision_recall_curve(all_labels.ravel(), all_predicts.ravel())
    average_precision_dict['macro'] = average_precision_score(all_labels, all_predicts, average='micro')

    drawMPR(precision_dict, recall_dict, save_path=output)