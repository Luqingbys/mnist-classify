import matplotlib.pyplot as plt


def drawCharts(history: dict):
    #对测试Loss进行可视化
    plt.plot(history['Train Loss'],label = 'Train Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    #对测试准确率进行可视化
    plt.plot(history['Train Accuracy'],color = 'red',label = 'Train Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


def drawMPR(precision_dict: dict, recall_dict: dict, save_path):
    plt.figure()
    plt.step(recall_dict['macro'], precision_dict['macro'], where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Average precision score, micro-averaged over all classes')
    plt.savefig(f'{save_path}Matro PR.png')
    plt.show()