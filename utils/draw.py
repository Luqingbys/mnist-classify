import matplotlib.pyplot as plt


def drawCharts(history: dict):
    #对测试Loss进行可视化
    plt.plot(history['Test Loss'],label = 'Test Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    #对测试准确率进行可视化
    plt.plot(history['Test Accuracy'],color = 'red',label = 'Test Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()