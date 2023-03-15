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