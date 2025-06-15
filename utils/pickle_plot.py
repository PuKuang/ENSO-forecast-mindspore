"""
@Project ：ENSO-forecast-mindspore
@File    ：pickle_plot.py
@Author  ：Huang Zihan
@Date    ：2025/6/14
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt


def trainPlot(filepath, save_path):
    File = open(filepath, "rb")
    Dic = pickle.load(File)
    print(Dic)
    fig = plt.figure(figsize=(10, 9))
    ax1 = fig.add_subplot(211)
    ax1.plot(np.arange(1, 24), Dic["ACCList"], "-o", label="CNN")
    ax1.hlines(0.5, 0.5, 23.5)
    ax1.set_xlim(0.5, 23.5)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Forecast lead (month)")
    ax1.set_ylabel("Correlation skill")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_title('Validation correlation')
    plt.legend()
    plt.xticks(np.arange(1, 24, 1))
    ax2 = fig.add_subplot(212)
    ax2.plot(Dic["lossList"], label="Train Loss")
    ax2.hlines(Dic["LossVal"], 0, len(Dic["lossList"]), label="Val Loss", colors="red")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Batch Number")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_title('Train and Val Loss')

    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == '__main__':
    trainPlot(r'D:\Desktop\huawei-ai\ENSO-forecast-mindspore\experiments\pre-0614-lr2-e12-bs500\logfile.pickle',
              r'D:\Desktop\huawei-ai\ENSO-forecast-mindspore\experiments\pre-0614-lr2-e12-bs500\newnew_fig.png')
