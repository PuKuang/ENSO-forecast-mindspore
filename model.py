"""
@Project ：ENSO-forecast-mindspore
@File    ：model.py
@Author  ：Huang Zihan
@Date    ：2025/6/14
"""

import mindspore as ms
from mindspore import nn, Tensor
import numpy as np


class ConvNetwork(nn.Cell):
    """
    M_Num, N_Num: convolutional filters和FCN层的神经元数量
    """
    def __init__(self, M_Num, N_Num):
        super(ConvNetwork, self).__init__()
        self.M = M_Num
        self.N = N_Num
        self.conv = nn.SequentialCell(
            nn.Conv2d(6, M_Num, kernel_size=(4, 8), pad_mode='same', has_bias=True),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(M_Num, M_Num, kernel_size=(4, 2), pad_mode='same', has_bias=True),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(M_Num, M_Num, kernel_size=(4, 2), stride=(1, 1), pad_mode='same', has_bias=True),
            nn.Tanh()
        )
        self.FCN = nn.SequentialCell(
            nn.Dense(6 * 18 * M_Num, N_Num),
            nn.Dense(N_Num, 23)
        )

    def construct(self, InData, *args, **kwargs):
        x = self.conv(InData)
        x = x.reshape((-1, 6 * 18 * self.M))  # MindSpore使用reshape代替view
        x = self.FCN(x)
        return x

if __name__ == '__main__':
    # Test Demo
    net = ConvNetwork(M_Num=30, N_Num=30)
    # batch_size=2, channels=6, height=24, width=72
    input_data = Tensor(np.random.randn(400, 6, 24, 72).astype(np.float32))
    print('input shape:', input_data.shape)
    output = net(input_data)
    print("Output shape:", output.shape)