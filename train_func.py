"""
@Project ：ENSO-forecast-mindspore
@File    ：train_func.py
@Author  ：Huang Zihan
@Date    ：2025/6/14

MindSpore版本训练和验证函数
"""

import datetime
import os
import scipy.stats as sps
import pickle
import mindspore as ms
from mindspore import nn, ops
from tqdm import tqdm


def valFunc(network, val_loader, criterion=nn.MSELoss()):
    # 切换到评估模式
    network.set_train(False)
    for data in val_loader:
        inputs, outputs = data # 一个批次完成所有数据加载
        break
    pred = network(inputs)
    loss_val = criterion(pred, outputs).asnumpy()
    cal_pred = pred.T.asnumpy()
    cal_outputs = outputs.T.asnumpy()
    acc_list = []
    p_list = []

    for index_month in range(23):
        acc, p_value = sps.pearsonr(cal_pred[index_month], cal_outputs[index_month])
        acc_list.append(acc)
        p_list.append(p_value)

    return loss_val, acc_list, p_list


class CustomTrainOneStep(nn.Cell):
    def __init__(self, network, optimizer, criterion):
        super().__init__()
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.grad = ops.value_and_grad(self.forward, None, optimizer.parameters)

    def forward(self, inputs, labels):
        pred = self.network(inputs)
        return self.criterion(pred, labels)

    def construct(self, inputs, labels):
        (loss), grads = self.grad(inputs, labels)
        grads = ops.clip_by_global_norm(grads, clip_norm=1.0)
        self.optimizer(grads)
        return loss

def trainFunc(network, train_loader, epochs, optimizer, save_name, val_loader,
              criterion=nn.MSELoss(), gen_log=True,
              save_model=True):
    loss_list = []
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    # Train step
    train_net = CustomTrainOneStep(network, optimizer, criterion)
    # Start training
    network.set_train(True)
    for epoch in range(epochs):
        epoch_loss = 0
        step = 0
        train_loader= tqdm(train_loader, desc=f'Epoch {epoch}', leave=True)
        for data in train_loader:
            inputs, outputs = data
            loss = train_net(inputs, outputs)
            loss_list.append(float(loss.asnumpy()))
            epoch_loss += loss.asnumpy()
            step += 1

            # Update info
            train_loader.set_postfix({
                'batch': step,
                'loss': f"{loss.asnumpy():.6f}",
                'avg_loss': f"{epoch_loss / step:.6f}"
            })

        # Average loss after an epoch
        print(f'Epoch {epoch} finished, average loss: {epoch_loss / len(train_loader):.6f}')

    # Save Model
    if save_model:
        os.makedirs(f"./experiments/{save_name}")
        ms.save_checkpoint(network, f"./experiments/{save_name}/pretrained_model.ckpt")

    # Validation
    loss_val, acc_list, p_list = valFunc(network, val_loader)

    # Save Log File
    if gen_log:
        save_dict = {
            "trainName": save_name,
            "train_time": time_str,
            "epoch_num": epochs,
            "lossList": loss_list,
            "LossVal": loss_val,
            "ACCList": acc_list,
            "Plist": p_list
        }
        with open(f"./experiments/{save_name}/logfile.pickle", "wb") as f:
            pickle.dump(save_dict, f)