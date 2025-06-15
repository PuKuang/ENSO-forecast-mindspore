"""
@Project ：ENSO-forecast-mindspore
@File    ：train.py
@Author  ：Huang Zihan
@Date    ：2025/6/14
"""

import os.path
import mindspore as ms
from mindspore import nn
from mindspore.dataset import GeneratorDataset
from data import ENSODataset, create_mindspore_dataloader
import TrainFuncVal as TFV
from model import ConvNetwork
from utils.pickle_plot import trainPlot
import configargparse

def parse_config():
    # experiment
    parser = configargparse.ArgumentParser(description='Train CNN ENSO prediction model with CMIP dataset')
    parser.add_argument('--device', type=str, default="CPU")
    parser.add_argument('--exp_name', type=str, default='pre-0614-lr2-e12-bs500')
    parser.add_argument('--pretrain', type=bool, default=True)

    # CNN model parameter
    parser.add_argument('--M_Num', type=int, default=30)
    parser.add_argument('--N_Num', type=int, default=30)

    # dataset time range
    parser.add_argument('--CMIP_start', type=int, default=1850)
    parser.add_argument('--CMIP_end', type=int, default=1973)
    parser.add_argument('--OBSTrain_start', type=int, default=1871)
    parser.add_argument('--OBSTrain_end', type=int, default=1973)
    parser.add_argument('--OBSVal_start', type=int, default=1980)
    parser.add_argument('--OBSVal_end', type=int, default=2019)

    # training
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--epochs', type=int, default=12)
    args = parser.parse_args()
    return args


def concat_mindspore_datasets(dataset1, dataset2, batch_size=100, shuffle=True):
    # 转换为GeneratorDataset
    ds1 = GeneratorDataset(source=dataset1, column_names=["data", "label"], shuffle=shuffle)
    ds2 = GeneratorDataset(source=dataset2, column_names=["data", "label"], shuffle=shuffle)
    combined_ds = ds1 + ds2

    # 批处理和类型转换
    combined_ds = combined_ds.batch(batch_size, drop_remainder=False)
    type_cast_op = ms.dataset.transforms.TypeCast(ms.float32)
    combined_ds = combined_ds.map(operations=type_cast_op, input_columns="data")
    combined_ds = combined_ds.map(operations=type_cast_op, input_columns="label")

    return combined_ds

if __name__ == '__main__':
    args =parse_config()
    exp_name = args.exp_name
    exp_dir = f'experiments/{exp_name}'
    if os.path.exists(exp_dir):
        raise ValueError('Experiment dir already existed, please rename this experiment!')

    pretrain = args.pretrain
    # Device
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)
    # Model
    model = ConvNetwork(M_Num=args.M_Num, N_Num=args.N_Num)
    # Data
    if not pretrain:
        # Use both datasets together for training
        pretrain_dataset = ENSODataset("CMIP",T_begin=args.CMIP_start, T_end=args.CMIP_end)
        finetune_dataset = ENSODataset("OBSTrain", T_begin=args.OBSTrain_start, T_end=args.OBSTrain_end)
        train_loader = concat_mindspore_datasets(pretrain_dataset, finetune_dataset, batch_size=args.batch_size)
    else:
        # use CMIP for pretrain and use OBSTrain for finetune later in "finetune.py"
        train_loader = create_mindspore_dataloader("CMIP", batch_size=args.batch_size, T_begin=args.CMIP_start, T_end=args.CMIP_end)
    val_loader = create_mindspore_dataloader("OBSVal", batch_size=args.batch_size, T_begin=args.OBSVal_start, T_end=args.OBSVal_end)

    # Optimizer
    optimizer = nn.Adam(model.trainable_params(), learning_rate=args.lr)

    # Start training
    TFV.trainFunc(
        network=model,
        train_loader=train_loader,
        epochs=args.epochs,
        optimizer=optimizer,
        save_name=args.exp_name,
        val_loader=val_loader
    )

    # save fig and visualize
    file_path = f"experiments/{exp_name}/logfile.pickle"
    save_path = f"experiments/{exp_name}/result_fig.png"
    trainPlot(filepath=file_path, save_path=save_path)


