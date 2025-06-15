"""
@Project ：ENSO-forecast-mindspore
@File    ：finetune.py
@Author  ：Huang Zihan
@Date    ：2025/6/14

Fine-tuning script for ENSO prediction model
"""

import mindspore as ms
from mindspore import nn
from mindspore.dataset import GeneratorDataset
from data import ENSODataset, create_mindspore_dataloader
import TrainFuncVal as TFV
from model import ConvNetwork
from utils.pickle_plot import trainPlot
import configargparse
import os

def parse_config():
    parser = configargparse.ArgumentParser(
        description='Fine-tune CNN ENSO prediction model with observation dataset'
    )
    parser.add_argument('--device', type=str, default="CPU")
    parser.add_argument('--exp_name', type=str, default='finetune_lr5_e10_bs_200_pre-0614-lr2-e12-bs500')
    parser.add_argument('--pretrained_model', type=str,
                        default=r'D:\Desktop\huawei-ai\ENSO-forecast-mindspore\experiments\pre-0614-lr2-e12-bs500\pretrained_model.ckpt',
                       help='Path to pretrained model parameters')

    # CNN model parameter (should match pretrained model)
    parser.add_argument('--M_Num', type=int, default=30)
    parser.add_argument('--N_Num', type=int, default=30)

    # dataset time range for fine-tuning
    parser.add_argument('--finetune_start', type=int, default=1871)
    parser.add_argument('--finetune_end', type=int, default=1973)
    parser.add_argument('--val_start', type=int, default=1980)
    parser.add_argument('--val_end', type=int, default=2019)

    # fine-tuning parameters
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.00005)  # typically smaller than pretraining lr
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--freeze_layers', default=True,
                       help='Whether to freeze some layers during fine-tuning')

    args = parser.parse_args()
    return args

def load_pretrained_model(model, pretrained_path):
    """Load pretrained model parameters"""
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained model not found at {pretrained_path}")

    param_dict = ms.load_checkpoint(pretrained_path)
    ms.load_param_into_net(model, param_dict)
    print(f"Loaded pretrained model from {pretrained_path}")
    return model

def freeze_layers(model, freeze=True):
    """Freeze all layers except the last one"""
    print('total layers: ', len(model.trainable_params()))
    for param in model.trainable_params()[:-3]:  # keep last layer trainable
        param.requires_grad = not freeze
    print("Layers frozen:", freeze)
    return model

if __name__ == '__main__':
    args = parse_config()
    exp_name = args.exp_name

    # Device setup
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)

    # Load pretrained model
    model = ConvNetwork(M_Num=args.M_Num, N_Num=args.N_Num)
    model = load_pretrained_model(model, args.pretrained_model)

    # Optionally freeze layers
    if args.freeze_layers:
        model = freeze_layers(model)

    # Prepare fine-tuning dataset
    finetune_loader = create_mindspore_dataloader("OBSTrain", batch_size=args.batch_size,
                                                T_begin=args.finetune_start, T_end=args.finetune_end)

    # Validation dataset
    val_loader = create_mindspore_dataloader("OBSVal", batch_size=args.batch_size,
                                           T_begin=args.val_start, T_end=args.val_end)

    # Optimizer (only trainable parameters will be updated)
    optimizer = nn.Adam(filter(lambda p: p.requires_grad, model.trainable_params()),
                       learning_rate=args.lr)

    # Start fine-tuning
    TFV.trainFunc(
        network=model,
        train_loader=finetune_loader,
        epochs=args.epochs,
        optimizer=optimizer,
        save_name=args.exp_name,
        val_loader=val_loader
    )

    # Save training curves
    file_path = f"experiments/{exp_name}/logfile.pickle"
    save_path = f"experiments/{exp_name}/result_fig.png"
    trainPlot(filepath=file_path, save_path=save_path)