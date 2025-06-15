"""
@Project ：ENSO-forecast-mindspore
@File    ：DataLoaderFunc_MindSpore.py
@Author  ：Huang Zihan
@Date    ：2025/6/14
"""


import numpy as np
import xarray as xr
from mindspore import dataset as ds
from mindspore import dtype as mstype
import matplotlib.pyplot as plt
import mindspore as ms

# 多个CMIP数据合在一起用于模型的预训练
CMIPTosLoc_1 = r"D:\Desktop\huawei-ai\ENSO-forecast-mindspore\TrainData\GFDL-ESM4_TosA_rename.nc"
CMIPZosLoc_1 = r"D:\Desktop\huawei-ai\ENSO-forecast-mindspore\TrainData\GFDL-ESM4_ZosA_rename.nc"
CMIPNinoLoc_1 = r"D:\Desktop\huawei-ai\ENSO-forecast-mindspore\TrainData\GFDL-ESM4_Nino34I_rename.nc"

CMIPTosLoc_2 = r"D:\Desktop\huawei-ai\ENSO-forecast-mindspore\TrainData\NorESM2-MM_TosA_rename.nc"
CMIPZosLoc_2 = r"D:\Desktop\huawei-ai\ENSO-forecast-mindspore\TrainData\NorESM2-MM_ZosA_rename.nc"
CMIPNinoLoc_2 = r"D:\Desktop\huawei-ai\ENSO-forecast-mindspore\TrainData\NorESM2-MM_Nino34.nc"

CMIPTosLoc_3 = r"D:\Desktop\huawei-ai\ENSO-forecast-mindspore\TrainData\IPSL-CM6A-LR_TosA_rename.nc"
CMIPZosLoc_3 = r"D:\Desktop\huawei-ai\ENSO-forecast-mindspore\TrainData\IPSL-CM6A-LR_ZosA_rename.nc"
CMIPNinoLoc_3 = r"D:\Desktop\huawei-ai\ENSO-forecast-mindspore\TrainData\IPSL-CM6A-LR_Nino34.nc"

# OBSTrain数据用于微调
OBSTrainSSTALoc = r"D:\Desktop\huawei-ai\ENSO-forecast-mindspore\TrainData\ersstv5ssta.nc"
OBSTrainSSHALoc = r"D:\Desktop\huawei-ai\ENSO-forecast-mindspore\TrainData\SODAssha.nc"
OBSTrainNinoLoc = r"D:\Desktop\huawei-ai\ENSO-forecast-mindspore\TrainData\ersstv5Nino34.nc"

# OBSVal数据用于验证
OBSValSSTALoc = r"D:\Desktop\huawei-ai\ENSO-forecast-mindspore\ValidationData\ersstv5ssta.nc"
OBSValSSHALoc = r"D:\Desktop\huawei-ai\ENSO-forecast-mindspore\ValidationData\GODASssha.nc"
OBSValNinoLoc = r"D:\Desktop\huawei-ai\ENSO-forecast-mindspore\ValidationData\ersstv5Nino34.nc"

class ENSODataset:
    def __init__(self, type_, T_begin, T_end):
        self.Type = type_
        if type_ == "CMIP":
            datasets = [
                self._load_single_cmip(CMIPTosLoc_1, CMIPZosLoc_1, CMIPNinoLoc_1, T_begin, T_end),
                self._load_single_cmip(CMIPTosLoc_2, CMIPZosLoc_2, CMIPNinoLoc_2, T_begin, T_end),
                self._load_single_cmip(CMIPTosLoc_3, CMIPZosLoc_3, CMIPNinoLoc_3, T_begin, T_end)
            ]
            self.all_samples = []
            for ssta, ssha, nino in datasets:
                samples = self._generate_samples(ssta, ssha, nino)
                self.all_samples.extend(samples)
            self.DataTimeLen = len(self.all_samples)
        elif type_ == "OBSTrain":
            datasets = [
                self._load_single_cmip(OBSTrainSSTALoc, OBSTrainSSHALoc, OBSTrainNinoLoc, T_begin, T_end)
            ]
            self.all_samples = []
            for ssta, ssha, nino in datasets:
                samples = self._generate_samples(ssta, ssha, nino)
                self.all_samples.extend(samples)
            self.DataTimeLen = len(self.all_samples)
        elif type_ == "OBSVal":
            datasets = [
                self._load_single_cmip(OBSValSSTALoc, OBSValSSHALoc, OBSValNinoLoc, T_begin, T_end)
            ]
            self.all_samples = []
            for ssta, ssha, nino in datasets:
                samples = self._generate_samples(ssta, ssha, nino)
                self.all_samples.extend(samples)
            self.DataTimeLen = len(self.all_samples)
        else:
            raise ValueError("Data type must be CMIP/OBSTrain/OBSVal")

    def _load_single_cmip(self, tos_path, zos_path, nino_path, T_begin, T_end):
        ssta = xr.open_dataset(tos_path)["ssta"].squeeze(drop=True).fillna(0)
        ssha = xr.open_dataset(zos_path)["ssha"].fillna(0)
        nino = xr.open_dataset(nino_path)["nino34"].squeeze(drop=True).fillna(0)

        if T_begin is not None:
            ssta_need_time = (ssta["time"].dt.year >= T_begin) & (ssta["time"].dt.year <= T_end)
            ssha_need_time = (ssha["time"].dt.year >= T_begin) & (ssha["time"].dt.year <= T_end)
            ssta, ssha, nino = ssta[ssta_need_time], ssha[ssha_need_time], nino[ssta_need_time]

        return ssta, ssha, nino

    def _generate_samples(self, ssta, ssha, nino):
        samples = []
        max_idx = len(nino) - 3 - 23
        for i in range(max_idx):
            x1 = np.array(ssta[i:i + 3])  # SSTA for 3 months
            x2 = np.array(ssha[i:i + 3])  # SSHA for 3 months
            y = np.array(nino[i + 3:i + 3 + 23])  # Nino3.4 for 23 months
            samples.append((np.concatenate([x1, x2], axis=0), y))
        return samples

    def __getitem__(self, index):
        x_data, y_data = self.all_samples[index]
        return x_data.astype(np.float32), y_data.astype(np.float32)

    def __len__(self):
        if self.Type == "CMIP":
            return int(self.DataTimeLen - 3 - 23)
        else:
            return int(self.DataTimeLen - 3 - 23 + 1)


def create_mindspore_dataloader(dataset_type, T_begin, T_end, batch_size=32, shuffle=True):
    custom_dataset = ENSODataset(dataset_type, T_begin=T_begin, T_end=T_end)
    ms_dataset = ds.GeneratorDataset(
        source=custom_dataset,
        column_names=["data", "label"],
        shuffle=shuffle,
        num_parallel_workers=1  # Windows平台必须设置为1
    )
    type_cast_op = ds.transforms.TypeCast(mstype.float32)

    ms_dataset = ms_dataset.map(
        operations=type_cast_op,
        input_columns="data"
    )
    ms_dataset = ms_dataset.map(
        operations=type_cast_op,
        input_columns="label"
    )
    ms_dataset = ms_dataset.batch(batch_size, drop_remainder=False)

    return ms_dataset


def visualize_zeros_nonzeros(tensor, zero_color='white', non_zero_color='red', show_values=True):
    """
    可视化 2D Tensor 中的 0 值和非 0 值

    参数:
        tensor (mindspore.Tensor): 输入的 2D Tensor
        zero_color (str): 0 值的颜色（默认白色）
        non_zero_color (str): 非 0 值的颜色（默认红色）
        show_values (bool): 是否在图上显示数值（默认显示）
    """
    data = tensor.asnumpy() if isinstance(tensor, ms.Tensor) else np.array(tensor)
    cmap = plt.cm.colors.ListedColormap([zero_color, non_zero_color])
    plt.imshow(data != 0, cmap=cmap, interpolation='nearest')
    if show_values:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                plt.text(j, i, f"{data[i, j]:.2f}",
                         ha='center', va='center',
                         color='black' if data[i, j] != 0 else 'gray')
    plt.colorbar(ticks=[0, 1], label=f"0 ({zero_color}) vs 非0 ({non_zero_color})")
    plt.title("0 value vs non 0 value")
    # plt.axis('off')  # 可选：隐藏坐标轴
    # plt.show()


if __name__ == '__main__':
    # test dataset and dataloader
    train_loader = create_mindspore_dataloader("CMIP", batch_size=400, T_begin=1850, T_end=1973)
    # train_loader = create_mindspore_dataloader("OBSVal", batch_size=400, T_begin=1980, T_end=2019)
    # train_loader = create_mindspore_dataloader("OBSTrain", batch_size=400, T_begin=1871, T_end=1973)

    for i, (data, label) in enumerate(train_loader):
        print('\nlabel min, max, mean:', label.min(), label.max(), label.mean())
        print(f"批次 {i + 1}:")
        print("输入数据类型:", type(data), "形状:", data.shape)
        print('data nan ratio:', data.isnan().float().mean().item())
        visualize_zeros_nonzeros(data[0][0])
        print("标签数据类型:", type(label), "形状:", label.shape)
        print('label nan ratio:', label.isnan().float().mean().item())