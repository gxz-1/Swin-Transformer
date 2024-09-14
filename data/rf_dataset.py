import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import scipy.io as sio
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F

from data.rf_transformer import EnergyToTensor

# class EnergyToTensor:
#     def __init__(self, normalization='minmax'):
#         """
#         normalization: 'minmax' or 'log', 根据需要选择不同的归一化方式
#         """
#         self.normalization = normalization

#     def __call__(self, data):
#         """
#         将数据转换为张量并进行标准化
#         """
#         if isinstance(data, np.ndarray):
#             # # 使用 torchvision 的 center_crop 裁剪图像到 224x224
#             Pxx = Image.fromarray(data)
#             Pxx = F.center_crop(Pxx, (224, 224))
#             Pxx = np.array(Pxx)# 再将裁剪后的 PIL Image 转换回 NumPy array
#             # 增加通道维度 (1, H, W)
#             Pxx = np.expand_dims(Pxx, axis=0)  
#             if self.normalization == 'minmax':
#                 # 最小-最大归一化
#                 Pxx_min = np.min(Pxx)
#                 Pxx_max = np.max(Pxx)
#                 Pxx_normalized = (Pxx - Pxx_min) / (Pxx_max - Pxx_min + 1e-8)  # 防止除零
#             elif self.normalization == 'log':
#                 # 对数缩放
#                 Pxx_normalized = np.log(Pxx + 1e-8)  # 防止 log(0)
#             else:
#                 raise ValueError("Unknown normalization method")
#             Pxx_normalized = torch.tensor(Pxx_normalized, dtype=torch.float32)
#             # Pxx_normalized = Pxx_normalized.repeat(3, 1, 1) #单通道图像复制 3 次来模拟 RGB 图像
#             return Pxx_normalized
#         elif isinstance(data, torch.Tensor):
#             return data.float()
#         else:
#             raise TypeError("Input should be a NumPy array or PyTorch Tensor")

class RFDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): 数据集文件夹路径，包含不同类别的子文件夹
            transform (callable, optional): 可选的转换函数 (如数据增强)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []  # 保存样本路径和对应的标签

        # 遍历数据文件夹，假设文件夹结构为 data_dir/class_name/*.mat
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for mat_file in os.listdir(class_dir):
                    if mat_file.endswith('.mat'):
                        file_path = os.path.join(class_dir, mat_file)
                        self.samples.append((file_path, class_name))

        # 将类别名称映射为索引
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(os.listdir(data_dir))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引返回样本和标签
        """
        mat_path, class_name = self.samples[idx]
        label = self.class_to_idx[class_name]

        # 读取 .mat 文件中的数据
        mat_data = sio.loadmat(mat_path)
        Pxx = mat_data['Pxx']
        freqs = mat_data['freqs'][0]  # 提取 freqs
        bins = mat_data['bins'][0]    # 提取 bins

        # 根据模型的需要对 Pxx 进行处理，例如将其形状调整为适合输入模型的格式
        self.transform = transforms.Compose([
            EnergyToTensor(normalization='log-symmetric',augment=False)  # 可以选择log或minmax
        ])
        if self.transform:
            Pxx = self.transform(Pxx)

        # 将标签转换为 Tensor
        label = torch.tensor(label, dtype=torch.long)

        return Pxx, label
