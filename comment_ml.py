# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# torch.cuda.is_available()           # GPU利用の場合コメントアウトを外す
