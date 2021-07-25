import torch
import os
# from torch._C import parse_ir
from torchvision.io import read_image
from torch.utils.data import Dataset


class Ommniglot_Dataset(Dataset):
    def __init__(self, train_pairs, transform=None, target_transform=None):
        self.train_pairs = train_pairs
        self.transform = transform

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        path_1, path_2, label = self.train_pairs[idx]

        img_1, img_2 = read_image(path_1), read_image(path_2)

        label = torch.Tensor([label])
        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return img_1.float(), img_2.float(), label.float()
