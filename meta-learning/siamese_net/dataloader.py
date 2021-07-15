import torch
import os
from torch._C import parse_ir
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class Ommniglot_Dataset(Dataset):
    def __init__(self, train_pairs, transform=None, target_transform=None):
        self.train_pairs = train_pairs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        path_1, path_2, label = self.train_pairs[idx]


        img_1, img_2 = read_image(path_1), read_image(path_2)
        
        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.trainform(img_2)
        if self.target_transform:
            label = self.target_transform(label)
        return img_1, img_2, label
