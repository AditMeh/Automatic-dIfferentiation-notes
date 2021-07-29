import torch
from torchvision.io import read_image
from torch.utils.data import Dataset


class Ommniglot_Dataset(Dataset):
    def __init__(self, pairs, is_val, transform=None):
        self.pairs = pairs
        self.transform = transform
        self.is_val = is_val

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path_1, path_2, label = self.pairs[idx]

        img_1, img_2 = read_image(path_1), read_image(path_2)

        label = torch.Tensor([label])
        if self.transform:
            img_1 = self.transform(img_1)/255
            img_2 = self.transform(img_2)/255

        return img_1.float(), img_2.float(), label.float()
