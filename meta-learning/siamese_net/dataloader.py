from typing import Iterable
import torch
from torchvision.io import read_image
from torch.utils.data import IterableDataset, Dataset
from utils import generate_random_pair
import cv2


class Ommniglot_Dataset(Dataset):
    def __init__(self, pairs, is_val, transform=None):
        self.pairs = pairs
        self.transform = transform
        self.is_val = is_val

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path_1, path_2, label = self.pairs[idx]

        img_1, img_2 = cv2.imread(path_1, 0), cv2.imread(path_2, 0)

        img_1, img_2 = torch.Tensor(img_1), torch.Tensor(img_2)
        label = torch.Tensor([label])
        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return torch.unsqueeze(img_1.float(), 0), torch.unsqueeze(img_2.float(), 0), label.float()


class RandomPairSampler(IterableDataset):
    def __init__(self, sample_mode, dataset, is_val):
        self.sample_mode = sample_mode
        self.dataset = dataset
        self.is_val = is_val

    def return_data(self):
        while True:
            x1, x2, label = generate_random_pair(
                dataset=self.dataset, sample_mode=self.sample_mode)

            img_1, img_2 = cv2.imread(x1, 0), cv2.imread(x2, 0)

            img_1, img_2 = torch.Tensor(img_1), torch.Tensor(img_2)
            label = torch.Tensor([label])

            yield torch.unsqueeze(img_1.float(), 0), torch.unsqueeze(img_2.float(), 0), label.float()

    def __iter__(self):
        return self.return_data()
