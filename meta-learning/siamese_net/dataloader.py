from random import sample
from typing import Iterable
import torch
from torchvision.io import read_image
from torch.utils.data import IterableDataset, Dataset, DataLoader
from utils import generate_random_pair, generate_dataset, dataset_to_dicts
import cv2


class Ommniglot_Dataset(Dataset):
    def __init__(self, size, dataset_dict, is_val, sample_mode, transform=None):
        self.size = size
        self.dataset_dict = dataset_dict
        self.transform = transform
        self.is_val = is_val
        self.sample_mode = sample_mode

        self.dataset = generate_dataset(
            self.dataset_dict, sample_mode=self.sample_mode, size=self.size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path_1, path_2, label = self.dataset[idx]
        #print(path_1)

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


if __name__ == "__main__":
    TRAIN_DATASET_PATH = "omniglot/python/images_background/"
    ds_train_structured, ds_train_unstructured = dataset_to_dicts(
        TRAIN_DATASET_PATH)

    loader = Ommniglot_Dataset(
        30000, ds_train_structured, is_val=None, sample_mode="within alphabet")

    loader = DataLoader(
        loader, batch_size=32)
    x1, x2, x3 = next(iter(loader))

    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
