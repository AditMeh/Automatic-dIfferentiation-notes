from random import sample
import torch
from torchvision.io import read_image
from torch.utils.data import IterableDataset, Dataset, DataLoader
from utils import generate_random_pair, generate_dataset, create_task_files
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
        # print(path_1)

        img_1, img_2 = cv2.imread(path_1, 0), cv2.imread(path_2, 0)

        img_1, img_2 = torch.Tensor(img_1), torch.Tensor(img_2)
        label = torch.Tensor([label])
        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return torch.unsqueeze(img_1.float(), 0), torch.unsqueeze(img_2.float(), 0), label.float()


class RandomPairSampler(IterableDataset):
    def __init__(self, sample_mode, dataset_dict, is_val):
        self.sample_mode = sample_mode
        self.dataset_dict = dataset_dict
        self.is_val = is_val

    def return_data(self):
        while True:
            x1, x2, label = generate_random_pair(
                dataset_dict=self.dataset_dict, sample_mode=self.sample_mode)

            img_1, img_2 = cv2.imread(x1, 0), cv2.imread(x2, 0)

            img_1, img_2 = torch.Tensor(img_1), torch.Tensor(img_2)
            label = torch.Tensor([label])

            yield torch.unsqueeze(img_1.float(), 0), torch.unsqueeze(img_2.float(), 0), label.float()

    def __iter__(self):
        return self.return_data()


class TaskDataset(Dataset):
    def __init__(self, pairs) -> None:
        self.pairs = pairs
        self.transforms = None

    def __getitem__(self, idx):
        x1, x2, label = self.pairs[idx]

        return self._load_file_as_image(x1), self._load_file_as_image(x2), torch.Tensor([label]).float

    def __len__(self):
        return len(self.pairs)

    def _load_file_as_image(path):
        img = cv2.imread(path, 0)
        img = torch.Tensor(img)
        return torch.unsqueeze(img.float(), 0)


def create_task_dataloader(pairs, N):
    batch_size = N

    pair_dataset = TaskDataset(pairs)
    task_loader = DataLoader(pair_dataset, batch_size=batch_size, num_workers=2)

    return task_loader