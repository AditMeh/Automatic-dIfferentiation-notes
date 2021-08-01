from dataloader import Ommniglot_Dataset
from torch.utils.data import DataLoader
from utils import generate_random_pairs
from model import SiameseNet
from train import train
import torch

TRAIN_DATASET_PATH = "omniglot/python/images_background/"
VALIDATION_DATASET_PATH = "omniglot/python/images_evaluation/"


if __name__ == "__main__":
    ds_train = generate_random_pairs(TRAIN_DATASET_PATH, 30000, train=True)
    ds_val = generate_random_pairs(VALIDATION_DATASET_PATH, 10000, train=False)

    train_dataset = Ommniglot_Dataset(pairs=ds_train, is_val=False)

    val_dataset = Ommniglot_Dataset(pairs=ds_val, is_val=True)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True)

    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=True)

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    print(f"Training on device {device}.")

    net = SiameseNet().to(device=device)

    train(net, train_loader, val_loader, n_epochs=20,
          lr=0.001, device=device, batch_size=32, save_path="blank")
