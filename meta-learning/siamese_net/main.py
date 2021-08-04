from random import sample
from dataloader import RandomPairSampler
from torch.utils.data import DataLoader
from utils import dataset_to_dicts
from model import SiameseNet
from train import train
import torch

TRAIN_DATASET_PATH = "omniglot/python/images_background/"
VALIDATION_DATASET_PATH = "omniglot/python/images_evaluation/"


if __name__ == "__main__":
    ds_train_structured, ds_train_unstructured = dataset_to_dicts(
        TRAIN_DATASET_PATH)
    ds_val_structured, ds_val_unstructured = dataset_to_dicts(
        VALIDATION_DATASET_PATH)

    train_dataset = RandomPairSampler(
        dataset=ds_train_unstructured, sample_mode="uniform", is_val=False)

    val_dataset = RandomPairSampler(
        dataset=ds_val_unstructured, sample_mode="uniform", is_val=True)

    train_loader = DataLoader(
        train_dataset, batch_size=32)

    val_loader = DataLoader(
        val_dataset, batch_size=32)

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    print(f"Training on device {device}.")

    net = SiameseNet().to(device=device)

    train(net, train_loader, val_loader, n_epochs=20,
          lr=0.001, device=device, batch_size=32, save_path="blank")
