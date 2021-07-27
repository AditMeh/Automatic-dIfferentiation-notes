from torchvision.io.image import decode_image
from torchvision.transforms.transforms import ToTensor
from dataloader import Ommniglot_Dataset
from torch.utils.data import DataLoader
from utils import generate_random_pairs, check_pickle_exists
from model import SiameseNet
from train import train
import torch

DATASET_PATH = "omniglot/python/images_background/"


if __name__ == "__main__":
    ds = check_pickle_exists(DATASET_PATH)

    ommniglot_dataset = Ommniglot_Dataset(
        train_pairs=ds)

    ommniglot_dataloader = DataLoader(
        ommniglot_dataset, batch_size=32, shuffle=True)

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
              
    print(f"Training on device {device}.")

    net = SiameseNet().to(device=device)

    train(20, net, ommniglot_dataloader, 0.00005, device, 32)
