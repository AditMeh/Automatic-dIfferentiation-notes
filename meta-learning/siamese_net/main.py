from dataloader import create_task_dataloader
from random import sample
from dataloader import RandomPairSampler, Ommniglot_Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import dataset_to_dicts
from model import SiameseNet
from train import train_fixed_dataset, train_random_samples
import torch

TRAIN_DATASET_PATH = "omniglot/python/images_background/"
VALIDATION_DATASET_PATH = "omniglot/python/images_evaluation/"


if __name__ == "__main__":
    TRAIN_DATASET_PATH = "omniglot/python/images_background/"
    VALIDATION_DATASET_PATH = "omniglot/python/images_evaluation/"

    ds_train_structured, ds_train_unstructured = dataset_to_dicts(
        TRAIN_DATASET_PATH)
    ds_val_structured, ds_val_unstructured = dataset_to_dicts(
        VALIDATION_DATASET_PATH)

    transforms_seq = transforms.Compose([transforms.RandomRotation((-20, 20)),
                                         transforms.GaussianBlur(3, (0.1, 5))])

    train_dataset_random = RandomPairSampler(
        dataset_dict=ds_train_unstructured, sample_mode="uniform", is_val=False, transform=transforms_seq)

    val_dataset_random = RandomPairSampler(
        dataset_dict=ds_val_unstructured, sample_mode="uniform", is_val=True, transform=transforms_seq)

    train_loader_random = DataLoader(
        train_dataset_random, batch_size=32, num_workers=2)

    val_loader_random = DataLoader(
        val_dataset_random, batch_size=32, num_workers=2)

    train_dataset_fixed = Ommniglot_Dataset(30000,
                                            dataset_dict=ds_train_structured, sample_mode="within alphabet", is_val=False, transform=transforms_seq)

    val_dataset_fixed = Ommniglot_Dataset(10000,
                                          dataset_dict=ds_val_structured, sample_mode="within alphabet", is_val=True, transform=transforms_seq)

    train_loader_fixed = DataLoader(
        train_dataset_fixed, batch_size=32, num_workers=2)

    val_loader_fixed = DataLoader(
        val_dataset_fixed, batch_size=32, num_workers=2)

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    print(f"Training on device {device}.")

    net = SiameseNet().to(device=device)

    for x1, x2, label in train_loader_fixed:
        print(x1.shape)
        break

    val_loss_history_fixed, train_loss_history_fixed = train_fixed_dataset(
        net, train_loader=train_loader_fixed, val_loader=val_loader_fixed, n_epochs=20, lr=0.00001, device=device, batch_size=32, save_path="wew")
    # val_loss_history_random, train_loss_history_random = train_random_samples(net, train_loader_fixed, val_loader_fixed, samples_per_epoch=30000, samples_val=10000, n_epochs=20,
    #         lr=0.00001, device=device, batch_size=32, save_path="blank")


    # from utils import evaluate_model_on_task

    # task_loader = create_task_dataloader(
    #     ds_train_unstructured, 10, sample_mode="uniform")

    # net = SiameseNet()

    # pred = []
    # for x1, x2, label in task_loader:
    #     pred.append(label)

    # predictions = torch.hstack(tuple(pred))
    # print(evaluate_model_on_task(predictions))
