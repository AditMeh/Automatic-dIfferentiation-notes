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

    train_dataset_random = RandomPairSampler(
        dataset_dict=ds_train_structured, sample_mode="uniform", is_val=False)

    val_dataset_random = RandomPairSampler(
        dataset_dict=ds_train_structured, sample_mode="within alphabet", is_val=True)

    train_loader_random = DataLoader(
        train_dataset_random, batch_size=32)

    val_loader_random = DataLoader(
        val_dataset_random, batch_size=32)

    from dataloader import create_task_dataloader

    from utils import evaluate_model_on_task


    task_loader = create_task_dataloader(
        ds_train_unstructured, 10, sample_mode="uniform")

    net = SiameseNet()

    pred = []
    for x1, x2, label in task_loader:
        pred.append(label)

    predictions = torch.hstack(tuple(pred))
    print(evaluate_model_on_task(predictions))

