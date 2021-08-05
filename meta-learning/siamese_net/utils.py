import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import sample
import random


def dataset_to_dicts(dataset_path):
    structured_dataset = {}  # {alpha: {char: [fp, fp, fp, ..]}...}
    unstructured_dataset = {}  # {char1: [fp, fp ...]}

    for alphabet in os.listdir(dataset_path):

        cwd_alpha = os.path.join(dataset_path, alphabet)
        structured_dataset[alphabet] = {}

        for character in os.listdir(cwd_alpha):

            cwd_character = os.path.join(cwd_alpha, character)
            structured_dataset[alphabet][character] = []
            unstructured_dataset[character] = []

            for sample in os.listdir(cwd_character):

                cwd_sample = os.path.join(cwd_character, sample)
                structured_dataset[alphabet][character].append(cwd_sample)
                unstructured_dataset[character].append(cwd_sample)

    return structured_dataset, unstructured_dataset


def generate_dataset(dataset_dict, sample_mode, size):
    ds = [generate_random_pair(dataset_dict, sample_mode=sample_mode)
          for _ in range(size)]
    return ds


def create_task(dataset, n):
    raise NotImplementedError


def generate_random_pair(dataset_dict, sample_mode):
    state = random.randint(0, 1)

    if sample_mode == "within alphabet":

        alphabet = random.sample(dataset_dict.keys(), 1)[0]

        if state == 1:
            character = random.sample(dataset_dict[alphabet].keys(), 1)[0]
            x1, x2 = random.sample(dataset_dict[alphabet][character], 2)
            return x1, x2, state

        elif state == 0:
            character1, character2 = random.sample(
                dataset_dict[alphabet].keys(), 2)
            x1, x2 = random.sample(dataset_dict[alphabet][character1], 1)[0], random.sample(
                dataset_dict[alphabet][character2], 1)[0]

            return x1, x2, state

    elif sample_mode == "uniform":
        if state == 1:
            character = random.sample(dataset_dict.keys(), 1)[0]
            x1, x2 = random.sample(dataset_dict[character], 2)

            return x1, x2, state

        elif state == 0:
            character1, character2 = random.sample(dataset_dict.keys(), 2)
            x1, x2 = random.sample(dataset_dict[character1], 1), random.sample(
                dataset_dict[character2], 1)

            return x1[0], x2[0], state


def plot_train_graph(**kwargs):
    plt.figure(figsize=(10, 5))
    plt.title("Statistics over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")

    for stat in kwargs:
        plt.plot(kwargs[stat], label=stat)

    plt.legend()
    plt.show()
