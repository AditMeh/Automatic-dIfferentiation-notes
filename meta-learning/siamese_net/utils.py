import os
import pickle
import numpy as np

DATASET_SIZE = 30000


def generate_random_pairs(dataset_path):
    dataset_pairs = []

    for alphabet in os.listdir(dataset_path):

        cwd_alpha = os.path.join(dataset_path, alphabet)
        # Generate 500 pairs of images belonging to the different characters
        for _ in range(500):
            sample = random_sampler(False, cwd_alpha)
            dataset_pairs.append(sample)

        # Generate 500 paris of images belonging to same characters
        for _ in range(500):
            sample = random_sampler(True, cwd_alpha)
            dataset_pairs.append(sample)

    return dataset_pairs


def random_sampler(flag_same, wd):
    if flag_same == False:
        char_1, char_2 = np.random.choice(
            os.listdir(wd), size=2, replace=False)

        char_1_wd, char_2_wd = os.path.join(
            wd, char_1), os.path.join(wd, char_2)

        return (os.path.join(char_1_wd, np.random.choice(os.listdir(char_1_wd))), os.path.join(char_2_wd, np.random.choice(os.listdir(char_2_wd))), int(flag_same))

    else:
        char = np.random.choice(os.listdir(wd))
        char_wd = os.path.join(wd, char)

        path_1, path_2 = (os.path.join(char_wd, path) for path in np.random.choice(
            os.listdir(char_wd), size=2, replace=False))
        return (path_1, path_2, int(flag_same))
