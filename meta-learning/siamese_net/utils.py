import os
import numpy as np
import matplotlib.pyplot as plt


def generate_random_pairs(dataset_path, dataset_size, train):
    split = calculate_dataset_split(dataset_size, train)

    dataset_pairs = []
    for alphabet in os.listdir(dataset_path):

        cwd_alpha = os.path.join(dataset_path, alphabet)
        # Generate 500 pairs of images belonging to the different characters
        for _ in range(split//2):
            sample = random_sampler(False, cwd_alpha)
            dataset_pairs.append(sample)

        # Generate 500 paris of images belonging to same characters
        for _ in range(split//2):
            sample = random_sampler(True, cwd_alpha)
            dataset_pairs.append(sample)

        np.random.shuffle(dataset_pairs)
    return dataset_pairs


def calculate_dataset_split(dataset_size, train):
    if train:
        return dataset_size // 30
    else:
        return dataset_size // 20


def random_sampler(flag_same, wd):
    if flag_same == False:
        rng = np.random.randint(0, 2)

        if rng % 2 == 0:
            # Alphabet same, class different
            char_1, char_2 = np.random.choice(
                os.listdir(wd), size=2, replace=False)

            char_1_wd, char_2_wd = os.path.join(
                wd, char_1), os.path.join(wd, char_2)

            return (os.path.join(char_1_wd, np.random.choice(os.listdir(char_1_wd))), os.path.join(char_2_wd, np.random.choice(os.listdir(char_2_wd))), int(flag_same))
        elif rng % 2 == 1:

            base_wd = os.path.dirname(wd)
            wd_new = os.path.join(
                base_wd, np.random.choice(os.listdir(base_wd)))

            char_1_wd, char_2_wd = os.path.join(wd, np.random.choice(os.listdir(
                wd))), os.path.join(wd_new, np.random.choice(os.listdir(wd_new)))

            return (os.path.join(char_1_wd, np.random.choice(os.listdir(char_1_wd))), os.path.join(char_2_wd, np.random.choice(os.listdir(char_2_wd))), int(flag_same))

    else:
        char = np.random.choice(os.listdir(wd))
        char_wd = os.path.join(wd, char)

        path_1, path_2 = (os.path.join(char_wd, path) for path in np.random.choice(
            os.listdir(char_wd), size=2, replace=False))
        return (path_1, path_2, int(flag_same))


def plot_train_graph(**kwargs):
    plt.figure(figsize=(10, 5))
    plt.title("Statistics over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")

    for stat in kwargs:
        plt.plot(kwargs[stat], label=stat)

    plt.legend()
    plt.show()
