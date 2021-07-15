from utils import generate_random_pairs
import pickle
import numpy as np
import os
DATASET_PATH = "omniglot/python/images_background/"


if __name__ == "__main__":
    if not os.path.exists("pairs.pickle"):
        ds = generate_random_pairs(DATASET_PATH)
        np.random.shuffle(ds)

        with open('pairs.pickle', 'wb') as f:
            pickle.dump(ds, f)
    
    with open('pairs.pickle') as f:
        ds = pickle.load(f)
    

