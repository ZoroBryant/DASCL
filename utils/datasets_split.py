import os
import random
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

# Set fixed seed
fixed_seed = 2025
random.seed(fixed_seed)
np.random.seed(fixed_seed)

def split_dataset(src_dir, dataset_path, train_size=0.9, random_state=fixed_seed):
    """
    Split  dataset into train and validation sets.

    Args:
        src_dir (str): Path to original dataset directory.
        dataset_path (str): Path to output dataset root.
        train_size (float): Ratio of train sets.
        random_state (int): Seed for reproducibility.
    """

    train_dir = os.path.join(dataset_path, 'train_valid')
    valid_dir = os.path.join(dataset_path, 'valid')

    # Remove old train/valid directories
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(valid_dir):
        shutil.rmtree(valid_dir)

    # Create new train/valid directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    # Get all class names
    classes = [d for d in os.listdir(src_dir)
               if os.path.isdir(os.path.join(src_dir, d))]


    for index, cls in enumerate(classes):
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, cls), exist_ok=True)

        cls_dir = os.path.join(src_dir, cls)
        paths = [os.path.join(cls_dir, f)
                 for f in os.listdir(cls_dir)
                 if os.path.isfile(os.path.join(cls_dir, f))]

        cls_random_state = random_state * (index + 1)
        # Split into train and validation sets
        train_files, test_files = train_test_split(
            paths,
            train_size=train_size,
            random_state=cls_random_state,
            shuffle=True
        )

        # Copy files
        for file in train_files:
            shutil.copy(file, os.path.join(train_dir, cls, os.path.basename(file)))

        for file in test_files:
            shutil.copy(file, os.path.join(valid_dir, cls, os.path.basename(file)))

        print(f"Class {cls} split complete.")
    print(f"All Classes split complete!")


def main():
    """
    Main entry: split dataset.
    """

    src_dir = os.path.normpath("../datasets/train")
    dataset_path = os.path.normpath("../datasets")

    split_dataset(src_dir, dataset_path)

if __name__ == '__main__':
    main()




