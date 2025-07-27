import os
import torch
import PIL.Image as Image
import torchvision.transforms as transforms
import numpy

def calculate_mean_std(dataset_path):
    """
    Calculate the mean and std of RGB channels.

    Args:
        dataset_path (str): Path to the dataset directory.
    """

    # Get all class names
    classes = [d for d in os.listdir(dataset_path)
               if os.path.isdir(os.path.join(dataset_path, d))]

    # Collect full paths
    image_files = []
    for cls in classes:
        cls_dir = os.path.join(dataset_path, cls)
        image = [os.path.join(cls_dir, f)
                 for f in os.listdir(cls_dir)
                 if os.path.isfile(os.path.join(cls_dir, f))]
        image_files.extend(image)

    num_images = len(image_files)

    # Iterate over all image files
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms.ToTensor()(img)
        mean += torch.mean(img_tensor, dim=(1, 2))
        std += torch.std(img_tensor, dim=(1,2))

    # Average
    mean /= num_images
    std /= num_images

    return mean.tolist(), std.tolist()


def main():
    """
    Main entry: calculate the mean and std.
    """

    dataset_path = "../datasets/train"
    mean, std = calculate_mean_std(dataset_path)

    print(f"Mean: {mean},\nStd: {std}")


if __name__ == '__main__':
    main()
