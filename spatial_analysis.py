import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.utils.data as data
from torchvision import transforms, datasets

from networks.MTC import SupConResNet
from utils.tools import visualize_tsne, median_absolute_deviation


def parse_option():

    parser = argparse.ArgumentParser('Argument for spatial analysis')

    parser.add_argument('--dataset_path', type=str, default='./datasets/train', help='path to train dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=16, help='#workers for dataloader')
    parser.add_argument('--encoder', type=str, default='resnet18', help='encoder')
    parser.add_argument('--model_path', type=str, help='path to checkpoint')

    parser.add_argument('--sne_save_path', type=str, default='./save/sne', help='dir to save t-SNE figs')
    parser.add_argument('--spatial_distribution_path', type=str, default='./save/spatial_distribution',
                         help='dir to save spatial distribution npz')

    opt = parser.parse_args()

    return opt


def set_loader(opt):

    mean, std = ((0.9920479655265808, 0.9398094415664673, 0.9398094415664673),
                 (0.08487337082624435, 0.21652734279632568, 0.21652734279632568))

    transform = transforms.Compose([
        transforms.Resize(size=(128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    dataset = datasets.ImageFolder(root=opt.dataset_path, transform=transform)

    data_loader = data.DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False,
                                  num_workers=opt.num_workers, pin_memory=True)

    return data_loader


def load_model(opt):

    print(f"Start loading model.")
    model = SupConResNet(encoder=opt.encoder)

    checkpoint = torch.load(opt.model_path)
    state_dict = checkpoint['model']

    # Load model weights with proper handling of DataParallel checkpoints.
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
            # If checkpoint was trained with DataParallel
            if any('module' in key for key in state_dict.keys()):
                model.load_state_dict(state_dict)
            else:
                # Add 'module' prefix for encoder keys
                new_state_dict = {}
                for k, v in state_dict.items():
                    if 'encoder' in k:
                        new_state_dict[k.replace('encoder.', 'encoder.module.')] = v
                    else:
                        new_state_dict[k] = v
                model.load_state_dict(new_state_dict)
        else:
            # Single GPU
            if any('module' in key for key in state_dict.keys()):
                # Remove 'module' prefix from encoder keys
                new_state_dict = {}
                for k, v in state_dict.items():
                    if 'encoder.module.' in k:
                        new_state_dict[k.replace('encoder.module.', 'encoder.')] = v
                    else:
                        new_state_dict[k] = v
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)
        model.cuda()

    return model


# Extract embeddings for each sample and group them by class.
def get_embeddings(data_loader, model, opt):

    print(f"Start calculating embeddings.")

    # Get class folders under dataset path
    classes = [d for d in os.listdir(opt.dataset_path)
               if os.path.isdir(os.path.join(opt.dataset_path, d))]
    embs_pool = {app: [] for app in range(len(classes))}

    with torch.no_grad():
        model.eval()
        for index, (images, labels) in enumerate(data_loader):
            if torch.cuda.is_available():
                images = images.cuda()
            # Forward pass to get embeddings
            embs = model(images).cpu().numpy()
            # Store embeddings per class
            for i, app in enumerate(labels.cpu().numpy()):
                embs_pool[app].append(embs[i])

    return embs_pool


# Visualize embeddings using t-SNE and save the plot.
def plot_tsne(embs_pool, opt):

    print(f"Start ploting T-SNE.")
    checkpoint = torch.load(opt.model_path)
    opt_train = checkpoint['opt']

    # Flatten embeddings and labels
    features = []
    labels = []
    for label, embs in embs_pool.items():
        features.append(np.array(embs))
        labels.extend([label] * len(embs))
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)

    # Run t-SNE visualization
    fig = visualize_tsne(features, labels)
    plt.show()

    # Save figure
    sne_save_path = os.path.join(opt.sne_save_path, opt_train.model_name)
    os.makedirs(sne_save_path, exist_ok=True)
    sne_file_name = os.path.basename(opt.model_path)
    sne_file_name_without_extension = os.path.splitext(sne_file_name)[0]

    sne_save = os.path.join(sne_save_path, sne_file_name_without_extension)
    train_or_test = os.path.basename(opt.dataset_path)
    fig.savefig(f'{sne_save}_{train_or_test}.png')
    plt.close()


# Compute the centroid and radius for each application/class.
def get_centroid_and_radius(embs_pool, opt):

    print(f"Start calculating centroids and radii.")
    classes = [d for d in os.listdir(opt.dataset_path)
               if os.path.isdir(os.path.join(opt.dataset_path, d))]

    apps_radius = []
    apps_centroid = []
    for app in range(len(classes)):
        embs = np.array(embs_pool[app])
        # Compute centroid for each application/class
        centroid = embs.mean(axis=0)
        apps_centroid.append(centroid)
        # Compute MAD of cosine distance as radius
        radius = 1.0 - cosine_similarity(embs, centroid.reshape(1, -1))
        apps_radius.append(median_absolute_deviation(radius))

    apps_radius = np.array(apps_radius)
    apps_centroid = np.array(apps_centroid)

    return apps_radius, apps_centroid


# Adjust radii to reduce overlap between different applications/classes.
def adjust_centroid_and_radius(apps_radius, apps_centroid, opt):

    print(f"Start adjusting radii.")
    classes = [d for d in os.listdir(opt.dataset_path)
               if os.path.isdir(os.path.join(opt.dataset_path, d))]

    for app1 in range(len(classes)):
        for app2 in range(app1 + 1, len(classes)):
            # Cosine distance between two class/application centroids
            centroid_1 = apps_centroid[app1]
            centroid_2 = apps_centroid[app2]
            distance = 1.0 - cosine_similarity(centroid_1.reshape(1, -1), centroid_2.reshape(1, -1))[0, 0]

            radius_1 = apps_radius[app1]
            radius_2 = apps_radius[app2]

            # If overlap, shrink radii proportionally
            if distance <= radius_1 + radius_2:
                print(f"{app1} vs {app2}: distance = {distance}, r1 = {radius_1}, r2 = {radius_2}")
                diff = radius_1 + radius_2 - distance
                apps_radius[app1] -= 1.0 * diff * radius_1 / (radius_1 + radius_2)
                apps_radius[app2] -= 1.0 * diff * radius_2 / (radius_1 + radius_2)

    return apps_radius


# Save centroids and adjusted radii as spatial distribution file.
def save_spatial_distribution(embs_pool, opt):
    print(f"Save spatial distribution.")

    apps_radius, apps_centroid = get_centroid_and_radius(embs_pool, opt)
    apps_radius = adjust_centroid_and_radius(apps_radius, apps_centroid, opt)

    # Save npz file
    checkpoint = torch.load(opt.model_path)
    opt_train = checkpoint['opt']
    spatial_distribution_path = os.path.join(opt.spatial_distribution_path, opt_train.model_name)
    os.makedirs(spatial_distribution_path, exist_ok=True)

    spatial_distribution_name = os.path.basename(opt.model_path)
    spatial_distribution_without_extension = os.path.splitext(spatial_distribution_name)[0]

    spatial_distribution_save = os.path.join(spatial_distribution_path, spatial_distribution_without_extension)
    train_or_test = os.path.basename(opt.dataset_path)
    np.savez(f"{spatial_distribution_save}_{train_or_test}.npz", centroid=apps_centroid, radius=apps_radius)
    print(f"File spatial_distribution generate completed.")


def main():

    opt = parse_option()

    test_loader = set_loader(opt)

    model = load_model(opt)

    embs_pool = get_embeddings(test_loader, model, opt)

    plot_tsne(embs_pool, opt)

    save_spatial_distribution(embs_pool, opt)


if __name__ == '__main__':
    main()