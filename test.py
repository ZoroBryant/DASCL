import argparse
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score

import torch
import torch.utils.data as data
from torchvision import transforms, datasets

from networks.MTC import SupConResNet
from utils.tools import measurement, plot_pr_curve_binned


# Parse command-line options.
def parse_option():
    parser = argparse.ArgumentParser('Argument for testing')

    parser.add_argument('--dataset_path', type=str, default='./datasets/test', help='path to test dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=16, help='#workers for dataloader')
    parser.add_argument('--encoder', type=str, default='resnet18', help='encoder')
    parser.add_argument('--model_path', type=str, help='path to checkpoint')

    parser.add_argument('--spatial_distribution_path', type=str, help='path to spatial_distribution npz')
    parser.add_argument('--scenario', type=str, default='closed-world',
                        choices=['closed-world','open-world'], help='evaluation scenario')

    opt = parser.parse_args()

    return opt


# Build test dataloader.
def set_test_loader(opt):

    mean, std = ((0.9920479655265808, 0.9398094415664673, 0.9398094415664673),
                 (0.08487337082624435, 0.21652734279632568, 0.21652734279632568))

    test_transform = transforms.Compose([
        transforms.Resize(size=(128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_dataset = datasets.ImageFolder(root=opt.dataset_path, transform=test_transform)
    print(test_dataset.class_to_idx)

    test_loader = data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False,
                                  num_workers=opt.num_workers, pin_memory=True)

    return test_loader


def load_model(opt):

    print(f"Start loading model.")

    model = SupConResNet(encoder=opt.encoder)
    checkpoint = torch.load(opt.model_path)
    state_dict = checkpoint['model']

    # Load model weights with proper handling of DataParallel checkpoints.
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)

            if any('module' in key for key in state_dict.keys()):
                model.load_state_dict(state_dict)
            else:
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


def load_spatial_distribution(opt):

    print(f"Start loading spatial_distribution.")

    spatial_data = np.load(opt.spatial_distribution_path)
    apps_radius = spatial_data['radius']
    apps_centroid = spatial_data['centroid']

    return apps_radius, apps_centroid



def model_evaluation(model, test_loader, apps_radius, apps_centroid, opt):

    print(f"Start evaluating model.")

    classes = [d for d in os.listdir(opt.dataset_path)
               if os.path.isdir(os.path.join(opt.dataset_path, d))]
    unknown_idx = len(classes) - 1  # assume the last index is 'unknown' in open-world

    with torch.no_grad():
        model.eval()
        y_pred = []
        y_true = []
        y_score = []
        for index, (images, labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = images.cuda()
            embs = model(images).cpu().numpy()
            labels = labels.cpu().numpy()

            # cosine distance to application/class centroids
            all_sims = 1 - cosine_similarity(embs, apps_centroid)  # shape: [batch, C]

            # score for PR: smaller is closer; use min distance to any centroid
            prc_score = np.min(all_sims, axis=1)
            y_score.append(prc_score)

            # decision: nearest centroid after subtracting application/class radius
            all_sims -= apps_radius
            outs = np.argmin(all_sims, axis=1)

            if opt.scenario == 'open-world':
                # if min adjusted distance still > margin, mark as unknown
                outs_d = np.min(all_sims, axis=1)
                open_indices = np.where(outs_d > 1e-2)[0]
                outs[open_indices] = unknown_idx

            y_pred.append(outs)
            y_true.append(labels)

        # Flatten all prediction/label/score lists into 1-D arrays.
        y_pred = np.concatenate(y_pred).flatten()
        y_true = np.concatenate(y_true).flatten()
        y_score = np.concatenate(y_score).flatten()

    # Select evaluation metrics according to scenario.
    if opt.scenario == 'open-world':
        eval_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    elif opt.scenario == 'closed-world':
        eval_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'P@min']
    else:
        raise ValueError(f"scenario {opt.scenario} is not matched.")

    # Compute and print classification metrics
    result = measurement(y_true, y_pred, eval_metrics)
    print(result)

    # Additional evaluation for open-world scenario.
    y_true_open = [1 if v == 16 else 0 for v in y_true]  # 1=unknown, 0=known
    # Average Precision for unknown detection.
    ap_open = average_precision_score(y_true_open, y_score, pos_label=1)
    print(ap_open)

    # Plot PR curve (binned) for open-set detection and return PR vectors.
    precision_open, recall_open, thresholds_open = (
        plot_pr_curve_binned(y_true_open, y_score, n_thresholds=20, title='PR Curve', show_ap=True))

    # Print PR vectors.
    print("Recall:", recall_open)
    print("Precision:", precision_open)


def main():

    opt = parse_option()

    test_loader = set_test_loader(opt)

    model = load_model(opt)

    apps_radius, apps_centroid = load_spatial_distribution(opt)

    model_evaluation(model, test_loader, apps_radius, apps_centroid, opt)


if __name__ == '__main__':
    main()