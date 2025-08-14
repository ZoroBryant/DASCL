import math
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score


# Create two augmented crops from the same image.
class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        # return two views for supervised contrastive training
        return [self.transform(x), self.transform(x)]


# Track current value, running sum, count, and average.
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        # reset running stats
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # update with a new value
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Step or cosine LR schedule per epoch.
def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Per-iteration linear warmup to target LR.
def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        # progress in [0, 1]
        p = (batch_id + (epoch - 1) * total_batches) / (args.warm_epochs * total_batches)
        p = max(0.0, min(1.0, p))  # clamp to avoid overshoot
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


# Build SGD optimizer from parsed options.
def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


# Save checkpoint containing opt, model, optimizer, and epoch.
def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


# Run t-SNE on features and return a Matplotlib Figure.
def visualize_tsne(features, labels, n_components=2, perplexity=30, learning_rate='auto', n_iter=1000, random_state=42):

    # input checks
    features = np.asarray(features)
    labels = np.asarray(labels)
    if features.ndim != 2:
        raise ValueError("`features` must be 2D array [N, D].")
    if labels.shape[0] != features.shape[0]:
        raise ValueError("`labels` length must match number of rows in `features`.")

    # fit t-SNE
    tsne = TSNE(n_components=n_components,
                perplexity=perplexity,
                learning_rate = learning_rate,
                n_iter=n_iter,
                random_state=random_state)
    features_tsne = tsne.fit_transform(features)

    # build figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # color map without seaborn dependency
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))

    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(
            features_tsne[mask, 0],
            features_tsne[mask, 1],
            s=14,
            alpha=0.65,
            label=f'Class {lbl}',
            c=[cmap(idx)]
        )

    ax.set_title('t-SNE Visualization of Features')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.legend(loc='best', fontsize=9, frameon=False)
    fig.tight_layout()
    return fig


# Compute MAD (Median Absolute Deviation).
def median_absolute_deviation(data):
    median = np.median(data)
    deviations = np.abs(data - median)
    mad = np.median(deviations)
    return mad


# Compute requested metrics (macro and per-class variants supported).
def measurement(y_true, y_pred, eval_metrics):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results = {}
    for eval_metric in eval_metrics:
        if eval_metric == 'Accuracy':
            overall_accuracy = accuracy_score(y_true, y_pred) * 100.0
            # per-class accuracy: correct/total per class
            labels = np.unique(y_true)
            per_class_accuracy = []
            for label in labels:
                true_label_idx = (y_true == label)
                correct_preds = np.sum((y_pred == y_true)[true_label_idx])
                per_class_acc = correct_preds / np.sum(true_label_idx)
                per_class_accuracy.append(round(per_class_acc * 100.0, 2))
            results[eval_metric] = {
                'overall': round(overall_accuracy, 2),
                'per_class': per_class_accuracy
            }

        elif eval_metric == 'Precision':
            macro_precision = precision_score(y_true, y_pred, average="macro") * 100.0
            per_class_precision = precision_score(y_true, y_pred, average=None) * 100.0
            results[eval_metric] = {
                'macro': round(macro_precision, 2),
                'per_class': [round(p, 2) for p in per_class_precision]
            }

        elif eval_metric == 'Recall':
            macro_recall = recall_score(y_true, y_pred, average="macro") * 100.0
            per_class_recall = recall_score(y_true, y_pred, average=None) * 100.0
            results[eval_metric] = {
                'macro': round(macro_recall, 2),
                'per_class': [round(r, 2) for r in per_class_recall]
            }

        elif eval_metric == 'F1-score':
            macro_f1 = f1_score(y_true, y_pred, average="macro") * 100.0
            per_class_f1 = f1_score(y_true, y_pred, average=None) * 100.0
            results[eval_metric] = {
                'macro': round(macro_f1, 2),
                'per_class': [round(f, 2) for f in per_class_f1]
            }

        elif eval_metric == 'P@min':
            results[eval_metric] = round(np.min(precision_score(y_true, y_pred, average=None)) * 100.0, 2)

        else:
            raise ValueError(f"Metric {eval_metric} is not matched.")

    return results


# Plot a binned PR curve over quantile thresholds and return (precision, recall, thresholds).
def plot_pr_curve_binned(y_true, y_score, n_thresholds=50, title='PR Curve', show_ap=True):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # thresholds from quantiles
    quantiles = np.linspace(0, 1, n_thresholds)
    thresholds = pd.Series(y_score).quantile(quantiles).unique()

    precision_list, recall_list = [], []
    # sweep thresholds
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)

        precision_list.append(precision)
        recall_list.append(recall)

    precision_array = np.array(precision_list)
    recall_array = np.array(recall_list)

    # focus on a recall interval
    mask = (recall_array >= 0.40) & (recall_array <= 0.98)
    filtered_precision = precision_array[mask]
    filtered_recall = recall_array[mask]
    filtered_thresholds = thresholds[mask]

    # plot
    plt.figure(figsize=(6, 4))
    plt.plot(filtered_recall, filtered_precision, marker='o', label='Filtered PR curve')

    if show_ap:
        ap_score = average_precision_score(y_true, y_score)
        plt.title(f'{title} (AP = {ap_score:.2f})')
    else:
        plt.title(title)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return filtered_precision.tolist(), filtered_recall.tolist(), filtered_thresholds.tolist()
