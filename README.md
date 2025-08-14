# DCI-based Application identification via SCL (DASCL)

DASCL is a Supervised Contrastive Learning(SCL) framework for application identification driven by Downlink Control Information(DCI), which is released solely for research purposes.

## Overview

DASCL identifies mobile applications based on their scheduling behavior in real-world cellular networks. The complete workflow—from raw DCI to recognition accuracy and precision—is as follows:
1. **Preprocessing of DCI Traffic**. Use sliding time windows to extract the Transport Block Size (TBS) values from raw DCI traces and generate TBS traffic graphs (`dci_traffic_preprocessing.py`).
2. **Model Training with Supervised Contrastive Learning** . Train an encoder using the supervised contrastive learning paradigm, where the encoder backbone can be selected from ResNet, ViT, or ConvNeXt (`train.py`).
3. **Embedding & Spatial Distribution Modeling** . Apply the trained encoder to embed all TBS traffic graphs and compute the centroid and radius for each application in the latent space (`spatial_analysis.py`).
4. **Evaluation**. Evaluate the system in either closed-world or open-world scenarios to measure performance metrics such as accuracy and precision (`test.py`).

## ⚙️ Installation

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate supcon
```

## 🛠️ How to use

### Data Preparation

Run `dci_traffic_preprocessing.py` to process raw DCI traffic data (from `dci_data_csv/` ) into TBS traffic graph images and organize them under the `datasets/` directory with `train/`, `train_valid/`, and `valid/` splits.

### Training

We provide all experiment scripts for WF attacks in the folder `./scripts/`. For example, you can reproduce the AWF attack on the Undefended dataset by executing the following command.

Train the encoder using supervised contrastive loss:

```bash
python -u train.py \
  --print_freq 20 \
  --save_freq 20 \
  --batch_size 512 \
  --num_workers 16 \
  --epochs 100 \
  --learning_rate 0.05 \
  --lr_decay_epochs 40,60,80 \
  --lr_decay_rate 0.1 \
  --weight_decay 1e-4 \
  --momentum 0.9 \
  --encoder resnet18 \
  --dataset_path ./datasets/train \
  --trial 0
```

Checkpoints are saved under `save/model/`, and TensorBoard logs are stored in `save/tensorboard/`.  
For detailed meaning of each parameter, refer to the comments in `train.py`.

### Spatial Analysis

After training, compute application centroids and radii:

```bash
python spatial_analysis.py \
  --model_path ./save/models/<checkpoint>.pth \
  --batch_size 512 \
  --spatial_distribution_path ./save/spatial_distribution
```

This script also generates t‑SNE visualizations and stores the spatial distribution as `.npz` files under `save/spatial_distribution/` for later evaluation.

### Evaluation

Evaluate the recognition performance of DASCL:

```bash
python test.py \
  --model_path ./save/models/<checkpoint>.pth \
  --batch_size 512 \
  --spatial_distribution_path ./save/spatial_distribution/<file>.npz \
  --dataset_path ./datasets/test \
  --scenario closed-world
```

Use `--scenario open-world` to enable unknown-class detection.

## Acknowledgements

This codebase is a research prototype. If you find it useful, please cite the repository.