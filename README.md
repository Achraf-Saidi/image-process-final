# Semantic Image Segmentation — Supervised & Self-Supervised Learning

Computer-vision project for **semantic segmentation of tiled imagery** into five classes:

1. others
2. water
3. buildings
4. farmlands
5. green spaces

The repository contains both a supervised segmentation pipeline and a self-supervised learning workflow based on representation learning and clustering.

## Highlights

### Supervised segmentation
- U-Net-style segmentation architecture.
- Configurable experiments through YAML files.
- Data augmentation and tiled-image processing.
- Spatial train/validation/test splitting.
- Evaluation with pixel accuracy, mean IoU, mean Dice and per-class metrics.
- Confusion matrices and visual overlays.

### Self-supervised learning
- Autoencoder-based representation learning.
- Latent-feature extraction.
- Clustering-based semantic grouping.
- Cluster-to-class matching for evaluation.
- Re-clustering of saved embeddings/checkpoints without retraining.
- MiniBatchKMeans evaluation workflow.

## Repository structure

```text
.
├── Dataset/
│   ├── images/
│   ├── annotations/
│   ├── dataLoader.py
│   └── makeGraph.py
├── Networks/
│   ├── Architectures/
│   ├── model.py
│   └── ssl_model.py
├── Todo_List/                  # YAML experiment configurations
├── Results/                    # Experiment outputs
├── main.py                     # Supervised training/evaluation
├── main_ssl.py                 # Self-supervised training
├── ssl_recluster_eval.py       # Re-clustering/evaluation utility
├── utils.py
└── requirements.txt
```

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Running an experiment

Supervised pipeline:

```bash
python main.py -exp UNet_Baseline
```

Self-supervised pipeline:

```bash
python main_ssl.py -exp BestConfigB
```

Re-cluster an existing learned representation:

```bash
python ssl_recluster_eval.py --exp BestConfigB --k 10
```

Experiment settings are stored in `Todo_List/*.yaml`.

## Evaluation

The project reports segmentation metrics including:

- pixel accuracy;
- mean Intersection over Union (mIoU);
- mean Dice score;
- class-wise IoU / Dice;
- confusion matrices.

## Author

**Achraf Saidi**  
Data Science & Machine Learning — UCLouvain
