# 🧬 TCGA Whole Slide Image (WSI) Classification

This project was conducted as part of the course **DSAA 5037** at **HKUST (Guangzhou)**.  
We explore deep learning and graph-based techniques for classifying Whole Slide Images (WSIs) from The Cancer Genome Atlas (TCGA), with a focus on cancer **subtype** and **tumor site** classification.

## 📌 Overview

Whole Slide Images (WSIs) are gigapixel-scale histopathology slides capturing rich spatial and morphological details. However, their ultra-high resolution, sparse labels, and heterogeneity make automated analysis a major challenge.  
This project investigates weakly supervised learning approaches and graph-based models for WSI classification. Specifically, we evaluate the performance of Attention-based MIL (ABMIL), Graph Convolutional Networks (GCN), and a proposed graph walk-based pooling method.

## 🧪 Methods

The end-to-end pipeline includes:

- 🔹 **Patch Extraction**: Tiling WSIs into manageable patches.
- 🔹 **Feature Encoding**: Using pretrained CNNs (e.g., ResNet) or vision transformers (e.g., ViT, DINO, UNI) to extract patch embeddings.
- 🔹 **Graph Construction**: Modeling spatial relationships between patches using k-NN or positional adjacency.
- 🔹 **WSI Classification**: Aggregating patch features via:
  - Attention-based Multiple Instance Learning (ABMIL)
  - Graph Convolutional Networks (GCN)
  - Graph Walk-based Weighted Pooling (proposed)

> All models operate under weak supervision, using only slide-level labels.

## 🗂️ Dataset

- **Source**: [TCGA - The Cancer Genome Atlas](https://portal.gdc.cancer.gov/)
- **Data Type**: Whole Slide Images (WSIs) from various cancer cohorts
- **Tasks**:
  - **Subtype classification** (e.g., BRCA-Basal vs. Luminal A/B)
  - **Site classification** (e.g., distinguishing organ sites)
- **Preprocessing**:
  - Tissue detection and tiling
  - Optional stain normalization
  - Feature extraction using frozen encoders

## 📊 Results

We evaluate each model using standard classification metrics:

- ✅ Accuracy  

> Comparative results are reported across four backbones:  
> `ResNet`, `ViT`, `DINO`, and `UNI`.  
> See:  
> `Table~\ref{exp:exp_subtype_resnet}`  
> `Table~\ref{exp:exp_subtype_vit}`  
> `Table~\ref{exp:exp_subtype_dino}`  
> `Table~\ref{exp:exp_subtype_uni}`

Key Findings:
- ABMIL achieves strong and consistent performance across most cancer subtypes.
- GCN performs competitively in certain scenarios and outperforms ABMIL under specific encoders.
- The graph walk-based method did not consistently improve results, suggesting room for further optimization.

## 🔮 Future Directions

- 🔍 **Smarter Patch Selection**: Incorporate attention or self-supervision to focus on diagnostically relevant tissue regions.
- 🧠 **Dynamic Graph Construction**: Move beyond grid-based tessellation to learn more adaptive and semantically meaningful graph topologies.
- 🧪 **Better Integration of Context**: Explore hierarchical models, spatial transformers, or hybrid attention-graph frameworks.

## 📁 Project Structure

```bash
.
├── data/               # Preprocessed tiles, patch features, or graph representations
├── models/             # Model implementations: ABMIL, GCN, etc.
├── utils/              # Data loaders, graph builders, evaluation metrics
├── train.py            # Main training and evaluation script
├── config.yaml         # YAML config file for hyperparameters and model selection
└── README.md           # This file