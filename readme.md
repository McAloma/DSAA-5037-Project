# TCGA Whole Slide Image (WSI) Classification

This project is part of the course **DSAA 5037 - Advanced Topics in Artificial Intelligence** at **HKUST (Guangzhou)**.  
We explore deep learning techniques for classifying Whole Slide Images (WSIs) from The Cancer Genome Atlas (TCGA).

## 📌 Project Overview

The goal of this project is to develop and evaluate models for the classification of histopathological WSIs.  
We use TCGA datasets and focus on identifying cancer subtypes or tumor sites using computational pathology approaches.

## 🧪 Methods

We apply a pipeline that includes:

- **Patch extraction** from WSIs
- **Feature encoding** using pre-trained CNNs or vision transformers
- **Graph-based modeling** (e.g., GCN, GAT) to capture spatial and contextual relationships
- **Classification** of WSIs based on aggregated patch-level or node-level information

## 🗂️ Dataset

- **Source**: TCGA (The Cancer Genome Atlas)
- **Content**: Whole Slide Images (WSIs) for various cancer types
- **Labels**: Tumor site or subtype
- **Preprocessing**: Tiling, filtering, stain normalization (if applied)

> Note: Due to the size of WSIs, we recommend preprocessing and feature extraction before training.

## 📊 Results

Performance is evaluated using standard classification metrics:

- Accuracy
- F1-score
- ROC-AUC (optional)
- Confusion Matrix

We report performance across different model variants and ablation studies.

## 📁 Project Structure

```bash
.
├── data/               # Processed WSI tiles or graph representations
├── models/             # Model definitions (e.g., GCN, GAT)
├── utils/              # Helper functions (e.g., graph building, evaluation)
├── train.py            # Main training script
├── config.yaml         # Training configuration
└── README.md           # This file