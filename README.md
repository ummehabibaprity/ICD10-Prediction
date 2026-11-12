# ICD-10 Prediction

This repository contains the Python scripts and Jupyter notebooks associated with the research paper **“Beyond Model Complexity: Rethinking Text Selection and Imbalance-Aware Learning for ICD-10 Prediction.”**

## Overview

This study investigates automated multilabel ICD-10 prediction from clinical discharge summaries, focusing on three aspects:

1. **Transformer Architecture Comparison:** Evaluating DistilBERT, BERT, BioBERT, and ClinicalBERT in terms of predictive performance and computational efficiency.
2. **Clinical Text Selection:** Comparing standard truncation, head/tail truncation, and diagnosis-guided text selection using DistilBERT.
3. **Imbalance-Aware Optimization:** Evaluating binary cross-entropy, focal loss, weighted sampling, and their combination using DistilBERT.

The experiments focus on the 50 most frequent ICD-10 codes and use patient-level data partitioning.

## Datasets

The study uses the following datasets:

- **MIMIC-IV (v3.1)**
- **MIMIC-IV-Note (v2.2)**

Both datasets are available through [PhysioNet](https://physionet.org/) and require appropriate credentialed access.

**Clinical datasets and patient records are not included in this repository.**

## Repository Contents

The repository contains Python scripts (`.py`) and Jupyter notebooks (`.ipynb`) used for data preparation, model training, clinical text selection, optimization experiments, and evaluation.

## Requirements

The experiments were implemented in Python using libraries including:

- PyTorch
- Hugging Face Transformers
- NumPy
- pandas
- scikit-learn
- LIME
- SHAP

A CUDA-compatible GPU is recommended for model training.

## Reproducibility

The experiments use a fixed random seed of 42 and patient-level training, validation, and testing partitions.

To reproduce the experiments, researchers should obtain authorized access to the MIMIC datasets, configure the required local data paths, install the necessary Python dependencies, and execute the relevant scripts or notebooks.

The manuscript provides additional details about preprocessing, hyperparameters, evaluation metrics, and experimental configurations.
