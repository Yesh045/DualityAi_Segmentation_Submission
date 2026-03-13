# Offroad Autonomy Segmentation - Team Submission

## 1. Project Overview
This repository contains our official submission for the Duality AI Offroad Autonomy Segmentation challenge. Our objective was to develop a highly robust semantic segmentation pipeline capable of parsing 10 distinct terrain classes in synthetic off-road environments. This scene understanding is critical for Unmanned Ground Vehicle (UGV) path planning and obstacle avoidance. 

## 2. Directory Structure
Our repository is cleanly organized into scripts, models, and empirical results to ensure full reproducibility:

```text
DUALITYAI_SEGMENTATION_SUBMISSION/
│
├── README.md                              <- Project documentation and report
├── requirements.txt                       <- Python dependencies
│
├── models/                                <- Training scripts and data loaders
│   ├── train_unetpp.py                    (UNet++ with tu-resnet34 / resnet18)
│   ├── train_deeplabv3.py                 (DeepLabV3+ with ResNet50)
│   ├── train_segformer.py                 (SegFormer mit-b2)
│   ├── dataset_loader.py                  (Albumentations augmentation pipeline)
│   └── All the trained models(weights) Drive link.txt  <- Link to download .pth files
│
├── evaluation_scripts/                    <- Scripts for inference and metric generation
│   ├── evaluate_unetpp.py
│   ├── evaluate_deeplab.py
│   ├── evaluate_segformer.py
│   ├── ensemble_evaluate.py               (Soft-voting multi-model ensemble)
│   └── generate_test_masks.py             (Creates final colorized test submission)
│
└── evaluation_results/                    <- Empirical proof of work and visualizations
    ├── unetpp_evaluation_results/         (Winning model metrics and test masks)
    ├── deeplab_evaluation_results/
    ├── segformer_evaluation_results/
    └── ensemble_evaluation_results/
    