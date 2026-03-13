```markdown
# Offroad Autonomy Segmentation - Team [Your Team Name]

## 1. Project Overview
[cite_start]This repository contains our official submission for the Duality AI Offroad Autonomy Segmentation challenge[cite: 3]. [cite_start]Our objective was to develop a robust semantic segmentation pipeline capable of parsing 10 distinct terrain classes in synthetic desert environments for Unmanned Ground Vehicle (UGV) navigation[cite: 12, 16]. [cite_start]This scene understanding is critical for obstacle avoidance and path planning in dynamic landscapes[cite: 17].

## 2. Directory Structure
Our repository is organized for clear reproducibility on Windows systems:

```text
DUALITYAI_SEGMENTATION_SUBMISSION/
│
[cite_start]├── README.md                              <- Project documentation and report [cite: 182]
├── requirements.txt                       <- Python dependencies
[cite_start]├── Hackathon_Report.pdf                   <- Final technical report [cite: 174]
│
[cite_start]├── models/                                <- Training scripts and data loaders [cite: 171]
│   ├── train_unetpp.py                    (Winning Architecture)
│   ├── train_deeplabv3.py                 (Benchmark Architecture)
│   ├── train_segformer.py                 (Benchmark Architecture)
│   ├── dataset_loader.py                  (Augmentation pipeline)
│   └── All the trained models(weights) Drive link.txt
│
[cite_start]├── evaluation_scripts/                    <- Scripts for metric generation [cite: 173]
│   ├── evaluate_unetpp.py
│   ├── evaluate_deeplab.py
│   ├── evaluate_segformer.py
│   ├── ensemble_evaluate.py               (Multi-model soft-voting)
│   └── generate_test_masks.py             (Final submission generator)
│
[cite_start]└── evaluation_results/                    <- Empirical proof of work [cite: 55]
    ├── unetpp_evaluation_results/         (Confusion Matrix & Per-Class IoU)
    ├── deeplab_evaluation_results/
    ├── segformer_evaluation_results/
    [cite_start]└── final_test_predictions/            (Colorized submission masks) [cite: 127]

```

## 3. Getting Started (Windows)

### A. Environment Setup

We utilized a custom environment to manage dependencies such as `segmentation-models-pytorch` and `transformers`.

1. Open **Anaconda Prompt** or **PowerShell**.


2. Create and activate the environment:

```powershell
conda create -n duality_env python=3.9 -y
conda activate duality_env

```

3. Install required libraries:

```powershell
pip install -r requirements.txt

```

### B. Loading Model Weights

Due to GitHub's file size limits, our trained `.pth` weights are hosted on Google Drive.

1. Open `models/All the trained models(weights) Drive link.txt`.
2. Download `best_unetpp_model.pth`.
3. Place the file in the root directory.

### C. Evaluation & Inference

To verify our validation metrics and generate the final submission:

```powershell
# [cite_start]1. Generate mIoU, Precision, and Recall Reports [cite: 129]
python evaluation_scripts/evaluate_unetpp.py

# [cite_start]2. Generate colorized test masks (< 50ms per image) [cite: 127, 257]
python evaluation_scripts/generate_test_masks.py

```

## 4. Methodology & Results

We benchmarked three distinct architectures—**UNet++**, **DeepLabV3+**, and **SegFormer**—to optimize for accuracy and generalizability.

* **Winning Model:** **UNet++ (tu-resnet34)**
* 
**Mean IoU (mIoU):** **0.7253** 


* 
**Global Pixel Accuracy:** **90.24%** 


* 
**Inference Speed:** **< 50ms** 



### Key Optimizations:

* 
**Automatic Mixed Precision (AMP):** Implemented to reduce VRAM usage and accelerate inference on Windows hardware.


* 
**Hybrid Loss:** Utilized a combination of **Cross-Entropy** and **Multiclass Dice Loss** to address class imbalance for obstacles like "Logs" and "Rocks".



## 5. Failure Case Analysis

Evaluation revealed boundary confusion between "Dry Grass" and "Ground Clutter". However, our use of Dice Loss significantly improved the recall of minority classes compared to standard baseline approaches.

## 6. Submission Team

* **Team Name:** Team Anant
* 
**Track:** Segmentation Track 



```

**Final Steps for Submission:**
1. Save this content as `README.md` in your main folder.
2. Ensure your team name is filled in.
3. [cite_start]Push everything to your private GitHub repository and add the reviewers as collaborators[cite: 196, 200].

Would you like me to draft a LinkedIn post for you to celebrate finishing the hackathon?

```