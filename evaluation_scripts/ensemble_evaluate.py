import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_loader import DesertSegmentationDataset, get_validation_augmentation

# Set matplotlib to non-interactive
plt.switch_backend('Agg')

# --- CONFIGURATION ---
IMAGE_SIZE = 512
BATCH_SIZE = 2  # Keep batch size very small! 3 models in memory = heavy VRAM usage
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
               'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def plot_bar_chart(metrics, metric_name, output_path):
    plt.figure(figsize=(12, 6))
    valid_indices = [i for i, m in enumerate(metrics) if not np.isnan(m)]
    valid_names = [class_names[i] for i in valid_indices]
    valid_metrics = [metrics[i] for i in valid_indices]

    colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(valid_metrics)))
    bars = plt.bar(valid_names, valid_metrics, color=colors, edgecolor='black')
    
    plt.title(f'Per-Class {metric_name}')
    plt.ylabel(metric_name)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    print(f"Hardware: {DEVICE} | Initializing Soft-Voting Ensemble")
    output_dir = 'ensemble_evaluation_results'
    os.makedirs(output_dir, exist_ok=True)

    val_ds = DesertSegmentationDataset("dataset/val/Color_Images", "dataset/val/Segmentation", get_validation_augmentation(IMAGE_SIZE))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 1. LOAD ALL THREE MODELS
    print("Loading UNet++ (ResNet18)...")
    unet = smp.UnetPlusPlus(encoder_name="tu-resnet34", encoder_weights=None, in_channels=3, classes=NUM_CLASSES).to(DEVICE)
    unet.load_state_dict(torch.load("best_unetpp_model.pth", map_location=DEVICE))
    unet.eval()

    print("Loading DeepLabV3+ (ResNet50)...")
    deeplab = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=NUM_CLASSES).to(DEVICE)
    deeplab.load_state_dict(torch.load("best_deeplabv3_model.pth", map_location=DEVICE))
    deeplab.eval()

    print("Loading SegFormer (mit-b2)...")
    segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2", num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
    segformer = segformer.to(DEVICE)
    segformer.load_state_dict(torch.load("best_segformer_model.pth", map_location=DEVICE))
    segformer.eval()

    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))

    print("Running Ensemble Evaluation...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating Ensemble"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            with torch.cuda.amp.autocast():
                # Model 1: UNet++
                logits_unet = unet(images)
                prob_unet = F.softmax(F.interpolate(logits_unet, size=masks.shape[-2:], mode="bilinear", align_corners=False), dim=1)
                
                # Model 2: DeepLabV3+
                logits_deep = deeplab(images)
                prob_deep = F.softmax(F.interpolate(logits_deep, size=masks.shape[-2:], mode="bilinear", align_corners=False), dim=1)
                
                # Model 3: SegFormer
                outputs_seg = segformer(pixel_values=images)
                logits_seg = F.interpolate(outputs_seg.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                prob_seg = F.softmax(logits_seg, dim=1)
                
                # ENSEMBLE: Average the probabilities
                ensemble_prob = (prob_unet + prob_deep + prob_seg) / 3.0
            
            preds = torch.argmax(ensemble_prob, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            
            hist += fast_hist(targets.flatten(), preds.flatten(), NUM_CLASSES)

    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    precision = np.diag(hist) / hist.sum(axis=0)
    recall = np.diag(hist) / hist.sum(axis=1)
    pixel_acc = np.diag(hist).sum() / hist.sum()

    mIoU = np.nanmean(iou)
    mPrecision = np.nanmean(precision)
    mRecall = np.nanmean(recall)

    print("\nGenerating charts and matrices...")
    plot_bar_chart(iou, 'IoU Score', os.path.join(output_dir, 'ensemble_per_class_iou.png'))

    report_path = os.path.join(output_dir, 'ensemble_final_metrics.txt')
    with open(report_path, 'w') as f:
        f.write("ENSEMBLE (SOFT-VOTING) FINAL EVALUATION METRICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Mean IoU (mIoU):       {mIoU:.4f}\n")
        f.write(f"Mean Precision (mAP):  {mPrecision:.4f}\n")
        f.write(f"Mean Recall:           {mRecall:.4f}\n")
        f.write(f"Global Pixel Accuracy: {pixel_acc:.4f}\n\n")

    print(f"\nENSEMBLE FINISHED! mIoU: {mIoU:.4f}")

if __name__ == "__main__":
    main()