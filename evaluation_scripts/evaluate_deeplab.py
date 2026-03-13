import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

from dataset_loader import DesertSegmentationDataset, get_validation_augmentation

# Set matplotlib to non-interactive
plt.switch_backend('Agg')

# --- CONFIGURATION ---
IMAGE_SIZE = 512
BATCH_SIZE = 4
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_deeplabv3_model.pth"

class_names = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
               'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']

def fast_hist(a, b, n):
    """Calculates confusion matrix efficiently."""
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def plot_confusion_matrix(cm, output_path):
    """Generates a beautiful heatmap for the confusion matrix."""
    plt.figure(figsize=(12, 10))
    # Normalize by row (Ground Truth) to show recall percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized) # handle division by zero

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix (Recall)')
    plt.ylabel('Ground Truth (Actual)')
    plt.xlabel('Prediction')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_bar_chart(metrics, metric_name, output_path):
    """Generates a bar chart for per-class metrics."""
    plt.figure(figsize=(12, 6))
    
    # Filter out NaNs for plotting
    valid_indices = [i for i, m in enumerate(metrics) if not np.isnan(m)]
    valid_names = [class_names[i] for i in valid_indices]
    valid_metrics = [metrics[i] for i in valid_indices]

    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_metrics)))
    bars = plt.bar(valid_names, valid_metrics, color=colors, edgecolor='black')
    
    plt.title(f'Per-Class {metric_name}')
    plt.ylabel(metric_name)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    print(f"Hardware: {DEVICE} | Evaluating DeepLabV3+ on Validation Set")
    
    output_dir = 'deeplab_evaluation_results'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Validation Data (Ground Truth is required for metrics)
    val_ds = DesertSegmentationDataset("dataset/val/Color_Images", "dataset/val/Segmentation", get_validation_augmentation(IMAGE_SIZE))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # 2. Load DeepLabV3+ Model
    print("Loading saved model weights...")
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50", 
        encoder_weights=None, # Weights are loaded from our saved file
        in_channels=3, 
        classes=NUM_CLASSES
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Initialize Confusion Matrix
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))

    # 3. Run Inference and Accumulate Matrix
    print("Running evaluation...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            logits = model(images)
            # Ensure spatial dimensions match exactly just in case
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            
            # Update confusion matrix
            hist += fast_hist(targets.flatten(), preds.flatten(), NUM_CLASSES)

    # 4. Calculate Metrics from Confusion Matrix
    # IoU = TP / (TP + FP + FN)
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    
    # Precision = TP / (TP + FP)
    precision = np.diag(hist) / hist.sum(axis=0)
    
    # Recall = TP / (TP + FN)
    recall = np.diag(hist) / hist.sum(axis=1)
    
    # Pixel Accuracy = Total TP / Total Pixels
    pixel_acc = np.diag(hist).sum() / hist.sum()

    mIoU = np.nanmean(iou)
    mPrecision = np.nanmean(precision)
    mRecall = np.nanmean(recall)

    # 5. Save Results and Charts
    print("\nGenerating charts and matrices...")
    plot_confusion_matrix(hist, os.path.join(output_dir, 'confusion_matrix.png'))
    plot_bar_chart(iou, 'IoU Score', os.path.join(output_dir, 'per_class_iou.png'))
    plot_bar_chart(precision, 'Precision', os.path.join(output_dir, 'per_class_precision.png'))
    plot_bar_chart(recall, 'Recall', os.path.join(output_dir, 'per_class_recall.png'))

    # Save Text Report
    report_path = os.path.join(output_dir, 'final_metrics_report.txt')
    with open(report_path, 'w') as f:
        f.write("DEEPLABV3+ FINAL EVALUATION METRICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Mean IoU (mIoU):       {mIoU:.4f}\n")
        f.write(f"Mean Precision (mAP):  {mPrecision:.4f}  <-- (Approximation of mAP for Semantic Segmentation)\n")
        f.write(f"Mean Recall:           {mRecall:.4f}\n")
        f.write(f"Global Pixel Accuracy: {pixel_acc:.4f}\n\n")
        
        f.write("PER-CLASS METRICS:\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'Class Name':<18} | {'IoU':<10} | {'Precision':<10} | {'Recall':<10}\n")
        f.write("-" * 75 + "\n")
        
        for i, name in enumerate(class_names):
            c_iou = iou[i] if not np.isnan(iou[i]) else 0.0
            c_prec = precision[i] if not np.isnan(precision[i]) else 0.0
            c_rec = recall[i] if not np.isnan(recall[i]) else 0.0
            f.write(f"{name:<18} | {c_iou:<10.4f} | {c_prec:<10.4f} | {c_rec:<10.4f}\n")

    print(f"\nSuccess! All matrices, charts, and reports have been saved to the '{output_dir}' folder.")

if __name__ == "__main__":
    main()