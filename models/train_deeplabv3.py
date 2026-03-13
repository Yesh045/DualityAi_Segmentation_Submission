import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from dataset_loader import DesertSegmentationDataset, get_training_augmentation, get_validation_augmentation

# Set matplotlib to non-interactive
plt.switch_backend('Agg')

# ============================================================================
# 1. OPTIMIZED HYPERPARAMETERS
# ============================================================================
IMAGE_SIZE = 512
BATCH_SIZE = 4                  # Safe for 6GB VRAM (RTX 4050)
EPOCHS = 50
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
PATIENCE = 7                    # Early stopping patience
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 2. VISUALIZATION & LOGGING UTILITIES (From Company Script)
# ============================================================================
class_names = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
               'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']

color_palette = np.array([
    [0, 0, 0], [34, 139, 34], [0, 255, 0], [210, 180, 140], [139, 90, 43],
    [128, 128, 0], [139, 69, 19], [128, 128, 128], [160, 82, 45], [135, 206, 235],
], dtype=np.uint8)

def mask_to_color(mask):
    """Convert a 2D class mask to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask

def compute_iou(pred, target, num_classes=10, ignore_index=255):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index: continue
        pred_inds, target_inds = (pred == class_id), (target == class_id)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0: iou_per_class.append(float('nan'))
        else: iou_per_class.append((intersection / union).cpu().numpy())
    return np.nanmean(iou_per_class)

def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds, target_inds = (pred == class_id), (target == class_id)
        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)
        dice_per_class.append(dice_score.cpu().numpy())
    return np.mean(dice_per_class)

def compute_pixel_accuracy(pred, target):
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()

def save_training_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 10))
    
    metrics = [('Loss', 'train_loss', 'val_loss'), ('IoU', 'train_iou', 'val_iou'), 
               ('Dice Score', 'train_dice', 'val_dice'), ('Pixel Accuracy', 'train_pixel_acc', 'val_pixel_acc')]
    
    for i, (title, train_key, val_key) in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        plt.plot(history[train_key], label='train')
        plt.plot(history[val_key], label='val')
        plt.title(f'{title} vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'))
    plt.close()

def save_history_to_file(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("TRAINING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write("Best Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 50 + "\n\n")

# ============================================================================
# 3. MAIN PIPELINE (Train -> Validate -> Early Stop -> Final Test)
# ============================================================================
def main():
    print(f"Hardware: {DEVICE} | Model: DeepLabV3+ (ResNet50 Backbone)")
    output_dir = 'deeplab_train_stats'
    os.makedirs(output_dir, exist_ok=True)

    # A. LOAD DATASET (With full Augmentations)
    train_ds = DesertSegmentationDataset("dataset/train/Color_Images", "dataset/train/Segmentation", get_training_augmentation(IMAGE_SIZE))
    val_ds = DesertSegmentationDataset("dataset/val/Color_Images", "dataset/val/Segmentation", get_validation_augmentation(IMAGE_SIZE))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # B. INITIALIZE MODEL, OPTIMIZER (AdamW), SCHEDULER, LOSS
    model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    ce_loss = torch.nn.CrossEntropyLoss()
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': [], 
               'train_dice': [], 'val_dice': [], 'train_pixel_acc': [], 'val_pixel_acc': []}

    best_val_iou = 0.0
    epochs_no_improve = 0

    # C. TRAINING LOOP WITH VALIDATION
    for epoch in range(EPOCHS):
        # TRAIN
        model.train()
        t_loss, t_iou, t_dice, t_acc = [], [], [], []
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, masks in loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(images)
            
            # Best Loss Function (Dice + CE)
            loss = ce_loss(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            
            t_loss.append(loss.item())
            t_iou.append(compute_iou(logits, masks))
            t_dice.append(compute_dice(logits, masks))
            t_acc.append(compute_pixel_accuracy(logits, masks))
            loop.set_postfix(loss=np.mean(t_loss))

        scheduler.step()

        # VALIDATION
        model.eval()
        v_loss, v_iou, v_dice, v_acc = [], [], [], []
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for images, masks in val_loop:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                logits = model(images)
                
                loss = ce_loss(logits, masks) + dice_loss(logits, masks)
                v_loss.append(loss.item())
                v_iou.append(compute_iou(logits, masks))
                v_dice.append(compute_dice(logits, masks))
                v_acc.append(compute_pixel_accuracy(logits, masks))

        # Update History Logs
        history['train_loss'].append(np.mean(t_loss))
        history['val_loss'].append(np.mean(v_loss))
        history['train_iou'].append(np.mean(t_iou))
        history['val_iou'].append(np.mean(v_iou))
        history['train_dice'].append(np.mean(t_dice))
        history['val_dice'].append(np.mean(v_dice))
        history['train_pixel_acc'].append(np.mean(t_acc))
        history['val_pixel_acc'].append(np.mean(v_acc))

        print(f"Summary: Val Loss: {np.mean(v_loss):.4f} | Val IoU: {np.mean(v_iou):.4f} | Val Acc: {np.mean(v_acc):.4f}")

        # D. EARLY STOPPING & SAVE BEST MODEL
        if np.mean(v_iou) > best_val_iou:
            best_val_iou = np.mean(v_iou)
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_deeplabv3_model.pth")
            print("-> Model saved (Best IoU improved!)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
                
    # Save the company's plots and text logs
    print("\nSaving training curves and history text file...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir)

    # ============================================================================
    # 4. FINAL TESTING (Run inference on Unseen Test Dataset)
    # ============================================================================
    print("\n--- Running Final Evaluation on Test Dataset ---")
    
    # Load the best weights we just saved
    model.load_state_dict(torch.load("best_deeplabv3_model.pth"))
    model.eval()
    
    test_images_dir = "dataset/test/Color_Images"
    test_output_dir = "deeplab_test_predictions"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Get test augmentations (just resize/normalize, no random crops)
    test_transform = get_validation_augmentation(IMAGE_SIZE)
    
    if os.path.exists(test_images_dir):
        test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.png', '.jpg'))]
        print(f"Found {len(test_images)} test images. Generating colorized masks...")
        
        with torch.no_grad():
            for img_name in tqdm(test_images, desc="Predicting Test Set"):
                img_path = os.path.join(test_images_dir, img_name)
                
                # Load and preprocess image
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply validation transforms
                augmented = test_transform(image=image)
                tensor_img = augmented['image'].unsqueeze(0).to(DEVICE)
                
                # Predict
                logits = model(tensor_img)
                prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
                
                # Resize prediction back to original image size
                orig_h, orig_w = image.shape[:2]
                prediction_resized = cv2.resize(prediction.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                
                # Convert to color palette and save
                color_pred = mask_to_color(prediction_resized)
                out_path = os.path.join(test_output_dir, f"{os.path.splitext(img_name)[0]}_pred_color.png")
                cv2.imwrite(out_path, cv2.cvtColor(color_pred, cv2.COLOR_RGB2BGR))
                
        print(f"\nAll test predictions successfully colorized and saved to '{test_output_dir}/'")
    else:
        print(f"Test directory '{test_images_dir}' not found. Skipping final testing step.")

    print("\n=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()