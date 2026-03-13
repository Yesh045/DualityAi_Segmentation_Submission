import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
import segmentation_models_pytorch as smp
from tqdm import tqdm
import time
import numpy as np

from dataset_loader import DesertSegmentationDataset, get_training_augmentation, get_validation_augmentation

# --- OPTIMIZED HYPERPARAMETERS ---
IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 6e-5            # Best for SegFormer (Transformers prefer slightly lower LR)
WEIGHT_DECAY = 0.01             # Helps prevent overfitting
PATIENCE = 7                    # Early stopping patience
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_iou(preds, labels, num_classes):
    """Calculates Intersection over Union (IoU) per class and averages it."""
    preds = torch.argmax(preds, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union > 0:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.mean(ious) if ious else 0.0

def main():
    print(f"Hardware: {DEVICE}")

    # 1. Load Data (UPDATED PATHS TO MATCH YOUR FOLDERS)
    train_ds = DesertSegmentationDataset("dataset/train/Color_Images", "dataset/train/Segmentation", get_training_augmentation(IMAGE_SIZE))
    val_ds = DesertSegmentationDataset("dataset/val/Color_Images", "dataset/val/Segmentation", get_validation_augmentation(IMAGE_SIZE))
    
    # Note: If your test folder doesn't have a 'Segmentation' folder yet, this line will throw an error during the final step.
    test_ds = DesertSegmentationDataset("dataset/test/Color_Images", "dataset/test/Segmentation", get_validation_augmentation(IMAGE_SIZE))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False) # Batch size 1 for inference speed test

    # 2. Initialize Model
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2", num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
    ).to(DEVICE)

    # 3. Best Optimizer, Loss, and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    ce_loss = torch.nn.CrossEntropyLoss()
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    
    # Cosine Annealing reduces LR smoothly following a cosine curve
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 4. Training Loop with Early Stopping
    best_val_iou = 0.0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss, train_iou = 0.0, 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, masks in loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=images)
            logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            
            loss = ce_loss(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_iou += calculate_iou(logits, masks, NUM_CLASSES)
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        # --- VALIDATION ---
        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for images, masks in val_loop:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(pixel_values=images)
                logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                
                loss = ce_loss(logits, masks) + dice_loss(logits, masks)
                val_loss += loss.item()
                val_iou += calculate_iou(logits, masks, NUM_CLASSES)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        print(f"Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        # Early Stopping & Model Saving
        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_segformer_model.pth")
            print("-> Model saved (Best IoU improved!)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break

    # 5. Final Testing Loop (Unseen Data)
    print("\n--- Running Final Evaluation on Test Dataset ---")
    model.load_state_dict(torch.load("best_segformer_model.pth"))
    model.eval()
    test_iou = 0.0
    inference_times = []

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            start_time = time.perf_counter()
            outputs = model(pixel_values=images)
            end_time = time.perf_counter()
            
            inference_times.append((end_time - start_time) * 1000) # Convert to ms
            
            logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            test_iou += calculate_iou(logits, masks, NUM_CLASSES)

    avg_test_iou = test_iou / len(test_loader)
    avg_inf_speed = np.mean(inference_times)
    
    print("\n=== FINAL RESULTS ===")
    print(f"Test IoU Score: {avg_test_iou:.4f}")
    print(f"Average Inference Speed: {avg_inf_speed:.2f} ms per image")
    if avg_inf_speed < 50:
        print("Success: Inference speed is under the 50ms benchmark!")
    else:
        print("Warning: Inference speed exceeds the 50ms benchmark. Consider optimizing.")

if __name__ == "__main__":
    main()