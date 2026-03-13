import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
from dataset_loader import get_validation_augmentation

# --- CONFIGURATION ---
IMAGE_SIZE = 512
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_segformer_model.pth"
TEST_IMAGES_DIR = "dataset/test/Color_Images"
OUTPUT_DIR = "final_test_predictions_segformer"

color_palette = np.array([
    [0, 0, 0], [34, 139, 34], [0, 255, 0], [210, 180, 140], [139, 90, 43],
    [128, 128, 0], [139, 69, 19], [128, 128, 128], [160, 82, 45], [135, 206, 235],
], dtype=np.uint8)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading SegFormer model...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2", num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    test_transform = get_validation_augmentation(IMAGE_SIZE)
    test_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.png', '.jpg'))]
    
    print(f"Generating colorized masks for {len(test_images)} test images...")
    
    with torch.no_grad():
        for img_name in tqdm(test_images):
            img_path = os.path.join(TEST_IMAGES_DIR, img_name)
            
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = image.shape[:2]
            
            augmented = test_transform(image=image)
            tensor_img = augmented['image'].unsqueeze(0).to(DEVICE)
            
            # Segformer doesn't use AMP natively in this setup the same way, so normal inference
            outputs = model(pixel_values=tensor_img)
            logits = F.interpolate(outputs.logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            
            prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            # Colorize
            color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            for class_id in range(NUM_CLASSES):
                color_mask[prediction == class_id] = color_palette[class_id]
                
            out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_pred_color.png")
            cv2.imwrite(out_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

    print(f"Done! Your SegFormer test images are ready in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()