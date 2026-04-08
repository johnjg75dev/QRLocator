import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pathlib import Path
import pathlib  # Added
import platform # Added

# FIX FOR WINDOWS:
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

from config import CFG
from model import QRViTDet
from dataset import build_dataloaders

def run_visual_eval(num_samples=4, checkpoint_name="best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running visual evaluation on {device}...")

    # 1. Load Model
    model = QRViTDet(CFG).to(device)
    checkpoint_path = f"{CFG.checkpoint_dir}/{checkpoint_name}"
    
    if not Path(checkpoint_path).exists():
        print(f"Error: {checkpoint_path} not found!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (F1: {checkpoint['metrics']['f1']:.4f})")

    # 2. Get Test Data
    _, _, test_dl = build_dataloaders(CFG)
    
    # 3. Setup Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    samples_found = 0
    with torch.no_grad():
        for i, (images, gt_boxes_list, _) in enumerate(test_dl):
            if samples_found >= num_samples:
                break
            
            # Forward Pass
            images = images.to(device)
            outputs = model(images)
            
            # Process results for the first image in batch
            img_np = (images[0, 0].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np).convert("RGB")
            draw = ImageDraw.Draw(pil_img)
            
            W, H = CFG.img_w, CFG.img_h
            
            # Draw Ground Truth (RED)
            for box in gt_boxes_list[0]:
                x1, y1, x2, y2 = box.tolist()
                draw.rectangle([x1*W, y1*H, x2*W, y2*H], outline="red", width=2)

            # Process Predictions (GREEN)
            pred_logits = outputs["pred_logits"][0] # (Q, 2)
            pred_boxes = outputs["pred_boxes"][0]   # (Q, 4)
            
            # Get probabilities for "QR Code" class (index 0)
            probs = pred_logits.softmax(-1)[:, 0]
            
            # Filter by threshold
            keep = probs > CFG.conf_threshold
            top_probs = probs[keep]
            top_boxes = pred_boxes[keep]

            for box, conf in zip(top_boxes, top_probs):
                x1, y1, x2, y2 = box.cpu().numpy()
                draw.rectangle([x1*W, y1*H, x2*W, y2*H], outline="#00FF00", width=2)
                draw.text((x1*W, y1*H - 10), f"{conf:.2f}", fill="#00FF00")

            # Plot
            axes[samples_found].imshow(pil_img)
            axes[samples_found].set_title(f"Sample {samples_found+1} (GT=Red, Pred=Green)")
            axes[samples_found].axis("off")
            
            samples_found += 1

    plt.tight_layout()
    plt.savefig("visual_results.png")
    print("Results saved to visual_results.png")
    plt.show()

if __name__ == "__main__":
    run_visual_eval()