
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from config import CFG
from model import QRViTDet
from dataset import build_dataloaders

@torch.no_grad()
def run_interpretability(checkpoint_path="/content/QRLocator/checkpoints/best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model = QRViTDet(CFG).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Interpreting Model from Epoch {checkpoint['epoch']}")

    # 2. Get a real sample
    _, _, test_dl = build_dataloaders(CFG)
    images, gt_boxes, _ = next(iter(test_dl))
    images = images.to(device)

    # 3. Forward Pass
    out = model(images)
    pred_boxes = out["pred_boxes"][0].cpu()    # (Q, 4)
    pred_logits = out["pred_logits"][0].cpu()  # (Q, 2)
    probs = pred_logits.softmax(-1)[:, 0]      # Confidence of being a QR

    # 4. Create Visualization
    W, H = CFG.img_w, CFG.img_h
    img_pil = Image.fromarray((images[0,0].cpu().numpy()*255).astype(np.uint8)).convert("RGB")

    draw_a = ImageDraw.Draw(img_pil.copy())
    img_b = img_pil.copy()
    draw_b = ImageDraw.Draw(img_b)

    cmap = plt.get_cmap('hsv')

    print("\n--- Query Analysis ---")
    for i in range(CFG.num_queries):
        box = pred_boxes[i]
        conf = probs[i].item()

        x1, y1, x2, y2 = box[0]*W, box[1]*H, box[2]*W, box[3]*H
        color = tuple((np.array(cmap(i/CFG.num_queries)[:3]) * 255).astype(int))

        draw_b.rectangle([x1, y1, x2, y2], outline=color, width=1)
        draw_b.text((x1, y1), f"Q{i}", fill=color)

        if conf > CFG.conf_threshold:
            print(f"Query {i:02d} detected QR code with {conf:.2%} confidence")
            draw_a = ImageDraw.Draw(img_pil)
            draw_a.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            draw_a.text((x1+5, y1+5), f"ID:{i} {conf:.2f}", fill=(0, 255, 0))

    # 5. Spatial Bias Test
    blank = torch.full((1, 1, H, W), 0.5).to(device)
    blank_out = model(blank)
    blank_boxes = blank_out["pred_boxes"][0].cpu()

    img_dream = Image.new("RGB", (W, H), (128, 128, 128))
    draw_dream = ImageDraw.Draw(img_dream)
    for i in range(CFG.num_queries):
        box = blank_boxes[i]
        x1, y1, x2, y2 = box[0]*W, box[1]*H, box[2]*W, box[3]*H
        color = tuple((np.array(cmap(i/CFG.num_queries)[:3]) * 255).astype(int))
        draw_dream.rectangle([x1, y1, x2, y2], outline=color, width=1)

    # 6. Plotting
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    ax[0].imshow(img_pil)
    ax[0].set_title("Detection (Threshold Filtered)")

    ax[1].imshow(img_b)
    ax[1].set_title("All 16 Query Slots (Spatial Bias)")

    ax[2].imshow(img_dream)
    ax[2].set_title("The 'Dream' Test (Blank Image)")

    for a in ax: a.axis("off")
    plt.tight_layout()
    plt.savefig("interpretability_report.png")
    print("\nReport saved to interpretability_report.png")
    plt.show()

if __name__ == "__main__":
    run_interpretability()