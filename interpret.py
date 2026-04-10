import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from config import CFG
from model import QRViTDet
from dataset import build_dataloaders

report = []


@torch.no_grad()
def run_interpretability(checkpoint_path="checkpoints/best.pt", num_samples=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model = QRViTDet(CFG).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Interpreting Model from Epoch {checkpoint['epoch']}")

    # 2. Get a real sample
    _, _, test_dl = build_dataloaders(CFG)
    itertest = iter(test_dl)
    for r in range(num_samples):
        images, gt_boxes, _ = next(itertest)
        images = images.to(device)

        # 3. Forward Pass
        out = model(images)
        pred_boxes = out["pred_boxes"][0].cpu()  # (Q, 4)
        pred_logits = out["pred_logits"][0].cpu()  # (Q, 2)
        probs = pred_logits.softmax(-1)[:, 0]  # Confidence of being a QR

        # 4. Create Visualization
        W, H = CFG.img_w, CFG.img_h
        img_np = (images[0, 0].cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np).convert("RGB")

        img_b = img_pil.copy()
        draw_a = ImageDraw.Draw(img_pil)
        draw_b = ImageDraw.Draw(img_b)

        cmap = plt.get_cmap("hsv")

        detected_qrs = []
        print(f"\n--- Sample {r + 1}/{num_samples} Query Analysis ---")
        for i in range(CFG.num_queries):
            box = pred_boxes[i]
            conf = probs[i].item()

            x1, y1, x2, y2 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
            color = tuple((np.array(cmap(i / CFG.num_queries)[:3]) * 255).astype(int))

            draw_b.rectangle([x1, y1, x2, y2], outline=color, width=1)
            draw_b.text((x1, y1), f"Q{i}", fill=color)

            if conf > CFG.conf_threshold:
                print(f"Query {i:02d} detected QR code with {conf:.2%} confidence")
                detected_qrs.append((i, conf, x1, y1, x2, y2))
                draw_a.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
                draw_a.text((x1 + 5, y1 + 5), f"ID:{i} {conf:.2f}", fill=(0, 255, 0))

        # 5. Spatial Bias Test
        blank = torch.full((1, 1, H, W), 0.5).to(device)
        blank_out = model(blank)
        blank_boxes = blank_out["pred_boxes"][0].cpu()
        blank_logits = blank_out["pred_logits"][0].cpu()
        blank_probs = blank_logits.softmax(-1)[:, 0]

        img_dream = Image.new("RGB", (W, H), (128, 128, 128))
        draw_dream = ImageDraw.Draw(img_dream)
        for i in range(CFG.num_queries):
            box = blank_boxes[i]
            x1, y1, x2, y2 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
            color = tuple((np.array(cmap(i / CFG.num_queries)[:3]) * 255).astype(int))
            draw_dream.rectangle([x1, y1, x2, y2], outline=color, width=1)
            draw_dream.text((x1 + 2, y1 + 2), f"Q{i}", fill=color)

        # 6. Save page for this sample
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_pil)
        axes[0].set_title(
            f"Sample {r + 1}/{num_samples} - Detections (Threshold={CFG.conf_threshold:.2f})"
        )
        axes[0].axis("off")

        axes[1].imshow(img_b)
        axes[1].set_title(f"All {CFG.num_queries} Query Slots (Spatial Bias)")
        axes[1].axis("off")

        axes[2].imshow(img_dream)
        axes[2].set_title("The 'Dream' Test (Blank Image)")
        axes[2].axis("off")

        axes[3].axis("off")
        axes[3].text(
            0.1,
            0.9,
            f"Sample {r + 1} Results",
            fontsize=14,
            fontweight="bold",
            transform=axes[3].transAxes,
        )
        axes[3].text(
            0.1,
            0.8,
            f"Detected QR codes: {len(detected_qrs)}",
            fontsize=12,
            transform=axes[3].transAxes,
        )
        axes[3].text(
            0.1,
            0.7,
            f"Highest confidence: {max([d[1] for d in detected_qrs], default=0):.2%}",
            fontsize=12,
            transform=axes[3].transAxes,
        )
        axes[3].text(0.1, 0.55, "Detected Queries:", fontsize=11, transform=axes[3].transAxes)
        for idx, (qid, conf, _, _, _, _) in enumerate(detected_qrs):
            axes[3].text(
                0.1,
                0.5 - idx * 0.05,
                f"  Q{qid:02d}: {conf:.2%}",
                fontsize=10,
                transform=axes[3].transAxes,
            )
        axes[3].text(
            0.1, 0.2, f"Epoch: {checkpoint['epoch']}", fontsize=10, transform=axes[3].transAxes
        )

        plt.tight_layout()
        plt.show()  # Display in Jupyter notebook
        report.append(plt.gcf())  # Store figure object
        print(f"Page {r + 1} displayed")
        plt.close()

    # 7. Generate summary page
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.axis("off")
    ax.text(
        0.5,
        0.95,
        "Interpretability Report Summary",
        fontsize=18,
        fontweight="bold",
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.85,
        f"Total samples tested: {num_samples}",
        fontsize=14,
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.75,
        f"Model epoch: {checkpoint['epoch']}",
        fontsize=12,
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.65,
        f"Confidence threshold: {CFG.conf_threshold}",
        fontsize=12,
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.55,
        f"Number of query slots: {CFG.num_queries}",
        fontsize=12,
        ha="center",
        transform=ax.transAxes,
    )
    ax.text(0.5, 0.45, "Pages generated:", fontsize=12, ha="center", transform=ax.transAxes)
    for i in range(num_samples):
        ax.text(
            0.5,
            0.35 - i * 0.05,
            f"  Page {i + 1}: Sample {i + 1}",
            fontsize=10,
            ha="center",
            transform=ax.transAxes,
        )
    plt.tight_layout()
    plt.show()  # Display in Jupyter notebook
    report.append(plt.gcf())  # Store figure object
    print(f"Summary displayed")
    plt.close()

    print(f"\n=== Report Complete ===")
    print(f"Generated {len(report)} pages:")
    for p in report:
        print(f"  - {p}")


if __name__ == "__main__":
    run_interpretability("checkpoints/last.pt", 5)
    # run_interpretability("checkpoints/best.pt", 5)
