import os
import platform
import pathlib
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
from pathlib import Path
import pickle
import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from matplotlib.backends.backend_pdf import PdfPages
from config import CFG
from model import QRViTDet
from dataset import build_dataloaders
from PIL import Image, ImageDraw

report = []


def create_filter_visualization(model, cfg):
    """Create a matplotlib figure showing the CNN filters from the CNNStem."""
    import math

    # Get the CNNStem layers
    backbone = model.backbone.body

    # We'll create a figure with subplots for each convolutional layer
    num_layers = len(backbone)
    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))
    if num_layers == 1:
        axes = [axes]

    for layer_idx, layer in enumerate(backbone):
        if isinstance(layer, nn.Conv2d):
            # Get the weight tensor: [out_channels, in_channels, kernel_size, kernel_size]
            weights = layer.weight.data

            # For grayscale input, in_channels = 1, so we can visualize each filter
            # as a 2D image (kernel_size x kernel_size)
            out_channels = weights.size(0)

            # Calculate grid dimensions
            cols = math.ceil(math.sqrt(out_channels))
            rows = math.ceil(out_channels / cols)

            # Create a grid of subplots for this layer's filters
            layer_fig, layer_axes = plt.subplots(cols, cols, figsize=(cols * 2, rows * 2))
            layer_fig.suptitle(
                f"Layer {layer_idx + 1} - Conv2d (in={weights.size(1)}, out={out_channels}, kernel={weights.size(2)}x{weights.size(3)})",
                fontsize=14,
            )

            # Flatten axes for easy iteration
            layer_axes = layer_axes.flatten() if cols > 1 or rows > 1 else [layer_axes]

            # Normalize weights for visualization
            min_val, max_val = weights.min().item(), weights.max().item()

            for i in range(out_channels):
                if i < len(layer_axes):
                    # Extract the filter (remove the in_channels dimension since it's 1)
                    filter_img = weights[i, 0, :, :].cpu().numpy()

                    # Plot the filter
                    layer_axes[i].imshow(filter_img, cmap="gray", vmin=min_val, vmax=max_val)
                    layer_axes[i].axis("off")
                    layer_axes[i].set_title(f"Filter {i + 1}")

            # Hide any unused subplots
            for i in range(len(layer_axes), out_channels):
                layer_axes[i].axis("off")

            # Add the layer figure to the report
            report.append(layer_fig)


def create_config_page(CFG, checkpoint):
    """Create a matplotlib figure showing the configuration settings."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.axis("off")
    ax.text(
        0.5,
        0.95,
        "Configuration Settings",
        fontsize=18,
        fontweight="bold",
        ha="center",
        transform=ax.transAxes,
    )
    cfg_attrs = [
        attr for attr in dir(CFG) if not attr.startswith("_") and not callable(getattr(CFG, attr))
    ]
    y_pos = 0.85
    for attr in cfg_attrs:
        value = getattr(CFG, attr)
        ax.text(0.1, y_pos, f"{attr}: {value}", fontsize=12, ha="left", transform=ax.transAxes)
        y_pos -= 0.05
    ax.text(
        0.1,
        y_pos,
        f"Checkpoint epoch: {checkpoint['epoch']}",
        fontsize=12,
        ha="left",
        transform=ax.transAxes,
    )
    plt.tight_layout()
    return fig


@torch.no_grad()
def run_interpretability(
    checkpoint_path="checkpoints/best.pt", num_samples=1, datasets_path="data/qr_dataset"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CFG.dataset_dir = Path(datasets_path)
    CFG.checkpoint_dir = Path(checkpoint_path)

    print(
        f"Running interpretability report on {device} with model '{checkpoint_path}' and dataset '{datasets_path}'..."
    )
    # 1. Load Model
    model = QRViTDet(CFG).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False, encoding="latin1")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Interpreting Model from Epoch {checkpoint['epoch']}")

    # Create config page and add to report
    config_fig = create_config_page(CFG, checkpoint)
    report.append(config_fig)

    # Create visualization of CNN filters from the model's CNNStem
    filter_fig = create_filter_visualization(model, CFG)
    report.append(filter_fig)
    print("Filter visualization page generated")

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
        # Store figure for report
        report.append(plt.gcf())
        print(f"Page {r + 1} generated")

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
    # Store figure for report
    report.append(plt.gcf())
    print(f"Summary generated")

    # Save all figures to a PDF file for pagination
    pdf_path = "interpretability_report.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in report:
            pdf.savefig(fig)
    print(f"\nReport saved to {pdf_path}")
    # Open PDF with default viewer
    if platform.system() == "Windows":
        os.startfile(pdf_path)
    elif platform.system() == "Darwin":
        subprocess.run(["open", pdf_path])
    else:
        subprocess.run(["xdg-open", pdf_path])
    plt.close("all")

    print(f"\n=== Report Complete ===")
    print(f"Generated {len(report)} pages:")
    for p in report:
        print(f"  - {p}")


if __name__ == "__main__":
    # run_interpretability("checkpoints/best.pt", 5, "data/qr_dataset")
    run_interpretability("checkpoints/best.pt", 5, "data/qr_dataset")
    """parser = argparse.ArgumentParser(prog="interpret.py", description="Interpretability Report: Run and visualize model predictions with detailed analysis.")
    parser.add_argument("--model","-m", type=str, required=True, help="Model name (e.g., 'checkpoints/best.pt')")
    parser.add_argument("--samples","-n", type=int, default=1, help="Number of samples for test (default: 1))")
    parser.add_argument("--dataset","-d", type=str, default="data/qr_dataset", help="Directory containing datasets (default: 'data/qr_dataset')")

    args = parser.parse_args()
    if args.model is None or args.dataset is None:
        print("Error: Please provide a valid model checkpoint path and dataset directory.")
        parser.print_help()
        print("\nExample: python interpret.py -m checkpoints/best.pt -d data/qr_dataset")
        exit(1)
    
    # Run the interpretability report
    run_interpretability(args.model, args.samples, args.dataset)"""
