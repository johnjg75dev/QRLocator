# train_eval.py
# ─────────────────────────────────────────────────────────────────────────────
# Training, evaluation, and inference CLI.
#
# Usage:
#   # Train
#   python train_eval.py --mode train
#
#   # Evaluate on test set
#   python train_eval.py --mode eval --checkpoint checkpoints/best.pt
#
#   # Run inference on a single image
#   python train_eval.py --mode infer --checkpoint checkpoints/best.pt \
#                        --image path/to/image.png --output out.png
# ─────────────────────────────────────────────────────────────────────────────
import os
import argparse
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from config import CFG
from dataset import build_dataloaders
from model import QRViTDet


# ──────────────────────────────────────────────────────────────────────────────
# LOSS
# ──────────────────────────────────────────────────────────────────────────────

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] → [x1, y1, x2, y2]. No-op here since we already use xyxy."""
    return boxes


def generalized_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise GIoU between two sets of boxes.
    Both inputs: (N, 4) in [x1, y1, x2, y2] format, values in [0, 1].
    Returns: (N, M) GIoU matrix.
    """
    N, M = len(boxes1), len(boxes2)
    x1 = boxes1[:, 0].unsqueeze(1).expand(N, M)
    y1 = boxes1[:, 1].unsqueeze(1).expand(N, M)
    x2 = boxes1[:, 2].unsqueeze(1).expand(N, M)
    y2 = boxes1[:, 3].unsqueeze(1).expand(N, M)

    gx1 = boxes2[:, 0].unsqueeze(0).expand(N, M)
    gy1 = boxes2[:, 1].unsqueeze(0).expand(N, M)
    gx2 = boxes2[:, 2].unsqueeze(0).expand(N, M)
    gy2 = boxes2[:, 3].unsqueeze(0).expand(N, M)

    inter_x1 = torch.max(x1, gx1)
    inter_y1 = torch.max(y1, gy1)
    inter_x2 = torch.min(x2, gx2)
    inter_y2 = torch.min(y2, gy2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area2 = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    union_area = area1 + area2 - inter_area + 1e-6

    iou = inter_area / union_area

    # Enclosing box
    enc_x1 = torch.min(x1, gx1)
    enc_y1 = torch.min(y1, gy1)
    enc_x2 = torch.max(x2, gx2)
    enc_y2 = torch.max(y2, gy2)
    enc_area = ((enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)).clamp(min=1e-6)

    giou = iou - (enc_area - union_area) / enc_area
    return giou  # (N, M)


def hungarian_match(
    pred_boxes: torch.Tensor,   # (Q, 4)
    pred_logits: torch.Tensor,  # (Q, 2)
    gt_boxes: torch.Tensor,     # (G, 4)
    cfg=CFG,
) -> Tuple[List[int], List[int]]:
    """
    Compute the optimal bipartite assignment between Q predictions and G ground-truths.
    Returns (pred_indices, gt_indices) after Hungarian matching.
    """
    Q, G = len(pred_boxes), len(gt_boxes)
    if G == 0:
        return [], []

    with torch.no_grad():
        # Class cost: negative probability of being a QR code
        prob = pred_logits.softmax(-1)          # (Q, 2)
        cost_class = -prob[:, 0].unsqueeze(1).expand(Q, G)   # (Q, G)

        # L1 cost
        cost_l1 = torch.cdist(pred_boxes, gt_boxes, p=1)     # (Q, G)

        # GIoU cost
        giou = generalized_iou(pred_boxes, gt_boxes)          # (Q, G)
        cost_giou = -giou

        cost = (
            cfg.cost_class * cost_class
            + cfg.cost_bbox * cost_l1
            + cfg.cost_giou * cost_giou
        )
        cost_np = cost.cpu().numpy()

    row_ind, col_ind = linear_sum_assignment(cost_np)
    return row_ind.tolist(), col_ind.tolist()


class SetCriterion(nn.Module):
    """
    DETR-style set prediction loss.
    For each image: run Hungarian matching, then compute
      - Cross-entropy for classification (incl. no-object class)
      - L1 + GIoU for matched box pairs
    """

    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        # Down-weight the "no-object" class to handle the class imbalance
        # between num_queries (many) and actual QR codes (few)
        eos_weight = torch.ones(2)
        eos_weight[1] = cfg.eos_coef   # index 1 = "no-object"
        self.register_buffer("eos_weight", eos_weight)

    def forward(
        self,
        pred_boxes_batch: torch.Tensor,    # (B, Q, 4)
        pred_logits_batch: torch.Tensor,   # (B, Q, 2)
        gt_boxes_list: List[torch.Tensor], # list of (G_i, 4)
    ) -> dict:
        B, Q, _ = pred_boxes_batch.shape
        device = pred_boxes_batch.device

        total_loss_class = torch.tensor(0.0, device=device)
        total_loss_bbox  = torch.tensor(0.0, device=device)
        total_loss_giou  = torch.tensor(0.0, device=device)
        num_boxes = 0

        for b in range(B):
            pred_b = pred_boxes_batch[b]    # (Q, 4)
            logit_b = pred_logits_batch[b]  # (Q, 2)
            gt_b = gt_boxes_list[b].to(device)  # (G, 4)
            G = len(gt_b)

            # ── Classification loss for ALL queries ────────────────────────
            # Target: "no-object" (1) for all, overwrite matched with 0
            tgt_labels = torch.ones(Q, dtype=torch.long, device=device)  # all no-obj

            if G > 0:
                pred_idx, gt_idx = hungarian_match(pred_b, logit_b, gt_b, self.cfg)
                tgt_labels[pred_idx] = 0   # matched queries → "qr_code" class

                # ── Box losses only for matched pairs ──────────────────────
                matched_pred = pred_b[pred_idx]    # (|match|, 4)
                matched_gt   = gt_b[gt_idx]         # (|match|, 4)
                n_match = len(pred_idx)

                # L1
                loss_l1 = F.l1_loss(matched_pred, matched_gt, reduction="sum")

                # GIoU
                giou_mat = generalized_iou(matched_pred, matched_gt)
                loss_giou = (1 - giou_mat.diag()).sum()

                total_loss_bbox += loss_l1
                total_loss_giou += loss_giou
                num_boxes += n_match

            # Cross-entropy (weighted for no-object imbalance)
            total_loss_class += F.cross_entropy(logit_b, tgt_labels, weight=self.eos_weight)

        # Normalise box losses by total matched boxes across batch
        norm = max(num_boxes, 1)
        loss_dict = {
            "loss_class": total_loss_class / B,
            "loss_bbox":  total_loss_bbox  / norm,
            "loss_giou":  total_loss_giou  / norm,
        }
        total = (
            self.cfg.loss_class * loss_dict["loss_class"]
            + self.cfg.loss_bbox  * loss_dict["loss_bbox"]
            + self.cfg.loss_giou  * loss_dict["loss_giou"]
        )
        loss_dict["total"] = total
        return loss_dict


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def iou_single(b1: np.ndarray, b2: np.ndarray) -> float:
    """IoU between two boxes [x1,y1,x2,y2]."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


@torch.no_grad()
def compute_map(
    model: nn.Module,
    dataloader,
    cfg=CFG,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Compute Precision, Recall, and mAP@0.5 over the dataloader.
    """
    model.eval()
    all_tp = all_fp = all_fn = 0
    total_iou = 0.0
    total_gt = 0

    for images, gt_boxes_list, _ in tqdm(dataloader, desc="Eval", leave=False):
        images = images.to(device)
        out = model(images)
        pred_boxes_batch  = out["pred_boxes"]   # (B, Q, 4)
        pred_logits_batch = out["pred_logits"]  # (B, Q, 2)

        B = images.size(0)
        for b in range(B):
            # Filter predictions by confidence
            probs = pred_logits_batch[b].softmax(-1)[:, 0]  # P(qr_code)
            keep  = probs >= cfg.conf_threshold
            pred_b = pred_boxes_batch[b][keep].cpu().numpy()

            gt_b = gt_boxes_list[b].numpy()
            G = len(gt_b)
            P = len(pred_b)
            total_gt += G

            if G == 0 and P == 0:
                continue
            if G == 0:
                all_fp += P
                continue
            if P == 0:
                all_fn += G
                continue

            # Greedy match by IoU
            matched_gt = set()
            for pred in pred_b:
                best_iou, best_j = 0.0, -1
                for j, gt in enumerate(gt_b):
                    if j in matched_gt:
                        continue
                    iou = iou_single(pred, gt)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= cfg.iou_threshold:
                    all_tp += 1
                    matched_gt.add(best_j)
                    total_iou += best_iou
                else:
                    all_fp += 1
            all_fn += G - len(matched_gt)

    eps = 1e-6
    precision = all_tp / (all_tp + all_fp + eps)
    recall    = all_tp / (all_tp + all_fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    mean_iou  = total_iou / (all_tp + eps)

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "mean_iou":  mean_iou,
        "tp": all_tp, "fp": all_fp, "fn": all_fn,
    }


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg=CFG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    train_dl, val_dl, _ = build_dataloaders(cfg)
    print(f"Train batches: {len(train_dl)} | Val batches: {len(val_dl)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = QRViTDet(cfg).to(device)

    # ── Optimiser: separate LR for backbone ─────────────────────────────────
    param_groups = [
        {"params": model.backbone_params,     "lr": cfg.lr_backbone},
        {"params": model.non_backbone_params, "lr": cfg.lr},
    ]
    optimizer = AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = StepLR(optimizer, step_size=cfg.lr_drop, gamma=0.1)
    criterion = SetCriterion(cfg).to(device)

    checkpoint_path = f"{cfg.checkpoint_dir}/best.pt"
    if os.path.exists(checkpoint_path):
        print(f"Resuming from existing checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        end_epoch = start_epoch + cfg.epochs + 1
        best_f1 = ckpt["metrics"]["f1"]
        print(f"Resumed at epoch {start_epoch}, current best F1: {best_f1:.4f}")
    else:
        start_epoch = 1
        end_epoch = cfg.epochs + 1
        best_f1 = 0.0
        print("No existing checkpoint found - starting fresh training")
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {n_params:,}")

    for epoch in range(start_epoch, end_epoch):
        model.train()
        running = {"total": 0.0, "loss_class": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
        t0 = time.time()
        #print("Starting epoch {:03d}...".format(epoch))

        for batch_idx, (images, gt_boxes_list, _) in enumerate(train_dl):
            images = images.to(device)

            out = model(images)
            loss_dict = criterion(out["pred_boxes"], out["pred_logits"], gt_boxes_list)
            loss = loss_dict["total"]
            #print(f"Batch {batch_idx+1}/{len(train_dl)}  Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            for k in running:
                running[k] += loss_dict[k].item()

            if (batch_idx + 1) % cfg.log_interval == 0:
                n = cfg.log_interval
                print(
                    f"Epoch {epoch:03d} [{batch_idx+1}/{len(train_dl)}] "
                    f"loss={running['total']/n:.4f}  "
                    f"cls={running['loss_class']/n:.4f}  "
                    f"bbox={running['loss_bbox']/n:.4f}  "
                    f"giou={running['loss_giou']/n:.4f}  "
                    f"({time.time()-t0:.1f}s)"
                )
                running = {k: 0.0 for k in running}
                t0 = time.time()

        scheduler.step()

        # ── Validation ───────────────────────────────────────────────────────
        metrics = compute_map(model, val_dl, cfg, device)
        print(
            f"  [Val] Epoch {epoch:03d}  "
            f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
            f"F1={metrics['f1']:.4f}  mIoU={metrics['mean_iou']:.4f}"
        )

        # ── Checkpoint ───────────────────────────────────────────────────────
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "metrics": metrics,
            "cfg": cfg,
        }
        torch.save(ckpt, cfg.checkpoint_dir / "last.pt")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(ckpt, cfg.checkpoint_dir / "best.pt")
            print(f"  ★ New best F1: {best_f1:.4f} → saved best.pt")

    print(f"\nTraining complete. Best val F1 = {best_f1:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(checkpoint_path: str, cfg=CFG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = QRViTDet(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    _, _, test_dl = build_dataloaders(cfg)
    metrics = compute_map(model, test_dl, cfg, device)

    print("\n── Test Set Metrics ────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:>12s}: {v}")
    print("────────────────────────────────────────────")
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(
    image_path: str,
    checkpoint_path: str,
    output_path: str = "output.png",
    cfg=CFG,
):
    """
    Run inference on a single image.
    Saves the input image with detected QR bounding boxes overlaid.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = QRViTDet(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load & preprocess image
    img = Image.open(image_path).convert("L").resize((cfg.img_w, cfg.img_h))
    img_np = np.array(img)
    img_tensor = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)  # (1, 1, H, W)

    # Forward
    out = model(img_tensor)
    pred_boxes  = out["pred_boxes"][0]   # (Q, 4)
    pred_logits = out["pred_logits"][0]  # (Q, 2)

    # Filter by confidence
    probs = pred_logits.softmax(-1)[:, 0]
    keep  = probs >= cfg.conf_threshold
    final_boxes = pred_boxes[keep].cpu().numpy()
    final_confs = probs[keep].cpu().numpy()

    # Visualise
    W, H = img.size
    pil_out = img.convert("RGB")
    draw = ImageDraw.Draw(pil_out)

    print(f"\nDetected {len(final_boxes)} QR code(s):")
    for i, (box, conf) in enumerate(zip(final_boxes, final_confs)):
        x1 = int(box[0] * W);  y1 = int(box[1] * H)
        x2 = int(box[2] * W);  y2 = int(box[3] * H)
        print(f"  [{i+1}] TL=({x1},{y1})  BR=({x2},{y2})  conf={conf:.3f}")
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        draw.text((x1 + 4, y1 + 4), f"{conf:.2f}", fill=(255, 255, 0))

    pil_out.save(output_path)
    print(f"Annotated image saved → {output_path}")
    return final_boxes, final_confs


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
DEBUG = False  # Set to False for full training/eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QR-ViT-Det: Train / Eval / Infer")
    parser.add_argument("--mode", choices=["train", "eval", "infer"], required=True)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (.pt) for eval/infer modes")
    parser.add_argument("--image",  type=str, default=None, help="Image path for infer")
    parser.add_argument("--output", type=str, default="output.png",
                        help="Output image path for infer")
    
    # Parser for epoch, batch size, learning rate
    '''parser.add_argument("--epochs", type=int, default=CFG.epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size, help="Training batch size")
    parser.add_argument("--lr", type=float, default=CFG.lr, help="Learning rate for non-backbone")
    parser.add_argument("--lr_backbone", type=float, default=CFG.lr_backbone, help="Learning rate for backbone")
    parser.add_argument("--grad_clip", type=float, default=CFG.grad_clip, help="Max gradient norm for clipping")
    parser.add_argument("--log_interval", type=int, default=CFG.log_interval, help="Batches between logging during training")
    #Debug flags -D or --debug to run quick train/eval with reduced dataset and epochs
    parser.add_argument("--debug","-D", action="store_true", help="Run in debug mode")'''

    if not DEBUG:
        args = parser.parse_args()

        if args.mode == "train":
            train(CFG)

        elif args.mode == "eval":
            if not args.checkpoint:
                raise ValueError("--checkpoint required for eval mode")
            evaluate(args.checkpoint, CFG)

        elif args.mode == "infer":
            if not args.checkpoint or not args.image:
                raise ValueError("--checkpoint and --image required for infer mode")
            infer(args.image, args.checkpoint, args.output, CFG)
    else:
        print("DEBUG MODE: Running quick train and eval with reduced dataset and epochs.")
        debug_cfg = CFG
        #debug_cfg.num_train = 100
        #debug_cfg.num_val = 20
        #debug_cfg.num_test = 20
        #debug_cfg.epochs = 2
        train(debug_cfg)
        evaluate(f"{debug_cfg.checkpoint_dir}last.pt", debug_cfg)