# dataset.py
# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset and DataLoader factories for the QR detection task.
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import CFG
from dataset_gen import build_augmentation_pipeline


class QRDataset(Dataset):
    """
    Loads pre-generated scenes from disk.

    Each sample returns:
      image  : FloatTensor (1, H, W)  values in [0, 1]
      boxes  : FloatTensor (N, 4)     [x1, y1, x2, y2] normalised
      n_obj  : int                    actual number of QR codes in this image
    """

    def __init__(self, split: str = "train", cfg=CFG):
        self.cfg = cfg
        self.split = split
        self.img_dir = cfg.dataset_dir / split / "images"
        label_file = cfg.dataset_dir / split / "labels.json"

        with open(label_file) as f:
            raw = json.load(f)

        # Convert to list of (filename, list_of_boxes)
        self.samples: List[Tuple[str, List[Dict]]] = [
            (fname, boxes) for fname, boxes in raw.items()
        ]
        self.augment = build_augmentation_pipeline(train=(split == "train"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fname, box_dicts = self.samples[idx]
        img = np.array(Image.open(self.img_dir / fname).convert("L"))  # (H, W)

        # Convert box dicts → list of [x1, y1, x2, y2]
        bboxes = [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in box_dicts]
        class_labels = [0] * len(bboxes)  # single class: "qr_code"

        # Apply albumentations (bbox-aware)
        result = self.augment(
            image=img,
            bboxes=bboxes,
            class_labels=class_labels,
        )
        img_aug = result["image"]           # (H, W) uint8
        bboxes_aug = result["bboxes"]       # list of (x1,y1,x2,y2) tuples

        # To tensor
        img_tensor = torch.from_numpy(img_aug).float().unsqueeze(0) / 255.0  # (1,H,W)

        n_obj = len(bboxes_aug)
        if n_obj > 0:
            boxes_tensor = torch.tensor(bboxes_aug, dtype=torch.float32)  # (n_obj, 4)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        return img_tensor, boxes_tensor, n_obj


def collate_fn(batch):
    """
    Custom collate: images are stacked, boxes kept as a list of tensors
    (variable length per image).
    """
    images, boxes_list, n_objs = zip(*batch)
    images = torch.stack(images, dim=0)  # (B, 1, H, W)
    # boxes_list: tuple of FloatTensor (n_i, 4)
    return images, list(boxes_list), list(n_objs)


def build_dataloaders(cfg=CFG):
    train_ds = QRDataset("train", cfg)
    val_ds   = QRDataset("val",   cfg)
    test_ds  = QRDataset("test",  cfg)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    return train_dl, val_dl, test_dl