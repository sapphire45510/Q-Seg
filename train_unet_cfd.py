from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_cfd_seg_mask(seg_path: str | Path, crack_label: int = 1) -> np.ndarray:
    width = height = None
    data_started = False
    runs: List[Tuple[int, int, int, int]] = []

    with Path(seg_path).open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("width "):
                width = int(line.split()[1])
                continue
            if line.startswith("height "):
                height = int(line.split()[1])
                continue
            if line == "data":
                data_started = True
                continue
            if data_started:
                parts = line.split()
                if len(parts) == 4:
                    runs.append(tuple(map(int, parts)))

    if width is None or height is None:
        raise ValueError(f"Cannot read width/height from {seg_path}")

    mask = np.zeros((height, width), dtype=np.uint8)
    for label, y, x_start, x_end in runs:
        if label == crack_label:
            mask[y, max(0, x_start):min(width - 1, x_end) + 1] = 1
    return mask


@dataclass(frozen=True)
class SamplePair:
    image_path: str
    mask_path: str
    sample_id: str


def discover_pairs(dataset_dir: str | Path) -> List[SamplePair]:
    root = Path(dataset_dir)
    image_dir = root / "image"
    seg_dir = root / "seg"

    pairs: List[SamplePair] = []
    images: List[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        images.extend(image_dir.glob(pattern))

    for image_path in sorted(images):
        seg_path = seg_dir / f"{image_path.stem}.seg"
        if seg_path.exists():
            pairs.append(SamplePair(str(image_path), str(seg_path), image_path.stem))

    if not pairs:
        raise RuntimeError(f"No CFD image/.seg pairs found under {root}")
    return pairs


def split_pairs(
    pairs: Sequence[SamplePair],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[SamplePair], List[SamplePair], List[SamplePair]]:
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    return (
        shuffled[:n_train],
        shuffled[n_train:n_train + n_val],
        shuffled[n_train + n_val:],
    )


class CFDDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[SamplePair],
        width: int = 480,
        height: int = 320,
        augment: bool = False,
    ) -> None:
        self.pairs = list(pairs)
        self.width = width
        self.height = height
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        image = Image.open(pair.image_path).convert("L")
        mask = load_cfd_seg_mask(pair.mask_path)

        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img = mask_img.resize((self.width, self.height), Image.Resampling.NEAREST)
            mask = (np.asarray(mask_img, dtype=np.uint8) > 0).astype(np.uint8)

        x = np.asarray(image, dtype=np.float32) / 255.0
        y = mask.astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                x = np.fliplr(x).copy()
                y = np.fliplr(y).copy()
            if random.random() < 0.5:
                x = np.flipud(x).copy()
                y = np.flipud(y).copy()

        return {
            "image": torch.from_numpy(x[None]).float(),
            "mask": torch.from_numpy(y[None]).float(),
            "sample_id": pair.sample_id,
        }


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(1, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.bottleneck = DoubleConv(c * 8, c * 16)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, 2)
        self.dec4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, 2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, 2)
        self.dec1 = DoubleConv(c * 2, c)
        self.out = nn.Conv2d(c, 1, 1)

    @staticmethod
    def match_size(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.match_size(self.up4(b), e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.match_size(self.up3(d4), e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.match_size(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.match_size(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 10.0) -> None:
        super().__init__()
        self.pos_weight_value = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = torch.tensor([self.pos_weight_value], device=logits.device)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight
        )
        probs = torch.sigmoid(logits).flatten(start_dim=1)
        flat_targets = targets.flatten(start_dim=1)
        intersection = (probs * flat_targets).sum(dim=1)
        dice = (2 * intersection + 1.0) / (probs.sum(dim=1) + flat_targets.sum(dim=1) + 1.0)
        return 0.5 * bce + 0.5 * (1.0 - dice.mean())


def calculate_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int(np.sum(p & g))
    tn = int(np.sum(~p & ~g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))
    eps = 1e-12
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "IoU": tp / (tp + fp + fn + eps),
        "Dice": 2 * tp / (2 * tp + fp + fn + eps),
        "Precision": tp / (tp + fp + eps),
        "Recall": tp / (tp + fn + eps),
        "Specificity": tn / (tn + fp + eps),
        "Pixel Accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "Predicted positive ratio": float(np.mean(p)),
        "Ground-truth positive ratio": float(np.mean(g)),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    tp = tn = fp = fn = count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                loss.backward()
                optimizer.step()

            pred = torch.sigmoid(logits) >= 0.5
            gt = masks >= 0.5
            tp += int(torch.sum(pred & gt))
            tn += int(torch.sum(~pred & ~gt))
            fp += int(torch.sum(pred & ~gt))
            fn += int(torch.sum(~pred & gt))
            total_loss += float(loss.item()) * images.shape[0]
            count += images.shape[0]

    eps = 1e-12
    return {
        "loss": total_loss / max(count, 1),
        "IoU": tp / (tp + fp + fn + eps),
        "Dice": 2 * tp / (2 * tp + fp + fn + eps),
        "Precision": tp / (tp + fp + eps),
        "Recall": tp / (tp + fn + eps),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)


def save_figure(path: Path, image: np.ndarray, gt: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, title, arr in zip(
        axes,
        ["Input", "Ground truth", "U-Net mask", "Probability"],
        [image, gt, pred, prob],
    ):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def train(args, train_pairs, val_pairs, device: torch.device) -> Path:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        CFDDataset(train_pairs, args.resize_width, args.resize_height, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        CFDDataset(val_pairs, args.resize_width, args.resize_height, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = UNet(args.base_channels).to(device)
    criterion = BCEDiceLoss(args.pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_dice = -1.0
    best_path = out / "best_unet.pt"
    history: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr = run_epoch(model, train_loader, criterion, device, optimizer)
        va = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va["Dice"])

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_iou": tr["IoU"],
            "train_dice": tr["Dice"],
            "val_loss": va["loss"],
            "val_iou": va["IoU"],
            "val_dice": va["Dice"],
            "val_precision": va["Precision"],
            "val_recall": va["Recall"],
            "seconds": time.time() - start,
        }
        history.append(row)
        write_csv(out / "training_history.csv", history)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train Dice={tr['Dice']:.4f} | "
            f"val Dice={va['Dice']:.4f}, IoU={va['IoU']:.4f}"
        )

        if va["Dice"] > best_dice:
            best_dice = va["Dice"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "base_channels": args.base_channels,
                    "resize_width": args.resize_width,
                    "resize_height": args.resize_height,
                    "epoch": epoch,
                    "best_val_dice": best_dice,
                },
                best_path,
            )
            print("  saved:", best_path)

    return best_path


def evaluate(args, pairs, checkpoint_path: Path, device: torch.device) -> None:
    out = Path(args.output_dir)
    masks_dir = out / "masks"
    figs_dir = out / "figures"
    masks_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = UNet(int(ckpt.get("base_channels", 32))).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    width = int(ckpt.get("resize_width", args.resize_width))
    height = int(ckpt.get("resize_height", args.resize_height))
    loader = DataLoader(CFDDataset(pairs, width, height), batch_size=1, shuffle=False)

    rows: List[Dict[str, object]] = []
    all_tp = all_tn = all_fp = all_fn = 0

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            gt = batch["mask"][0, 0].numpy()
            sample_id = batch["sample_id"][0]
            prob = torch.sigmoid(model(image))[0, 0].cpu().numpy()
            pred = (prob >= args.threshold).astype(np.uint8)
            metrics = calculate_metrics(pred, gt)
            rows.append({"sample_id": sample_id, **metrics})
            all_tp += metrics["TP"]
            all_tn += metrics["TN"]
            all_fp += metrics["FP"]
            all_fn += metrics["FN"]
            save_mask(masks_dir / f"{sample_id}_unet_mask.png", pred)
            np.save(masks_dir / f"{sample_id}_probability.npy", prob)
            save_figure(figs_dir / f"{sample_id}_comparison.png", image[0, 0].cpu().numpy(), gt, pred, prob)

    eps = 1e-12
    summary = {
        "sample_id": "ALL",
        "TP": all_tp,
        "TN": all_tn,
        "FP": all_fp,
        "FN": all_fn,
        "IoU": all_tp / (all_tp + all_fp + all_fn + eps),
        "Dice": 2 * all_tp / (2 * all_tp + all_fp + all_fn + eps),
        "Precision": all_tp / (all_tp + all_fp + eps),
        "Recall": all_tp / (all_tp + all_fn + eps),
        "Specificity": all_tn / (all_tn + all_fp + eps),
        "Pixel Accuracy": (all_tp + all_tn) / (all_tp + all_tn + all_fp + all_fn + eps),
        "Predicted positive ratio": (all_tp + all_fp) / (all_tp + all_tn + all_fp + all_fn + eps),
        "Ground-truth positive ratio": (all_tp + all_fn) / (all_tp + all_tn + all_fp + all_fn + eps),
    }
    rows.append(summary)
    write_csv(out / "metrics.csv", rows)
    print(
        f"Test: IoU={summary['IoU']:.4f}, Dice={summary['Dice']:.4f}, "
        f"Precision={summary['Precision']:.4f}, Recall={summary['Recall']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="U-Net baseline for CrackForest Dataset")
    p.add_argument("--mode", choices=["train", "evaluate"], default="train")
    p.add_argument("--dataset_dir", default="CrackForest-dataset")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--output_dir", default="unet_cfd_results")
    p.add_argument("--resize_width", type=int, default=480)
    p.add_argument("--resize_height", type=int, default=320)
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--pos_weight", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    pairs = discover_pairs(args.dataset_dir)
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs, args.train_ratio, args.val_ratio, args.seed
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "dataset_split.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train": [asdict(x) for x in train_pairs],
                "validation": [asdict(x) for x in val_pairs],
                "test": [asdict(x) for x in test_pairs],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Dataset={len(pairs)} | train={len(train_pairs)} | "
        f"validation={len(val_pairs)} | test={len(test_pairs)}"
    )

    if args.mode == "train":
        best = train(args, train_pairs, val_pairs, device)
        evaluate(args, test_pairs, best, device)
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required in evaluate mode")
        evaluate(args, test_pairs, Path(args.checkpoint), device)


if __name__ == "__main__":
    main()
