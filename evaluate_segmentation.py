import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import ListedColormap


def load_mask(path, target_shape=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        mask = np.load(path)

    else:
        image = Image.open(path)

        # Labelme 的 label.png 通常是 palette/indexed image。
        # 必須直接讀取類別 ID，不能先 convert("L")。
        if image.mode in ("P", "I", "I;16"):
            mask = np.asarray(image)
        else:
            # 一般黑白 mask 才轉成灰階。
            mask = np.asarray(image.convert("L"))

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got {mask.shape}: {path}")

    if target_shape is not None and tuple(mask.shape) != tuple(target_shape):
        # 保留離散類別 ID，必須使用 NEAREST。
        image = Image.fromarray(mask.astype(np.uint8))
        image = image.resize(
            (target_shape[1], target_shape[0]),
            resample=Image.Resampling.NEAREST,
        )
        mask = np.asarray(image)

    unique_values = np.unique(mask)

    # Labelme mask 或 npy mask：0/1 類別 ID。
    if np.all(np.isin(unique_values, [0, 1])):
        return mask.astype(np.uint8)

    # 一般黑白 PNG：0/255。
    return (mask >= 128).astype(np.uint8)


def calculate_metrics(prediction, ground_truth):
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: prediction={prediction.shape}, "
            f"ground_truth={ground_truth.shape}"
        )

    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    eps = 1e-12

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "IoU": tp / (tp + fp + fn + eps),
        "Dice": 2.0 * tp / (2.0 * tp + fp + fn + eps),
        "Precision": tp / (tp + fp + eps),
        "Recall": tp / (tp + fn + eps),
        "Specificity": tn / (tn + fp + eps),
        "Pixel Accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "Predicted positive ratio": float(prediction.mean()),
        "Ground-truth positive ratio": float(ground_truth.mean()),
    }


def evaluate_method(name, prediction, ground_truth, allow_global_flip=False):
    original = calculate_metrics(prediction, ground_truth)
    selected_prediction = prediction
    flipped = False
    selected = original

    if allow_global_flip:
        flipped_prediction = 1 - prediction
        flipped_metrics = calculate_metrics(flipped_prediction, ground_truth)
        if flipped_metrics["IoU"] > original["IoU"]:
            selected_prediction = flipped_prediction
            selected = flipped_metrics
            flipped = True

    return {"Method": name, "Globally flipped": flipped, **selected}, selected_prediction


def save_csv(rows, output_path):
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_results(rows):
    print("\nEvaluation results")
    print("=" * 110)
    print(
        f"{'Method':28s}{'IoU':>10s}{'Dice':>10s}{'Precision':>12s}"
        f"{'Recall':>10s}{'Accuracy':>12s}{'Flipped':>10s}"
    )
    print("-" * 110)
    for row in rows:
        print(
            f"{row['Method'][:28]:28s}"
            f"{row['IoU']:10.4f}"
            f"{row['Dice']:10.4f}"
            f"{row['Precision']:12.4f}"
            f"{row['Recall']:10.4f}"
            f"{row['Pixel Accuracy']:12.4f}"
            f"{str(row['Globally flipped']):>10s}"
        )


def save_comparison_figure(ground_truth, predictions, output_path):
    cmap = ListedColormap(["#2E8B57", "#C2B280"])
    total = 1 + len(predictions)
    columns = min(3, total)
    rows = int(np.ceil(total / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows))
    axes = np.atleast_1d(axes).ravel()

    axes[0].imshow(ground_truth, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title("Ground Truth\n0=vegetation, 1=non-vegetation")
    axes[0].axis("off")

    for index, (name, mask) in enumerate(predictions, start=1):
        axes[index].imshow(mask, cmap=cmap, vmin=0, vmax=1)
        axes[index].set_title(name)
        axes[index].axis("off")

    for index in range(total, len(axes)):
        axes[index].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_error_maps(ground_truth, predictions, output_dir):
    error_cmap = ListedColormap([
        "#2E8B57",  # correct vegetation
        "#FF8C00",  # false positive
        "#8A2BE2",  # false negative
        "#C2B280",  # true positive
    ])

    for name, prediction in predictions:
        gt = ground_truth.astype(bool)
        pred = prediction.astype(bool)

        error_map = np.zeros_like(ground_truth, dtype=np.uint8)
        error_map[np.logical_and(pred, ~gt)] = 1
        error_map[np.logical_and(~pred, gt)] = 2
        error_map[np.logical_and(pred, gt)] = 3

        safe_name = name.lower().replace(" ", "_").replace("+", "plus").replace("/", "_")
        plt.figure(figsize=(6, 6))
        plt.imshow(error_map, cmap=error_cmap, vmin=0, vmax=3)
        plt.title(f"Error map: {name}\norange=FP, purple=FN")
        plt.axis("off")
        plt.savefig(
            os.path.join(output_dir, f"{safe_name}_error_map.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate vegetation/non-vegetation segmentation masks. "
            "Convention: 0=vegetation, 1=non-vegetation."
        )
    )
    parser.add_argument("--gt", required=True, help="Ground-truth Labelme label.png or .npy mask.")
    parser.add_argument("--hsv", help="HSV/soil-score baseline mask (.npy or image).")
    parser.add_argument("--raw", help="Raw Q-Seg mask (.npy or image).")
    parser.add_argument("--overlap", help="Overlap-aligned Q-Seg mask.")
    parser.add_argument("--soil", help="Soil-score-aligned Q-Seg mask.")
    parser.add_argument(
        "--allow_global_flip",
        action="store_true",
        help=(
            "Also evaluate 1-mask and retain the orientation with the higher IoU. "
            "Use only for label-invariant analysis."
        ),
    )
    parser.add_argument("--output_dir", default="evaluation_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ground_truth = load_mask(args.gt)

    print("Ground-truth shape:", ground_truth.shape)
    print("Ground-truth values:", np.unique(ground_truth, return_counts=True))
    print("Positive class: non-vegetation = 1")

    method_paths = [
        ("HSV baseline", args.hsv),
        ("Q-Seg raw", args.raw),
        ("Q-Seg + overlap alignment", args.overlap),
        ("Q-Seg + soil alignment", args.soil),
    ]

    rows = []
    predictions = []

    for name, path in method_paths:
        if not path:
            continue
        prediction = load_mask(path, target_shape=ground_truth.shape)
        print(f"\n{name}")
        print("Path:", path)
        print("Values:", np.unique(prediction, return_counts=True))

        result, selected_prediction = evaluate_method(
            name,
            prediction,
            ground_truth,
            allow_global_flip=args.allow_global_flip,
        )
        rows.append(result)
        predictions.append((name, selected_prediction))

    if not rows:
        raise ValueError("Provide at least one of --hsv, --raw, --overlap, or --soil.")

    print_results(rows)

    csv_path = os.path.join(args.output_dir, "segmentation_metrics.csv")
    comparison_path = os.path.join(args.output_dir, "segmentation_comparison.png")

    save_csv(rows, csv_path)
    save_comparison_figure(ground_truth, predictions, comparison_path)
    save_error_maps(ground_truth, predictions, args.output_dir)

    print("\nSaved:")
    print("-", csv_path)
    print("-", comparison_path)
    print("- one error map per method")


if __name__ == "__main__":
    main()
