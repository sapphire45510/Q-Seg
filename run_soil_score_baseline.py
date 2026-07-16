import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from matplotlib.colors import rgb_to_hsv, ListedColormap

def circular_hue_distance(H, center):
    d = np.abs(H - center)
    return np.minimum(d, 1.0 - d)


def load_full_image_hsv(image_path, target_size=None, median_size=5):
    img = Image.open(image_path).convert("RGB")

    if median_size and median_size > 1:
        img = img.filter(ImageFilter.MedianFilter(size=median_size))

    if target_size is not None:
        img = img.resize((target_size, target_size), resample=Image.Resampling.BOX)

    rgb = np.array(img).astype(float) / 255.0
    hsv = rgb_to_hsv(rgb)
    return rgb, hsv


def compute_soil_score(hsv):
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    soil_dist = circular_hue_distance(H, 0.10)
    green_dist = circular_hue_distance(H, 1.0 / 3.0)

    soil_score = (
        -soil_dist
        + 0.35 * S
        + 0.20 * V
        + 0.35 * green_dist
    )

    return soil_score


def soil_score_segmentation(soil_score, target_ratio=0.3):
    threshold = np.quantile(soil_score, 1.0 - target_ratio)
    mask = (soil_score >= threshold).astype(np.uint8)
    return mask, threshold


def make_boundary_overlay(rgb, mask):
    h, w = mask.shape
    overlay = rgb.copy()
    boundary = np.zeros_like(mask, dtype=bool)

    for y in range(h):
        for x in range(w):
            if y + 1 < h and mask[y, x] != mask[y + 1, x]:
                boundary[y, x] = True
                boundary[y + 1, x] = True
            if x + 1 < w and mask[y, x] != mask[y, x + 1]:
                boundary[y, x] = True
                boundary[y, x + 1] = True

    overlay[boundary] = [1.0, 0.0, 0.0]
    return overlay


def save_results(rgb, soil_score, mask, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    seg_cmap = ListedColormap(["#2E8B57", "#C2B280"])
    # 0 = green, 1 = soil/tan

    plt.figure()
    plt.imshow(rgb)
    plt.title("Input image")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "input.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(soil_score, cmap="viridis")
    plt.title("Soil score map")
    plt.axis("off")
    #plt.colorbar()
    plt.savefig(os.path.join(output_dir, "soil_score_map.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(mask, cmap=seg_cmap, vmin=0, vmax=1)
    plt.title("Soil-score baseline mask")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "soil_score_baseline_mask.png"), dpi=300, bbox_inches="tight")
    plt.close()

    overlay = make_boundary_overlay(rgb, mask)
    plt.figure()
    plt.imshow(overlay)
    plt.title("Boundary overlay: soil-score baseline")
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, "soil_score_boundary_overlay.png"), dpi=300, bbox_inches="tight")
    plt.close()

    np.save(os.path.join(output_dir, "soil_score.npy"), soil_score)
    np.save(os.path.join(output_dir, "soil_score_baseline_mask.npy"), mask)


def main():
    parser = argparse.ArgumentParser(description="Traditional soil-score segmentation baseline.")
    parser.add_argument("--image_path", type=str, default="71619_sat_26.jpg")
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument("--median_size", type=int, default=5)
    parser.add_argument("--target_ratio", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str, default="soil_score_baseline_results")
    args = parser.parse_args()

    target_size = None if args.target_size == 0 else args.target_size

    rgb, hsv = load_full_image_hsv(
        args.image_path,
        target_size=target_size,
        median_size=args.median_size,
    )

    soil_score = compute_soil_score(hsv)

    mask, threshold = soil_score_segmentation(
        soil_score,
        target_ratio=args.target_ratio,
    )

    print("Image shape:", rgb.shape)
    print("Target ratio:", args.target_ratio)
    print("Threshold:", threshold)
    print("Mask ratio:", float(np.mean(mask)))
    print("Unique mask values:", np.unique(mask, return_counts=True))

    save_results(rgb, soil_score, mask, args.output_dir)

    print("Saved results to:", args.output_dir)


if __name__ == "__main__":
    main()