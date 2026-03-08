"""
convert_kitti.py
────────────────
Converts KITTI object detection labels → YOLO format and
organises images + labels into train / val splits.

KITTI label format (15 columns per line):
  type | truncated | occluded | alpha |
  x1 y1 x2 y2 (bbox in pixels) |
  h w l | x y z | rotation_y

YOLO label format (5 columns, normalised 0-1):
  class_id  cx  cy  w  h

Usage:
  python convert_kitti.py --kitti_images  path/to/training/image_2
                          --kitti_labels  path/to/training/label_2
                          --output_dir    data
                          --val_split     0.2
                          --img_w         1242
                          --img_h         375

Here's what the script handles automatically:

Converts all 7481 KITTI labels from 15-column pixel format → 5-column normalised YOLO format
Skips DontCare annotations (KITTI's ignore regions — you don't want to train on these)
Skips degenerate boxes (zero-size or malformed annotations)
80/20 train/val split with a fixed random seed so it's reproducible
Copies images into the correct folder structure that Cell 3 in train.ipynb expects
"""

import os
import shutil
import random
import argparse
from pathlib import Path


# ── KITTI class name → YOLO index ───────────────────────────
KITTI_CLASSES = {
    'Car':             0,
    'Van':             1,
    'Truck':           2,
    'Pedestrian':      3,
    'Person_sitting':  4,
    'Cyclist':         5,
    'Tram':            6,
    'Misc':            7,
}

# Classes to skip entirely (background / ignore)
SKIP_CLASSES = {'DontCare'}


def convert_kitti_box_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """
    Convert pixel-space x1y1x2y2 → normalised cxcywh.
    Clamps values to [0, 1] to handle edge cases.
    """
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    # Clamp to valid range
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))
    return cx, cy, w, h


def parse_kitti_label(label_path, img_w, img_h):
    """
    Parse one KITTI .txt label file.
    Returns list of YOLO-format strings: 'class_id cx cy w h'
    Skips DontCare and unknown classes.
    """
    yolo_lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            obj_class = parts[0]

            if obj_class in SKIP_CLASSES:
                continue
            if obj_class not in KITTI_CLASSES:
                print(f'  [WARN] Unknown class "{obj_class}" in {label_path.name} — skipping')
                continue

            class_id = KITTI_CLASSES[obj_class]
            x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            cx, cy, w, h = convert_kitti_box_to_yolo(x1, y1, x2, y2, img_w, img_h)

            # Skip boxes that are too small (likely annotation noise)
            if w < 0.001 or h < 0.001:
                continue

            yolo_lines.append(f'{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')

    return yolo_lines


def make_dirs(output_dir):
    """Create the YOLO folder structure."""
    dirs = [
        output_dir / 'images' / 'train',
        output_dir / 'images' / 'val',
        output_dir / 'labels' / 'train',
        output_dir / 'labels' / 'val',
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def convert(kitti_images, kitti_labels, output_dir, val_split, img_w, img_h):
    kitti_images = Path(kitti_images)
    kitti_labels = Path(kitti_labels)
    output_dir   = Path(output_dir)

    # Collect all image stems that also have a label file
    img_extensions = {'.png', '.jpg', '.jpeg'}
    all_stems = sorted([
        f.stem for f in kitti_images.iterdir()
        if f.suffix.lower() in img_extensions
        and (kitti_labels / (f.stem + '.txt')).exists()
    ])

    if not all_stems:
        raise FileNotFoundError(
            f'No matching image+label pairs found.\n'
            f'  Images dir : {kitti_images}\n'
            f'  Labels dir : {kitti_labels}\n'
            f'Check that both paths are correct.'
        )

    print(f'Found {len(all_stems)} image+label pairs.')

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_stems)
    n_val   = int(len(all_stems) * val_split)
    val_stems   = set(all_stems[:n_val])
    train_stems = set(all_stems[n_val:])

    print(f'Split  →  train: {len(train_stems)}  |  val: {len(val_stems)}')

    make_dirs(output_dir)

    converted = skipped = 0

    for stem in all_stems:
        split = 'val' if stem in val_stems else 'train'

        # ── Find image file ──────────────────────────────────
        img_src = None
        for ext in img_extensions:
            candidate = kitti_images / (stem + ext)
            if candidate.exists():
                img_src = candidate
                break
        if img_src is None:
            skipped += 1
            continue

        # ── Convert label ────────────────────────────────────
        label_src  = kitti_labels / (stem + '.txt')
        yolo_lines = parse_kitti_label(label_src, img_w, img_h)

        # Skip images with no valid annotations
        if not yolo_lines:
            skipped += 1
            continue

        # ── Copy image ───────────────────────────────────────
        img_dst = output_dir / 'images' / split / img_src.name
        shutil.copy2(img_src, img_dst)

        # ── Write YOLO label ─────────────────────────────────
        label_dst = output_dir / 'labels' / split / (stem + '.txt')
        with open(label_dst, 'w') as f:
            f.write('\n'.join(yolo_lines))

        converted += 1

    print(f'\n✅ Done!')
    print(f'   Converted : {converted} images')
    print(f'   Skipped   : {skipped} images (no valid labels)')
    print(f'   Output    : {output_dir.resolve()}')
    print(f'\nFolder structure:')
    print(f'   {output_dir}/images/train/  ← {len(train_stems) - skipped} images')
    print(f'   {output_dir}/images/val/    ← {len(val_stems)} images')
    print(f'   {output_dir}/labels/train/')
    print(f'   {output_dir}/labels/val/')

    # ── Class summary ────────────────────────────────────────
    print(f'\nClass mapping used:')
    for name, idx in KITTI_CLASSES.items():
        print(f'   {idx} : {name}')


# ── CLI ──────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KITTI → YOLO label converter')
    parser.add_argument('--kitti_images', required=True,
                        help='Path to KITTI training/image_2 folder')
    parser.add_argument('--kitti_labels', required=True,
                        help='Path to KITTI training/label_2 folder')
    parser.add_argument('--output_dir',   default='data',
                        help='Output root (default: data/)')
    parser.add_argument('--val_split',    type=float, default=0.2,
                        help='Fraction of data for validation (default: 0.2)')
    parser.add_argument('--img_w',        type=int,   default=1242,
                        help='KITTI image width in pixels (default: 1242)')
    parser.add_argument('--img_h',        type=int,   default=375,
                        help='KITTI image height in pixels (default: 375)')
    args = parser.parse_args()

    convert(
        kitti_images = args.kitti_images,
        kitti_labels = args.kitti_labels,
        output_dir   = args.output_dir,
        val_split    = args.val_split,
        img_w        = args.img_w,
        img_h        = args.img_h,
    )