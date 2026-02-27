import os
import h5py
import numpy as np
import random
import argparse

import pandas as pd
from tqdm import tqdm
import cv2


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def get_volume_id(filename):
    return filename.split("_slice_")[0]  # volume_1

def split_volumes(file_list, train_ratio=0.7, val_ratio=0.15):
    volumes = sorted(list(set([get_volume_id(f) for f in file_list])))
    random.shuffle(volumes)

    train_split_idx = int(len(volumes) * train_ratio)
    val_split_idx = train_split_idx + int(len(volumes) * val_ratio)

    train_volumes = volumes[:train_split_idx]
    val_volumes = volumes[train_split_idx:val_split_idx]
    test_volumes = volumes[val_split_idx:]

    return train_volumes, val_volumes, test_volumes

def normalize_image(img):
    img = (img - np.mean(img)) / (np.std(img) + 1e-8)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img



def main(data_dir, output_dir, max_files=None):

    set_seed()

    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".h5")])
    print(f"Found {len(files)} files in {data_dir}")

    if max_files is not None:
        try:
            max_files = int(max_files)
        except (ValueError, TypeError):
            max_files = None

    if max_files is not None:
        files = files[:max_files]

    train_volumes, val_volumes, test_volumes = split_volumes(files, train_ratio=0.7, val_ratio=0.15)
    print(f"Splitting by patient into {len(train_volumes)} train, {len(val_volumes)} val, and {len(test_volumes)} test volumes.")

    metadata = []
    saved_count = 0

    for f in tqdm(files):

        volume_id = get_volume_id(f)

        subset = ""
        if volume_id in train_volumes:
            subset = "train"
        elif volume_id in val_volumes:
            subset = "val"
        elif volume_id in test_volumes:
            subset = "test"

        file_path = os.path.join(data_dir, f)

        with h5py.File(file_path, "r") as hf:
            image = hf["image"][:]
            mask = hf["mask"][:]

        image = normalize_image(image)

        # If image has channels, take first channel or convert appropriately
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[..., 0]
        elif image.ndim == 3 and image.shape[2] > 1:
            image = image[..., 0]

        image = cv2.resize(image, (128, 128))

        # Ensure mask is single channel
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)

        tumor_pixels = int(np.sum(mask > 0))

        label = "tumor" if tumor_pixels > 5 else "normal"

        save_dir = os.path.join(output_dir, subset, label)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f.replace(".h5", ".png"))

        # Convert to uint8 before saving
        img_to_save = np.clip(image, 0.0, 1.0)
        img_to_save = (img_to_save * 255.0).round().astype(np.uint8)

        cv2.imwrite(save_path, img_to_save)
        saved_count += 1

        metadata.append({
            "volume_id": volume_id,
            "subset": subset,
            "filename": f,
            "label": label,
            "tumor_pixels": tumor_pixels
        })

    df = pd.DataFrame(metadata)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    print(f"Preprocessing complete. Saved {saved_count} images to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess H5 slices into PNGs and split by patient volume.")
    parser.add_argument("data_dir", nargs="?", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "brats"), help="Directory containing .h5 files (default: data/raw/brats)")
    parser.add_argument("output_dir", nargs="?", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "brats"), help="Output directory to save processed images (default: data/processed/brats)")
    parser.add_argument("--max-files", dest="max_files", default=None, help="Optional: maximum number of files to process (for quick tests)")

    args = parser.parse_args()

    print(f"Data dir: {args.data_dir}\nOutput dir: {args.output_dir}\nMax files: {args.max_files}")

    main(args.data_dir, args.output_dir, args.max_files)