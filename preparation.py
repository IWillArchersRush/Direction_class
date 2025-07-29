import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# Thư mục chứa dữ liệu gốc
root_dirs = [
    r"E:\In\R1\transport\train_balanced_sample_movements_fill_blobs(2025-06-27)\P1",
    r"E:\In\R1\transport\train_balanced_sample_movements_fill_blobs(2025-06-27)\P2"
]

# Nhãn hợp lệ
valid_labels = {"toward": 0, "away": 1, "stationary": 2}

# Output directory
output_dir = r"E:\In\R1\direction_sequences"
os.makedirs(output_dir, exist_ok=True)

# Resize input size
img_size = (224, 224)

for root in root_dirs:
    subfolders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]

    for folder in tqdm(subfolders, desc=f"Processing {root}"):
        folder_path = os.path.join(root, folder)
        anno_path = os.path.join(folder_path, "annotations.json")

        if not os.path.exists(anno_path):
            print(f"⚠️ Missing annotations.json in {folder_path}")
            continue

        with open(anno_path, "r") as f:
            data = json.load(f)

        label_name = data.get("label", "unknown").strip().lower()
        if label_name not in valid_labels:
            print(f"⛔ Skipping unknown or invalid label: {label_name}")
            continue
        label_idx = valid_labels[label_name]

        annotations = data.get("annotations", {})
        sequence = []

        for img_name in sorted(annotations.keys()):
            img_path = os.path.join(folder_path, img_name)
            if not os.path.exists(img_path):
                continue

            bbox = annotations[img_name]
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            resized = cv2.resize(crop, img_size)
            resized = resized[:, :, ::-1]  # BGR to RGB
            tensor = resized.transpose(2, 0, 1)  # HWC to CHW
            sequence.append(tensor)

        if len(sequence) == 0:
            print(f"⚠️ No valid frames in {folder_path}")
            continue

        # Save as npz
        sequence_np = np.stack(sequence)  # (T, C, H, W)
        label_folder = os.path.join(output_dir, label_name)
        os.makedirs(label_folder, exist_ok=True)
        save_path = os.path.join(label_folder, folder + ".npz")
        np.savez_compressed(save_path, sequence=sequence_np, label=label_idx)

print("✅ DONE. Sequences saved to:", output_dir)
