import os
import shutil

# Các đường dẫn
base_path = r"E:\In\R1\transport\train_balanced_sample_movements_fill_blobs(2025-06-27)"
pairs = [
    ("P1", "UnknownP1"),
    ("P2", "UnknownP2")
]

for p_folder, unknown_folder in pairs:
    path_p = os.path.join(base_path, p_folder)
    path_unknown = os.path.join(base_path, unknown_folder)

    if not os.path.isdir(path_p) or not os.path.isdir(path_unknown):
        print(f" One of the paths doesn't exist: {path_p} or {path_unknown}")
        continue

    # Lấy danh sách folder con
    p_folders = set(os.listdir(path_p))
    unknown_folders = set(os.listdir(path_unknown))

    # Tìm folder trùng lặp
    duplicate_folders = p_folders & unknown_folders

    print(f"🔍 Checking {p_folder} vs {unknown_folder}: {len(duplicate_folders)} duplicates found.")

    # Xoá folder trùng trong P*
    for folder in duplicate_folders:
        full_path = os.path.join(path_p, folder)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
            print(f" Deleted {full_path}")

print(" Done.")
