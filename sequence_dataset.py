import os
import numpy as np
from torch.utils.data import Dataset

class DirectionSequenceDataset(Dataset):
    def __init__(self, root_dir, max_frames=16):
        self.root_dir = root_dir
        self.samples = []
        self.max_frames = max_frames

        label_map = {"toward": 0, "away": 1, "stationary": 2}
        for label_name, label_id in label_map.items():
            label_folder = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_folder):
                continue
            for file in os.listdir(label_folder):
                if file.endswith(".npz"):
                    self.samples.append({
                        "path": os.path.join(label_folder, file),
                        "label": label_id
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = np.load(sample["path"])
        sequence = data["sequence"]  # shape (T, 3, 224, 224)
        label = int(data["label"])

        sequence = sequence.astype(np.float32) / 255.0
        T, C, H, W = sequence.shape

        if T < self.max_frames:
            pad = np.zeros((self.max_frames - T, C, H, W), dtype=np.float32)
            sequence = np.concatenate([sequence, pad], axis=0)
        elif T > self.max_frames:
            sequence = sequence[:self.max_frames]

        return sequence, label
