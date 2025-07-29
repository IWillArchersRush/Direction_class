import os
import uuid
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from sklearn.metrics import accuracy_score
import wandb

from sequence_dataset import DirectionSequenceDataset
from model import DirectionClassifier

# ========== GHI LOG RA FILE + TERMINAL ==========
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger("log5.txt")

# ========== GLOBAL SEED ==========
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ========== CONFIG ==========
root_dir = r"E:\In\R1\direction_sequences"
batch_size = 4  # 🔧 Giảm batch size để tránh lỗi RAM
epochs = 250
lr = 1e-4
max_frames = 16
checkpoint_path = "checkpoint.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== WANDB RESUME SETUP ==========
run_id_file = "wandb_run_id.txt"

if os.path.exists(run_id_file):
    with open(run_id_file, "r") as f:
        wandb_run_id = f.read().strip()
    resume_mode = "allow"
else:
    wandb_run_id = str(uuid.uuid4())
    with open(run_id_file, "w") as f:
        f.write(wandb_run_id)
    resume_mode = None  # chạy mới hoàn toàn

# ========== INIT WANDB ==========
wandb.init(
    project="direction-classifier222",
    name="directional_detection",
    id=wandb_run_id if resume_mode else None,
    resume=resume_mode,
    config={
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "model": "MobileNetV2 + TemporalPooling",
        "input_size": (224, 224),
        "sequence_level": True,
        "max_frames": max_frames
    }
)

# ========== LOAD DATA ==========
dataset = DirectionSequenceDataset(root_dir, max_frames=max_frames)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(SEED)
train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True,
    drop_last=True, pin_memory=False, num_workers=0
)
val_loader = DataLoader(
    val_set, batch_size=batch_size, shuffle=False,
    pin_memory=False, num_workers=0
)

# ========== MODEL, LOSS, OPTIM ==========
model = DirectionClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ========== RESUME FROM CHECKPOINT ==========
start_epoch = 0
best_acc = 0.0

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    best_acc = checkpoint["best_acc"]
    print(f"🔁 Resuming training from epoch {start_epoch} (best val_acc: {best_acc:.4f})")
else:
    print("🚀 Starting training from scratch.")

# ========== TRAIN LOOP ==========
for epoch in range(start_epoch, epochs):
    print(f"\n🟡 Epoch {epoch + 1}/{epochs}...")

    model.train()
    total_loss = 0

    for sequences, labels in train_loader:
        sequences = sequences.clone().detach().to(device)
        labels = labels.to(device)

        outputs = model(sequences)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # ===== EVAL =====
    model.eval()
    all_preds, all_labels = [], []
    val_loss_total = 0

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.clone().detach().to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            val_loss_total += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_val_loss = val_loss_total / len(val_loader)
    acc = accuracy_score(all_labels, all_preds)
    current_lr = optimizer.param_groups[0]['lr']

    print(f" Epoch {epoch+1:03d} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.4f}")

    wandb.log({
        "epoch": epoch + 1,
        "learning_rate": current_lr,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_accuracy": acc
    })

    # ===== Save checkpoint every epoch =====
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_acc": best_acc
    }, checkpoint_path)

    # ===== Save best model only if val acc improves =====
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_direction_model.pt")
        print("  Saved best model (val_acc ↑)")

print(f"\n Training complete. Best validation accuracy: {best_acc:.4f}")
