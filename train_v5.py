#!/usr/bin/env python3
"""
train_v5.py — Drone RF classifier v5 (GPU, full dataset, proven v3 recipe).

Same architecture as train_v3.py (DroneCNN, IQ-only) but with the fixes that
made v3 hit 0.77 val:
  - FULL dataset (98,705), no per-class cap -> DJI (2,194) seen as often as rest
  - Inverse-count class-weighted CrossEntropy (mild, not double-weighted)
  - Cosine LR schedule, gradient clipping, early stopping on val acc
  - GPU (v3 ran on CPU; this is the same model ~10x faster)
Classes: 0=DJI 1=FutabaT14 2=FutabaT7 3=Graupner 4=Noise 5=Taranis 6=Turnigy
"""
import time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from collections import Counter

DATA_PATH  = "/work/data/kaggle_drone_rf/dataset.pt"
OUT_STRICT = "/work/best_model_v5_state.pt"
OUT_FULL   = "/work/best_model_v5.pt"
REPORT     = "/work/train_v5_report.json"
EPOCHS = 40
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
VAL_FRAC = 0.1
PATIENCE = 8

torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}", flush=True)
CLASS_NAMES = ["DJI", "FutabaT14", "FutabaT7", "Graupner", "Noise", "Taranis", "Turnigy"]
NUM_C = len(CLASS_NAMES)

class DroneDS(Dataset):
    def __init__(self, x_iq, y):
        self.x = x_iq; self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.x[i], self.y[i].item()

class DroneCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, 15, stride=2, padding=7), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, 11, padding=5), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(32, 64, 11, stride=2, padding=5), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.15),
            nn.Conv1d(64, 128, 7, stride=2, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2),
            nn.Conv1d(128, 256, 5, stride=2, padding=2), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))

def main():
    t0 = time.time()
    ds = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    x_iq, y = ds["x_iq"], ds["y"]
    print(f"loaded {tuple(x_iq.shape)} in {time.time()-t0:.1f}s", flush=True)
    print("class counts:", dict(Counter(y.tolist())), flush=True)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    val_n = int(len(y) * VAL_FRAC)
    val_idx, train_idx = list(perm[:val_n]), list(perm[val_n:])
    full = DroneDS(x_iq, y)
    train_ds, val_ds = Subset(full, train_idx), Subset(full, val_idx)
    print(f"train: {len(train_idx)}  val: {len(val_idx)}  (FULL, no cap)", flush=True)

    counts = Counter(y[i].item() for i in train_idx)
    cw = torch.tensor([1.0 / max(1, counts.get(c, 1)) for c in range(NUM_C)], dtype=torch.float32)
    cw = cw / cw.sum()
    print("loss class weights:", {CLASS_NAMES[c]: round(float(cw[c]), 4) for c in range(NUM_C)}, flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DroneCNN(NUM_C).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=cw.to(device))

    best_val_acc, best_epoch, patience_counter = -1.0, 0, 0
    history = []
    for epoch in range(EPOCHS):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for x, yb in train_loader:
            x, yb = x.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            t_loss += loss.item() * x.size(0)
            t_correct += (logits.argmax(1) == yb).sum().item()
            t_total += x.size(0)
        model.eval()
        pred_all, label_all = [], []
        with torch.no_grad():
            for x, yb in val_loader:
                x, yb = x.to(device), yb.to(device)
                pred = model(x).argmax(1)
                pred_all.extend(pred.cpu().tolist()); label_all.extend(yb.cpu().tolist())
        val_acc = float(np.mean(np.array(pred_all) == np.array(label_all)))
        tp = np.zeros(NUM_C); fp = np.zeros(NUM_C); fn = np.zeros(NUM_C)
        for p, l in zip(pred_all, label_all):
            if p == l: tp[l] += 1
            else: fp[p] += 1; fn[l] += 1
        prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9); f1 = 2*prec*rec/(prec+rec+1e-9)
        macro_f1 = float(f1.mean())
        print(f"epoch {epoch+1:2d}/{EPOCHS} | loss={t_loss/t_total:.4f} train={t_correct/t_total:.4f} "
              f"val={val_acc:.4f} macroF1={macro_f1:.4f} lr={scheduler.get_last_lr()[0]:.6f}", flush=True)
        print("  val per-class:", " ".join(f"{CLASS_NAMES[c]}:P{prec[c]:.2f}/R{rec[c]:.2f}/F1{f1[c]:.2f}" for c in range(NUM_C)), flush=True)
        history.append({"epoch": epoch+1, "train_acc": float(t_correct/t_total), "val_acc": val_acc,
                        "macro_f1": macro_f1, "per_class_f1": {CLASS_NAMES[c]: float(f1[c]) for c in range(NUM_C)}})
        if val_acc > best_val_acc:
            best_val_acc, patience_counter, best_epoch = val_acc, 0, epoch + 1
            torch.save(model.state_dict(), OUT_STRICT); torch.save(model, OUT_FULL)
            with open(REPORT, "w") as f:
                json.dump({"best_val_acc": val_acc, "best_macro_f1": macro_f1, "best_epoch": best_epoch,
                           "class_names": CLASS_NAMES, "params": int(sum(p.numel() for p in model.parameters())),
                           "per_class_f1": {CLASS_NAMES[c]: float(f1[c]) for c in range(NUM_C)},
                           "history": history}, f, indent=2)
            print(f"  *** best saved (val={val_acc:.4f}, macroF1={macro_f1:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  early stop at epoch {epoch+1} (patience={PATIENCE})", flush=True); break
    print(f"\nDONE. best_val_acc={best_val_acc:.4f} best_epoch={best_epoch} -> {OUT_STRICT} / {OUT_FULL}", flush=True)

if __name__ == "__main__":
    main()
