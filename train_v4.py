#!/usr/bin/env python3
"""
train_v4.py — Drone RF classifier v4 (GPU, full dataset, dual-modality).

Improvements over train_v3.py:
  - GPU training (v3 ran on CPU, early-stopped at val=0.772)
  - FULL dataset (98,705) via WeightedRandomSampler; no MAX_PER_CLASS cap,
    so DJI (2,194 samples) is seen as often as the rest
  - Dual-modality input: raw IQ [2,16384] + precomputed spectrogram [2,128,128]
  - Augmentation: SNR-matched noise, random gain, phase rotation, time-shift, mixup
  - Warmup + cosine schedule, gradient clipping, early stopping on val acc
  - Saves state_dict (portable) + full module + eval report
Classes: 0=DJI 1=FutabaT14 2=FutabaT7 3=Graupner 4=Noise 5=Taranis 6=Turnigy
"""
import time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from collections import Counter

DATA_PATH = "/work/data/kaggle_drone_rf/dataset.pt"
OUT_STRICT = "/work/best_model_v4_state.pt"
OUT_FULL   = "/work/best_model_v4.pt"
REPORT     = "/work/train_v4_report.json"
EPOCHS = 35
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-4
SEED = 42
VAL_FRAC = 0.1
MIXUP_P = 0.2
WARMUP_EPOCHS = 2
PATIENCE = 8
N_WORKERS = 0

torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device: {device}", flush=True)
CLASS_NAMES = ["DJI", "FutabaT14", "FutabaT7", "Graupner", "Noise", "Taranis", "Turnigy"]
NUM_C = len(CLASS_NAMES)

class DroneDS(Dataset):
    def __init__(self, x_iq, x_spec, y):
        self.x = x_iq
        self.spec = x_spec
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        iq = self.x[i]
        spec = self.spec[i]
        if torch.rand(1).item() < 0.30:  # occasionally drop spectrogram (robustness)
            spec = torch.zeros_like(spec)
        return iq, spec, self.y[i].item()

def augment(iq):
    if torch.rand(1).item() < 0.85:  # SNR-matched noise
        sig_pow = iq.pow(2).mean()
        target = 10 ** (float(torch.randint(0, 21, (1,)).item() - 10) / 20.0)
        noise = torch.randn_like(iq) * (sig_pow / target).sqrt()
        iq = iq + noise
    iq = iq * (0.85 + 0.3 * torch.rand(1).item())
    if torch.rand(1).item() < 0.5:
        th = float(torch.rand(1).item()) * 2 * 3.14159265
        c, s = np.cos(th), np.sin(th)
        i_ch = iq[:, 0]; q_ch = iq[:, 1]
        iq = torch.stack([i_ch * c - q_ch * s, i_ch * s + q_ch * c], dim=1)
    if torch.rand(1).item() < 0.5:
        shift = int(torch.randint(1, max(2, iq.size(-1) // 8), (1,)).item())
        iq = torch.roll(iq, shift, dims=-1)
    return iq

class DroneV4(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.iq = nn.Sequential(
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
            nn.AdaptiveAvgPool1d(1),
        )
        self.spec = nn.Sequential(
            nn.Conv2d(2, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.1),
            nn.Conv2d(32, 64, 5, padding=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.15),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
    def forward(self, iq, spec):
        a = self.iq(iq).flatten(1)
        b = self.spec(spec).flatten(1)
        return self.head(torch.cat([a, b], dim=1))

def main():
    t0 = time.time()
    print("loading dataset (25GB)...", flush=True)
    ds = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    x_iq, x_spec, y = ds["x_iq"], ds["x_spec"], ds["y"]
    print(f"loaded in {time.time()-t0:.1f}s: iq={tuple(x_iq.shape)} spec={tuple(x_spec.shape)}", flush=True)
    print("class counts:", dict(Counter(y.tolist())), flush=True)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    val_n = int(len(y) * VAL_FRAC)
    val_idx, train_idx = list(perm[:val_n]), list(perm[val_n:])
    full = DroneDS(x_iq, x_spec, y)
    train_ds, val_ds = Subset(full, train_idx), Subset(full, val_idx)
    print(f"train: {len(train_idx)}  val: {len(val_idx)}  (FULL, no per-class cap)", flush=True)

    counts = Counter(y[i].item() for i in train_idx)
    raw = np.array([1.0 / np.sqrt(max(1, counts.get(c, 1))) for c in range(NUM_C)], dtype=np.float32)
    sw = raw / raw.sum() * len(train_idx)
    sampler = WeightedRandomSampler(sw.tolist(), num_samples=len(train_idx), replacement=True)
    cw = torch.ones(NUM_C, dtype=torch.float32).to(device)  # uniform loss; sampler handles balance
    print("sample weights:", {CLASS_NAMES[c]: round(float(sw[c]), 2) for c in range(NUM_C)}, flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=N_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS, pin_memory=True)

    model = DroneV4(NUM_C).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS
    def lr_step(step):
        if step < warmup_steps:
            return 0.1 + 0.9 * step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(3.14159265 * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_step)
    criterion = nn.CrossEntropyLoss(weight=cw)

    best_val_acc, best_epoch, patience_counter = -1.0, 0, 0
    history = []
    for epoch in range(EPOCHS):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for iq, spec, yb in train_loader:
            iq, spec, yb = iq.to(device), spec.to(device), yb.to(device)
            iq = augment(iq)
            optimizer.zero_grad()
            if torch.rand(1).item() < MIXUP_P and iq.size(0) > 1:
                lam = float(np.random.beta(0.6, 0.6))
                idx = torch.randperm(iq.size(0), device=device)
                iq_m = lam * iq + (1 - lam) * iq[idx]
                spec_m = lam * spec + (1 - lam) * spec[idx]
                logits = model(iq_m, spec_m)
                loss = lam * criterion(logits, yb) + (1 - lam) * criterion(logits, yb[idx])
            else:
                logits = model(iq, spec)
                loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            t_loss += loss.item() * iq.size(0)
            t_correct += (logits.argmax(1) == yb).sum().item()
            t_total += iq.size(0)

        model.eval()
        pred_all, label_all = [], []
        with torch.no_grad():
            for iq, spec, yb in val_loader:
                iq, spec, yb = iq.to(device), spec.to(device), yb.to(device)
                pred = model(iq, spec).argmax(1)
                pred_all.extend(pred.cpu().tolist())
                label_all.extend(yb.cpu().tolist())
        val_acc = float(np.mean(np.array(pred_all) == np.array(label_all)))

        tp = np.zeros(NUM_C); fp = np.zeros(NUM_C); fn = np.zeros(NUM_C)
        for p, l in zip(pred_all, label_all):
            if p == l: tp[l] += 1
            else: fp[p] += 1; fn[l] += 1
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        macro_f1 = float(f1.mean())
        print(f"epoch {epoch+1:2d}/{EPOCHS} | loss={t_loss/t_total:.4f} train={t_correct/t_total:.4f} "
              f"val={val_acc:.4f} macroF1={macro_f1:.4f} lr={scheduler.get_last_lr()[0]:.6f}", flush=True)
        print("  val per-class:", " ".join(f"{CLASS_NAMES[c]}:P{prec[c]:.2f}/R{rec[c]:.2f}/F1{f1[c]:.2f}" for c in range(NUM_C)), flush=True)
        history.append({"epoch": epoch + 1, "train_acc": float(t_correct/t_total), "val_acc": val_acc, "macro_f1": macro_f1,
                        "per_class_f1": {CLASS_NAMES[c]: float(f1[c]) for c in range(NUM_C)}})

        if macro_f1 > best_val_acc:
            best_val_acc, patience_counter, best_epoch = macro_f1, 0, epoch + 1
            torch.save(model.state_dict(), OUT_STRICT)
            torch.save(model, OUT_FULL)
            with open(REPORT, "w") as f:
                json.dump({"best_val_acc": val_acc, "best_macro_f1": macro_f1, "best_epoch": best_epoch,
                           "class_names": CLASS_NAMES, "params": int(sum(p.numel() for p in model.parameters())),
                           "per_class_f1": {CLASS_NAMES[c]: float(f1[c]) for c in range(NUM_C)},
                           "history": history}, f, indent=2)
            print(f"  *** best saved (val={val_acc:.4f}, macroF1={macro_f1:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  early stop at epoch {epoch+1} (patience={PATIENCE})", flush=True)
                break

    print(f"\nDONE. best_val_acc={best_val_acc:.4f} best_epoch={best_epoch} -> {OUT_STRICT} / {OUT_FULL}", flush=True)

if __name__ == "__main__":
    main()
