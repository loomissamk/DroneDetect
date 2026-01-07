#!/usr/bin/env python3
# CheckModel.py
# Verify loading of a PyTorch model checkpoint with architecture changes.

import math, pickle, torch
import torch.nn as nn
from torch.nn.modules.container import Sequential, ModuleList
from torch.nn.modules.conv import Conv1d, Conv2d
from torch.nn.modules.activation import ReLU
from torch.nn.modules.rnn import GRU, LSTM, RNN
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveMaxPool1d, AdaptiveAvgPool1d
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.parameter import Parameter

MODEL_PATH = "best_model.pt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Labels: 7 classes (non-drone removed)
drone_labels = ["DJI", "FutabaT14", "FutabaT7", "Graupner", "Taranis", "Turnigy", "Noise"]

# ---- Stubs for old pickled classes ----
class OptimizedTemporalModel(nn.Module):
    def __init__(self, *a, **k): super().__init__()
    def forward(self, *a, **k): raise NotImplementedError

class AttentionLayer(nn.Module):
    """Minimal stub so unpickler can resolve symbol; we won't execute it."""
    def __init__(self, *a, **k): super().__init__()
    def forward(self, *a, **k): raise NotImplementedError

# ---- Our new model ----
class EnhancedTemporalModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, stride=1, padding=2), nn.ReLU(),
            nn.Conv1d(64,128,kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(128,256,kernel_size=9, stride=2, padding=4), nn.ReLU(),
            nn.AdaptiveMaxPool1d(64)
        )
        self.rnn = nn.GRU(input_size=256, hidden_size=256, num_layers=3, batch_first=True, dropout=0.3)
        enc = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, dropout=0.3, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=3)
        self.fc = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.cnn(x).permute(0,2,1)
        x, _ = self.rnn(x)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        return self.fc(x)

# ---- Allow-list everything the pickle might reference ----
ALLOWLIST = [
    OptimizedTemporalModel, AttentionLayer, EnhancedTemporalModel,
    Sequential, ModuleList, Parameter,
    Conv1d, Conv2d, ReLU, GRU, LSTM, RNN, Linear,
    AdaptiveMaxPool1d, AdaptiveAvgPool1d, Dropout,
    TransformerEncoder, TransformerEncoderLayer,
    # module aliases that sometimes appear in pickles
    nn.Conv1d, nn.ReLU, nn.GRU, nn.Linear, nn.AdaptiveMaxPool1d,
    nn.Dropout, nn.TransformerEncoder, nn.TransformerEncoderLayer, nn.Sequential
]
torch.serialization.add_safe_globals(ALLOWLIST)

def normalize_to_state_dict(obj):
    if isinstance(obj, dict) and any(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    raise RuntimeError("Checkpoint is neither a tensor state_dict nor an nn.Module.")

def guess_num_classes_from_state_dict(sd: dict) -> int:
    # Try obvious classifier keys first
    preferred_keys = ["fc.3.weight", "fc.2.weight", "classifier.weight", "head.weight"]
    for k in preferred_keys:
        t = sd.get(k, None)
        if isinstance(t, torch.Tensor) and t.ndim == 2:
            print(f"[loader] Using '{k}' to infer classes: {t.shape[0]}")
            return int(t.shape[0])
    # Fallback heuristic: smallest 2D tensor rows <= 64
    cands = []
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2:
            out = v.shape[0]
            if 2 <= out <= 64:
                cands.append((out, k))
    if not cands:
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                cands.append((v.shape[0], k))
    if not cands:
        raise RuntimeError("Could not infer num_classes (no 2D tensors).")
    cands.sort(key=lambda x: x[0])
    inferred, name = cands[0]
    print(f"[loader] Inferred num_classes={inferred} from '{name}' shape {sd[name].shape}")
    return int(inferred)

def summarize_arch_from_state_dict(sd: dict):
    lines = []
    conv1 = sd.get("cnn.0.weight")
    conv2 = sd.get("cnn.2.weight")
    conv3 = sd.get("cnn.4.weight")
    if isinstance(conv1, torch.Tensor):
        lines.append(f"cnn.0: Conv1d({conv1.shape[1]}->{conv1.shape[0]}, k={conv1.shape[2]})")
    if isinstance(conv2, torch.Tensor):
        lines.append(f"cnn.2: Conv1d({conv2.shape[1]}->{conv2.shape[0]}, k={conv2.shape[2]})")
    if isinstance(conv3, torch.Tensor):
        lines.append(f"cnn.4: Conv1d({conv3.shape[1]}->{conv3.shape[0]}, k={conv3.shape[2]})")

    wih = sd.get("rnn.weight_ih_l0")
    if isinstance(wih, torch.Tensor) and wih.ndim == 2:
        hidden = wih.shape[0] // 3
        inp = wih.shape[1]
        layers = 0
        for k in sd.keys():
            if k.startswith("rnn.weight_ih_l"):
                try:
                    layers = max(layers, int(k.split("rnn.weight_ih_l")[1]) + 1)
                except Exception:
                    pass
        if layers == 0:
            layers = 1
        lines.append(f"rnn: GRU(input_size={inp}, hidden_size={hidden}, num_layers={layers})")

    t_inproj = sd.get("transformer.layers.0.self_attn.in_proj_weight")
    t_ff1 = sd.get("transformer.layers.0.linear1.weight")
    if isinstance(t_inproj, torch.Tensor) and t_inproj.ndim == 2:
        d_model = t_inproj.shape[1]
        dim_ff = t_ff1.shape[0] if isinstance(t_ff1, torch.Tensor) and t_ff1.ndim == 2 else "?"
        layers = 0
        for k in sd.keys():
            if k.startswith("transformer.layers."):
                try:
                    idx = int(k.split(".")[2])
                    layers = max(layers, idx + 1)
                except Exception:
                    pass
        if layers == 0:
            layers = 1
        lines.append(f"transformer: layers={layers}, d_model={d_model}, dim_ff={dim_ff}")

    fc0 = sd.get("fc.0.weight")
    fc3 = sd.get("fc.3.weight")
    if isinstance(fc0, torch.Tensor) and isinstance(fc3, torch.Tensor):
        lines.append(f"fc: Linear({fc0.shape[1]}->{fc0.shape[0]}) -> Linear({fc3.shape[1]}->{fc3.shape[0]})")
    return lines

def main():
    print(f"[info] Loading checkpoint: {MODEL_PATH}")
    with torch.serialization.safe_globals(ALLOWLIST):
        # Try safest first
        try:
            obj = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        except TypeError:
            obj = torch.load(MODEL_PATH, map_location=DEVICE)
        except pickle.UnpicklingError:
            print("[warn] weights_only=True failed; attempting load with weights_only=False (trusted file assumed).")
            obj = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    obj_full = obj if isinstance(obj, nn.Module) else None
    sd = normalize_to_state_dict(obj)
    if obj_full is None:
        try:
            with torch.serialization.safe_globals(ALLOWLIST):
                obj_full = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            if isinstance(obj_full, nn.Module):
                sd = obj_full.state_dict()
            else:
                obj_full = None
        except Exception as exc:
            print(f"[warn] Could not load full module for architecture: {exc}")

    if obj_full is not None:
        print("\n=== Checkpoint Model Architecture ===")
        print(obj_full)
        print("====================================")
    else:
        print("\n=== Inferred Architecture Summary ===")
        for line in summarize_arch_from_state_dict(sd):
            print(line)
        print("Note: stride/padding, dropout, nhead, and batch_first are not in the state_dict.")
        print("=====================================")

    n_out = guess_num_classes_from_state_dict(sd)
    print(f"[info] Building EnhancedTemporalModel(num_classes={n_out}) on {DEVICE}")
    model = EnhancedTemporalModel(n_out).to(DEVICE)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[info] Missing keys: {sorted(missing)}")
    print(f"[info] Unexpected keys: {sorted(unexpected)}")
    model.eval()

    print("\n=== Reference Architecture (EnhancedTemporalModel) ===")
    print(model)
    try:
        attn = model.transformer.layers[0].self_attn
        print(f"[ref] transformer: d_model={attn.embed_dim}, nhead={attn.num_heads}, batch_first={attn.batch_first}")
    except Exception:
        pass
    print("======================================================")

    print("\n=== Model Loaded Successfully ===")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint classes: {n_out}")
    print(f"Active labels (UI): {drone_labels}")
    if n_out != len(drone_labels):
        print("[warn] Output classes != label list length. We removed 'non-drone'; "
              "ensure your old checkpoint was trained on exactly these 7 labels.")
    print("=================================")

if __name__ == "__main__":
    main()
