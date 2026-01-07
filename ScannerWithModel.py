#!/usr/bin/env python3
# ScannerWithModel.py
# Live IQ signal scanner with drone classification using a PyTorch model.

import os
import sys
import csv
import subprocess
from datetime import datetime, timezone
from collections import deque

import numpy as np
from scipy import signal

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QPalette, QColor
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.serialization as tser

# ====================== CONFIG ======================
DEFAULT_START_FREQ = 10e6
DEFAULT_END_FREQ   = 6e9
STEP_SIZE          = 1e6
SAMPLE_RATE        = 20e6
RECORD_SECS        = 0.2
GAIN               = 20
THRESHOLD_DB       = -50
TMP_FILE           = "temp_iq.raw"
CSV_LOG_FILE       = "signal_log.csv"
MODEL_PATH         = "best_model.pt"

# Confidence gates for "confirmed" identification
CONFIDENCE_MIN = 0.90   # require at least 80% top-1 probability
MARGIN_MIN     = 0.50   # require at least 20% margin over #2
TOPK_TO_LOG    = 2      # how many top classes to log/display

# GPU/CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===================== LABELS =======================
# We removed "non-drone" from the UI list (7 classes total).
drone_labels = ["DJI", "FutabaT14", "FutabaT7", "Graupner", "Taranis", "Turnigy", "Noise"]
# In the old 8-class checkpoint, "non-drone" was the last index (7).
NON_DRONE_IDX_IN_OLD = 7

# ===================== MODEL ========================
class EnhancedTemporalModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, stride=1, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4), nn.ReLU(),
            nn.AdaptiveMaxPool1d(64)
        )
        self.rnn = nn.GRU(input_size=256, hidden_size=256, num_layers=3, batch_first=True, dropout=0.3)
        # Use batch_first=True to silence the nested_tensor warning and match our code
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=1024, dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, iq_signal):
        # iq_signal: [B, 2, T]
        x = self.cnn(iq_signal).permute(0, 2, 1)  # [B, L, 256]
        x, _ = self.rnn(x)                        # [B, L, 256]
        x = self.transformer(x)                   # [B, L, 256]
        x = torch.mean(x, dim=1)                  # [B, 256]
        return self.fc(x)                         # [B, C]

# ================== SAFE UNPICKLING =================
# Old pickle referenced custom classes; stub them and allow-list.
class OptimizedTemporalModel(nn.Module):
    def __init__(self, *a, **k): super().__init__()
    def forward(self, *a, **k): raise NotImplementedError

class AttentionLayer(nn.Module):
    def __init__(self, *a, **k): super().__init__()
    def forward(self, *a, **k): raise NotImplementedError

from torch.nn.modules.container import Sequential, ModuleList
from torch.nn.modules.conv import Conv1d, Conv2d
from torch.nn.modules.activation import ReLU
from torch.nn.modules.rnn import GRU, LSTM, RNN
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import AdaptiveMaxPool1d, AdaptiveAvgPool1d
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.parameter import Parameter

ALLOWLIST = [
    OptimizedTemporalModel, AttentionLayer, EnhancedTemporalModel,
    Sequential, ModuleList, Parameter,
    Conv1d, Conv2d, ReLU, GRU, LSTM, RNN, Linear,
    AdaptiveMaxPool1d, AdaptiveAvgPool1d, Dropout,
    TransformerEncoder, TransformerEncoderLayer,
    # common aliases PyTorch pickles sometimes store
    nn.Conv1d, nn.ReLU, nn.GRU, nn.Linear, nn.AdaptiveMaxPool1d,
    nn.Dropout, nn.TransformerEncoder, nn.TransformerEncoderLayer, nn.Sequential
]
tser.add_safe_globals(ALLOWLIST)

def load_checkpoint_as_state_dict(path: str, map_location):
    """
    Loads either a state_dict or a full nn.Module from a possibly pickled checkpoint.
    Always returns a state_dict.
    """
    # Try safest path first (weights_only=True)
    try:
        with tser.safe_globals(ALLOWLIST):
            obj = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older torch without weights_only kwarg
        with tser.safe_globals(ALLOWLIST):
            obj = torch.load(path, map_location=map_location)
    except Exception:
        # Fall back to non-weights-only inside allowlist
        with tser.safe_globals(ALLOWLIST):
            obj = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict) and any(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    raise RuntimeError("Checkpoint is neither a Module nor a tensor state_dict.")

# Build model with 8 outputs to match the checkpoint, then slice at inference.
model = EnhancedTemporalModel(num_classes=8).to(device)
state_dict = load_checkpoint_as_state_dict(MODEL_PATH, device)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"[loader] Missing keys: {sorted(missing)}")
print(f"[loader] Unexpected keys: {sorted(unexpected)}")
model.eval()

def drop_non_drone(logits: torch.Tensor, drop_idx: int = NON_DRONE_IDX_IN_OLD) -> torch.Tensor:
    """Remove the 'non-drone' column from 8-logit output -> 7-logit output."""
    # logits: [B, 8]  -> returns [B, 7]
    if logits.shape[1] != 8:
        return logits  # if you later swap to a true 7-class head, no-op
    return torch.cat([logits[:, :drop_idx], logits[:, drop_idx+1:]], dim=1)

# =================== RUNTIME SIZES ==================
SAMPLES = int(SAMPLE_RATE * RECORD_SECS)   # complex samples (I+Q per sample)
# Note: hackrf_transfer -n expects number of *complex* samples, not bytes.

# ===================== UI/Plot ======================
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(15, 5), facecolor='black')
        self.ax_psd = self.fig.add_subplot(131, facecolor='black')
        self.ax_const = self.fig.add_subplot(132, facecolor='black')
        self.ax_waterfall = self.fig.add_subplot(133, facecolor='black')
        super().__init__(self.fig)
        self.setParent(parent)
        self.waterfall_buffer = deque(maxlen=100)

    def update_plot(self, iq_data, freq_hz):
        self.ax_psd.clear()
        f_axis, Pxx = signal.welch(iq_data, fs=SAMPLE_RATE, nperseg=2048)
        self.ax_psd.plot(f_axis, Pxx, color='lime')
        self.ax_psd.set_title(f"PSD @ {freq_hz/1e6:.2f} MHz", color='white')
        self.ax_psd.set_xlabel("Hz", color='white')
        self.ax_psd.set_ylabel("Power", color='white')
        self.ax_psd.tick_params(colors='white')
        self.ax_psd.grid(True, color='gray')

        self.ax_const.clear()
        iq_down = iq_data[::5]
        self.ax_const.plot(iq_down.real, iq_down.imag, '.', alpha=0.3, color='lime')
        self.ax_const.set_title("IQ Constellation", color='white')
        self.ax_const.set_xlabel("I", color='white')
        self.ax_const.set_ylabel("Q", color='white')
        scale = max(0.1, min(np.max(np.abs(iq_down)) * 1.2, 2.0))
        self.ax_const.set_xlim(-scale, scale)
        self.ax_const.set_ylim(-scale, scale)
        self.ax_const.tick_params(colors='white')
        self.ax_const.grid(True, color='gray')

        self.ax_waterfall.clear()
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(iq_data)))**2
        spectrum_db = 10 * np.log10(spectrum + 1e-12)
        self.waterfall_buffer.append(spectrum_db.astype(np.float32))
        wf_matrix = np.vstack(self.waterfall_buffer)
        self.ax_waterfall.imshow(
            wf_matrix, aspect='auto', origin='lower', cmap='viridis',
            extent=[-SAMPLE_RATE/2, SAMPLE_RATE/2, 0, len(wf_matrix)]
        )
        self.ax_waterfall.set_title("Waterfall", color='white')
        self.ax_waterfall.set_xlabel("Freq Offset (Hz)", color='white')
        self.ax_waterfall.set_ylabel("Time", color='white')
        self.ax_waterfall.tick_params(colors='white')
        self.draw()

# ================== SCAN THREAD =====================
class SignalScanner(QtCore.QThread):
    signal_detected = QtCore.pyqtSignal(np.ndarray, float)
    status_update   = QtCore.pyqtSignal(str)

    def __init__(self, canvas, start_freq=DEFAULT_START_FREQ, end_freq=DEFAULT_END_FREQ):
        super().__init__()
        self.canvas = canvas
        self.running = False
        self.paused = False
        self.last_freq_displayed = None
        self.start_freq = start_freq
        self.end_freq = end_freq

    def run(self):
        self.running = True
        current_freq = self.start_freq

        with open(CSV_LOG_FILE, 'w', newline='') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                "Timestamp","Frequency (Hz)","Power (dB)","Detected",
                "PredClassIdx","PredLabel","Top1Prob","Top2Prob"
            ])

            while self.running:
                if self.paused:
                    self.msleep(100)
                    continue

                if current_freq > self.end_freq:
                    current_freq = self.start_freq

                freq = current_freq
                current_freq += STEP_SIZE
                self.status_update.emit(f"Scanning {freq/1e6:.2f} MHz")
                iq = self.record_iq(freq)
                if iq is None or len(iq) == 0:
                    continue

                power = 10 * np.log10(np.mean(np.abs(iq)**2) + 1e-12)
                detected = power > THRESHOLD_DB
                timestamp = datetime.now(timezone.utc).isoformat()
                predicted_class = -1
                label = ""
                p1 = 0.0
                p2 = 0.0

                if detected and freq != self.last_freq_displayed:
                    with torch.no_grad():
                        # Cap to 1,048,576 samples for consistent model input
                        re = iq.real[:1048576]
                        im = iq.imag[:1048576]
                        if re.size == 0 or im.size == 0:
                            # fall through to logging
                            pass
                        else:
                            tensor = torch.tensor(
                                np.stack((re, im)), dtype=torch.float32
                            ).unsqueeze(0).to(device)

                            logits8 = model(tensor)            # [B, 8] from checkpoint head
                            logits7 = drop_non_drone(logits8)  # [B, 7]; remove 'non-drone'

                            probs = F.softmax(logits7, dim=1)           # [B, 7]
                            topk_prob, topk_idx = torch.topk(
                                probs, k=min(TOPK_TO_LOG, probs.shape[1]), dim=1
                            )

                            p1 = float(topk_prob[0, 0].item())
                            c1 = int(topk_idx[0, 0].item())
                            p2 = float(topk_prob[0, 1].item()) if topk_prob.size(1) > 1 else 0.0

                            # Gate: only "confirm" if confident and well-separated from #2
                            if p1 >= CONFIDENCE_MIN and (p1 - p2) >= MARGIN_MIN:
                                predicted_class = c1
                                label = drone_labels[c1]
                            else:
                                predicted_class = -1
                                label = "Uncertain/Unknown"

                    # UI update
                    self.status_update.emit(
                        f"Signal @ {freq/1e6:.2f} MHz | Power: {power:.2f} dB | "
                        f"Class: {label} | p1={p1:.2f} p2={p2:.2f}"
                    )
                    self.canvas.update_plot(iq, freq)
                    self.last_freq_displayed = freq

                writer.writerow([
                    timestamp, int(freq), round(power, 2), "Yes" if detected else "No",
                    predicted_class, (label if predicted_class != -1 else ""),
                    round(p1, 4), round(p2, 4)
                ])
                log_file.flush()

    def record_iq(self, freq_hz):
        # -n expects the number of complex samples (I+Q as one sample)
        samples = int(SAMPLE_RATE * RECORD_SECS)
        cmd = [
            "hackrf_transfer",
            "-r", TMP_FILE,
            "-f", str(int(freq_hz)),
            "-s", str(int(SAMPLE_RATE)),
            "-g", str(GAIN),
            "-n", str(samples)
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not os.path.exists(TMP_FILE):
            return None
        raw = np.fromfile(TMP_FILE, dtype=np.int8)
        try:
            os.remove(TMP_FILE)
        except Exception:
            pass
        if len(raw) < 2:
            return None
        i = raw[::2].astype(np.float32)
        q = raw[1::2].astype(np.float32)
        return (i + 1j * q) / 128.0

    def stop(self):
        self.running = False

    def toggle_pause(self):
        self.paused = not self.paused

# ===================== MAIN UI ======================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live IQ Signal Scanner (with Classification & Confidence Gate)")
        self.setGeometry(100, 100, 1400, 700)

        self.canvas = PlotCanvas(self)
        self.setCentralWidget(self.canvas)

        self.status_label = QtWidgets.QLabel("Status: Idle")
        self.start_btn = QtWidgets.QPushButton("Start")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.freq_input = QtWidgets.QLineEdit()
        self.freq_input.setPlaceholderText("Enter range (e.g. 100m-400m or 2.4g-2.5g)")

        self.start_btn.clicked.connect(self.start_scan)
        self.pause_btn.clicked.connect(self.toggle_pause)

        toolbar = self.addToolBar("Controls")
        toolbar.addWidget(self.freq_input)
        toolbar.addWidget(self.status_label)
        toolbar.addWidget(self.start_btn)
        toolbar.addWidget(self.pause_btn)

        self.thread = None

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(40, 40, 40))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        self.setPalette(palette)

    def parse_freq_range(self):
        txt = self.freq_input.text().strip().lower()
        if '-' in txt:
            try:
                left, right = txt.split('-')
                left_val  = float(left[:-1])  * (1e9 if 'g' in left  else 1e6)
                right_val = float(right[:-1]) * (1e9 if 'g' in right else 1e6)
                return left_val, right_val
            except Exception:
                pass
        return DEFAULT_START_FREQ, DEFAULT_END_FREQ

    def start_scan(self):
        if self.thread and self.thread.isRunning():
            return
        start, end = self.parse_freq_range()
        self.thread = SignalScanner(self.canvas, start_freq=start, end_freq=end)
        self.thread.status_update.connect(self.status_label.setText)
        self.thread.start()

    def toggle_pause(self):
        if self.thread:
            self.thread.toggle_pause()
            if self.thread.paused:
                self.status_label.setText("Status: Paused")
                self.pause_btn.setText("Resume")
            else:
                self.status_label.setText("Status: Scanning...")
                self.pause_btn.setText("Pause")

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
        event.accept()

# ====================== BOOT =======================
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
