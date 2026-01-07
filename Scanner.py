#!/usr/bin/env python3

""" 
Scanner.py — Spectrum scanner using HackRF or SoapySDR backends with a PyQt5 UI.
Captures IQ data, computes PSD, constellation, and waterfall plots in real-time.
Logs detected signals to a CSV file.

The objective is to provide a GUI tool for scanning RF spectrum ranges,
visualizing signal characteristics, and logging detections for further analysis,
and test hardware.

"""


import os
import csv
import sys
import time
import shutil
import subprocess
from datetime import datetime, timezone
from collections import deque

import numpy as np
from scipy import signal

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QPalette, QColor
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32, SOAPY_SDR_TIMEOUT
    HAVE_SOAPY = True
except Exception:
    SoapySDR = None
    SOAPY_SDR_RX = SOAPY_SDR_CF32 = SOAPY_SDR_TIMEOUT = None
    HAVE_SOAPY = False

# ========================== CONFIG ==========================
DEFAULT_START_FREQ = 10e6        # 10 MHz
DEFAULT_END_FREQ   = 6e9         # 6 GHz
STEP_SIZE          = 1e6         # 1 MHz step
SAMPLE_RATE        = 20e6        # 20 Msps (HackRF max; SDRplay often lower)
RECORD_SECS        = 0.2         # seconds of capture per step
GAIN               = 20          # RF gain (0..40)
THRESHOLD_DB       = -50         # simple power threshold for "detection"
CSV_LOG_FILE       = "spectrum_log.csv"
TMP_FILE           = "temp_iq.raw"
DEFAULT_BACKEND    = "auto"      # auto -> prefer SoapySDR if available, else HackRF
DEFAULT_SOAPY_ARGS = ""          # Examples: driver=rtlsdr,serial=XXXX | driver=sdrplay | driver=uhd

# Waterfall history length (rows)
WATERFALL_HISTORY  = 120

# ========================== UTILS ===========================
def have_hackrf() -> bool:
    return shutil.which("hackrf_transfer") is not None

def have_soapy() -> bool:
    return HAVE_SOAPY

def soapy_enumerate():
    if not HAVE_SOAPY:
        return []
    try:
        return SoapySDR.Device.enumerate()
    except Exception:
        return []

def soapy_kwargs_to_dict(kwargs):
    if kwargs is None:
        return {}
    if isinstance(kwargs, dict):
        return kwargs
    if hasattr(kwargs, "toDict"):
        try:
            return kwargs.toDict()
        except Exception:
            pass
    if hasattr(kwargs, "keys"):
        try:
            return {k: kwargs[k] for k in kwargs.keys()}
        except Exception:
            pass
    try:
        return dict(kwargs)
    except Exception:
        return {}

def soapy_pick_sample_rate(dev, requested: float) -> float:
    try:
        if hasattr(dev, "listSampleRates"):
            rates = dev.listSampleRates(SOAPY_SDR_RX, 0)
            if rates:
                rates = [float(r) for r in rates]
                return min(rates, key=lambda r: abs(r - requested))
    except Exception:
        pass
    try:
        ranges = dev.getSampleRateRange(SOAPY_SDR_RX, 0)
        if ranges:
            best = None
            best_dist = float("inf")
            for r in ranges:
                lo = r.minimum() if callable(getattr(r, "minimum", None)) else getattr(r, "minimum", None)
                hi = r.maximum() if callable(getattr(r, "maximum", None)) else getattr(r, "maximum", None)
                if lo is None or hi is None:
                    continue
                if lo <= requested <= hi:
                    return requested
                cand = lo if requested < lo else hi
                dist = abs(cand - requested)
                if dist < best_dist:
                    best_dist = dist
                    best = cand
            if best is not None:
                return best
    except Exception:
        pass
    try:
        driver = str(dev.getDriverKey()).lower()
        if "rtlsdr" in driver:
            return 2.4e6
    except Exception:
        pass
    return requested

def parse_soapy_args(text: str) -> dict:
    args = {}
    txt = (text or "").strip()
    if not txt:
        return args
    for part in txt.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, val = part.split("=", 1)
            args[key.strip()] = val.strip()
        else:
            if "driver" not in args:
                args["driver"] = part
            else:
                args[part] = ""
    return args

def select_soapy_device(user_args: dict):
    if user_args:
        return soapy_kwargs_to_dict(user_args)
    devices = [soapy_kwargs_to_dict(d) for d in soapy_enumerate()]
    if not devices:
        return None
    preferred = ["sdrplay", "rtlsdr", "hackrf", "airspy", "uhd", "bladerf", "lime"]
    for want in preferred:
        for dev in devices:
            driver = str(dev.get("driver", "")).lower()
            if want in driver:
                return dev
    for dev in devices:
        driver = str(dev.get("driver", "")).lower()
        if driver != "audio":
            return dev
    return devices[0]

def human_freq(f_hz: float) -> str:
    if f_hz >= 1e9:  return f"{f_hz/1e9:.3f} GHz"
    if f_hz >= 1e6:  return f"{f_hz/1e6:.3f} MHz"
    if f_hz >= 1e3:  return f"{f_hz/1e3:.3f} kHz"
    return f"{f_hz:.0f} Hz"

def parse_range(text: str):
    """Parse '100m-400m' or '2.4g-2.5g' (case-insensitive), fallback to defaults."""
    txt = text.strip().lower().replace(" ", "")
    if "-" not in txt:
        return DEFAULT_START_FREQ, DEFAULT_END_FREQ

    def to_hz(part: str) -> float:
        mult = 1.0
        if part.endswith("g"):
            mult = 1e9; part = part[:-1]
        elif part.endswith("m"):
            mult = 1e6; part = part[:-1]
        elif part.endswith("k"):
            mult = 1e3; part = part[:-1]
        return float(part) * mult

    try:
        left, right = txt.split("-")
        return to_hz(left), to_hz(right)
    except Exception:
        return DEFAULT_START_FREQ, DEFAULT_END_FREQ

# ========================== CAPTURE =========================
def record_iq_once(freq_hz: float, sample_rate: float, seconds: float, gain: int):
    """
    Uses hackrf_transfer to capture int8 interleaved IQ into TMP_FILE.
    Returns complex64 numpy array normalized to [-1,1] approx or None on error.
    """
    # HackRF -n expects number of *complex* samples (I+Q as one sample).
    # Each complex sample consists of two int8 values (I and Q).
    samples = int(sample_rate * seconds)
    if samples <= 0:
        return None

    cmd = [
        "hackrf_transfer",
        "-r", TMP_FILE,
        "-f", str(int(freq_hz)),
        "-s", str(int(sample_rate)),
        "-g", str(int(gain)),
        "-n", str(samples)
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if not os.path.exists(TMP_FILE):
            return None
        raw = np.fromfile(TMP_FILE, dtype=np.int8)
        os.remove(TMP_FILE)
        if raw.size < 2:
            return None
        # De-interleave I,Q
        i = raw[0::2].astype(np.float32)
        q = raw[1::2].astype(np.float32)
        iq = (i + 1j * q) / 128.0  # normalize
        return iq.astype(np.complex64, copy=False)
    except Exception:
        # Clean up temp file if anything goes wrong
        try:
            if os.path.exists(TMP_FILE):
                os.remove(TMP_FILE)
        except Exception:
            pass
        return None

class SoapyCapture:
    def __init__(self, device_args: dict):
        if not HAVE_SOAPY:
            raise RuntimeError("SoapySDR not available")
        self.device_args = device_args
        self.dev = SoapySDR.Device(device_args)
        self.stream = None
        self.sample_rate = None
        self.gain = None

    def _ensure_stream(self, sample_rate: float):
        if self.stream is None:
            self.stream = self.dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
            self.dev.activateStream(self.stream)
        if self.sample_rate != sample_rate:
            actual_rate = sample_rate
            try:
                self.dev.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
            except Exception:
                actual_rate = soapy_pick_sample_rate(self.dev, sample_rate)
                try:
                    self.dev.setSampleRate(SOAPY_SDR_RX, 0, actual_rate)
                except Exception:
                    return None
            try:
                self.dev.setBandwidth(SOAPY_SDR_RX, 0, actual_rate)
            except Exception:
                pass
            self.sample_rate = actual_rate
        return self.sample_rate

    def record(self, freq_hz: float, sample_rate: float, seconds: float, gain: int):
        actual_rate = self._ensure_stream(sample_rate)
        if actual_rate is None:
            return None
        sample_rate = actual_rate
        samples = int(sample_rate * seconds)
        if samples <= 0:
            return None
        try:
            if self.gain != gain:
                try:
                    self.dev.setGain(SOAPY_SDR_RX, 0, gain)
                    self.gain = gain
                except Exception:
                    pass
            self.dev.setFrequency(SOAPY_SDR_RX, 0, freq_hz)
        except Exception:
            return None

        buff = np.empty(samples, np.complex64)
        read = 0
        timeout_us = 100000
        deadline = time.time() + max(1.0, seconds * 2.0)
        while read < samples and time.time() < deadline:
            sr = self.dev.readStream(self.stream, [buff[read:]], samples - read, timeout_us)
            if sr.ret > 0:
                read += sr.ret
            elif sr.ret == SOAPY_SDR_TIMEOUT:
                continue
            else:
                return None
        if read < samples:
            return None
        return buff

    def close(self):
        if self.stream is None:
            return
        try:
            self.dev.deactivateStream(self.stream)
        except Exception:
            pass
        try:
            self.dev.closeStream(self.stream)
        except Exception:
            pass
        self.stream = None

# ========================== PLOTTING ========================
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(15, 5), facecolor='black')
        self.ax_psd = self.fig.add_subplot(131, facecolor='black')
        self.ax_const = self.fig.add_subplot(132, facecolor='black')
        self.ax_waterfall = self.fig.add_subplot(133, facecolor='black')
        super().__init__(self.fig)
        self.setParent(parent)
        self.waterfall = deque(maxlen=WATERFALL_HISTORY)

    def update_plots(self, iq: np.ndarray, center_freq_hz: float, fs: float):
        # PSD via Welch
        self.ax_psd.clear()
        nperseg = min(4096, max(256, len(iq)//8))
        f_axis, Pxx = signal.welch(iq, fs=fs, nperseg=nperseg, return_onesided=False, scaling="density")
        # Shift to center zero
        idx = np.argsort(f_axis)
        f_sorted = f_axis[idx]
        p_sorted = 10*np.log10(np.maximum(Pxx[idx], 1e-15))
        self.ax_psd.plot(f_sorted, p_sorted, color='lime', linewidth=0.9)
        self.ax_psd.set_title(f"PSD @ {human_freq(center_freq_hz)}", color='white')
        self.ax_psd.set_xlabel("Frequency (Hz, baseband)", color='white')
        self.ax_psd.set_ylabel("Power Density (dB)", color='white')
        self.ax_psd.tick_params(colors='white')
        self.ax_psd.grid(True, color='grey', alpha=0.3)

        # Constellation (downsample for speed)
        self.ax_const.clear()
        ds = iq[::max(1, len(iq)//5000)]
        self.ax_const.plot(ds.real, ds.imag, '.', alpha=0.3, color='lime', markersize=2)
        self.ax_const.set_title("Constellation", color='white')
        self.ax_const.set_xlabel("I", color='white')
        self.ax_const.set_ylabel("Q", color='white')
        scale = max(0.25, min(np.max(np.abs(ds)) * 1.3, 3.0))
        self.ax_const.set_xlim(-scale, scale)
        self.ax_const.set_ylim(-scale, scale)
        self.ax_const.grid(True, color='grey', alpha=0.3)
        self.ax_const.tick_params(colors='white')

        # Waterfall (FFT magnitude dB)
        self.ax_waterfall.clear()
        # Make a quick spectrum line
        spec = np.abs(np.fft.fftshift(np.fft.fft(iq)))**2
        spec_db = 10*np.log10(spec + 1e-12)
        # Limit width for speed/memory
        max_bins = 2048
        if spec_db.size > max_bins:
            # Centered decimate
            step = spec_db.size // max_bins
            spec_db = spec_db[::step][:max_bins]
        self.waterfall.append(spec_db.astype(np.float32))
        wf = np.vstack(self.waterfall) if len(self.waterfall) > 1 else np.expand_dims(spec_db, 0)
        self.ax_waterfall.imshow(
            wf, aspect='auto', origin='lower', cmap='viridis',
            extent=[-fs/2, fs/2, 0, wf.shape[0]]
        )
        self.ax_waterfall.set_title("Waterfall", color='white')
        self.ax_waterfall.set_xlabel("Freq Offset (Hz)", color='white')
        self.ax_waterfall.set_ylabel("Time (sweeps)", color='white')
        self.ax_waterfall.tick_params(colors='white')

        self.draw()

# ====================== SCAN THREAD ========================
class SweepThread(QtCore.QThread):
    status = QtCore.pyqtSignal(str)

    def __init__(self, canvas: PlotCanvas,
                 start_freq=DEFAULT_START_FREQ, end_freq=DEFAULT_END_FREQ,
                 step=STEP_SIZE, fs=SAMPLE_RATE, seconds=RECORD_SECS, gain=GAIN,
                 threshold_db=THRESHOLD_DB, log_csv=CSV_LOG_FILE,
                 backend=DEFAULT_BACKEND, soapy_args=DEFAULT_SOAPY_ARGS):
        super().__init__()
        self.canvas = canvas
        self.start_f = float(start_freq)
        self.end_f = float(end_freq)
        self.step = float(step)
        self.fs = float(fs)
        self.seconds = float(seconds)
        self.gain = int(gain)
        self.threshold_db = float(threshold_db)
        self.log_csv = log_csv
        self.backend = backend
        self.soapy_args = soapy_args

        self.running = False
        self.paused = False
        self.last_plotted_freq = None

    def run(self):
        self.running = True
        backend_pref = (self.backend or DEFAULT_BACKEND).strip().lower()
        backend = None
        capture = None

        if backend_pref == "hackrf":
            if not have_hackrf():
                self.status.emit("hackrf_transfer not found in PATH.")
                return
            backend = "hackrf"
            self.status.emit("Using HackRF (hackrf_transfer)")
        elif backend_pref in ("auto", "soapy"):
            if not have_soapy():
                if backend_pref == "soapy":
                    self.status.emit("SoapySDR not available.")
                    return
            else:
                user_args = parse_soapy_args(self.soapy_args)
                chosen_args = select_soapy_device(user_args)
                if chosen_args:
                    try:
                        capture = SoapyCapture(chosen_args)
                        backend = "soapy"
                        device_desc = chosen_args.get("label") or chosen_args.get("driver") or "SoapySDR"
                        self.status.emit(f"Using SoapySDR ({device_desc})")
                    except Exception as exc:
                        capture = None
                        if backend_pref == "soapy":
                            self.status.emit(f"SoapySDR init failed: {exc}")
                            return

            if backend is None:
                if backend_pref == "soapy":
                    self.status.emit("No SoapySDR devices found.")
                    return
                if have_hackrf():
                    backend = "hackrf"
                    self.status.emit("Using HackRF (hackrf_transfer)")
                else:
                    self.status.emit("No SDR backend found (HackRF or SoapySDR).")
                    return
        else:
            self.status.emit(f"Unknown backend: {self.backend}")
            return

        # Open CSV
        try:
            with open(self.log_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp_utc", "center_freq_hz", "avg_power_db", "detected"])

                f_cur = self.start_f
                while self.running:
                    if self.paused:
                        self.msleep(100)
                        continue

                    if f_cur > self.end_f:
                        f_cur = self.start_f

                    msg = f"Sweeping {human_freq(f_cur)}  (step {human_freq(self.step)})"
                    self.status.emit(msg)

                    if backend == "hackrf":
                        iq = record_iq_once(f_cur, self.fs, self.seconds, self.gain)
                    else:
                        iq = capture.record(f_cur, self.fs, self.seconds, self.gain)
                        if capture.sample_rate and capture.sample_rate != self.fs:
                            self.fs = capture.sample_rate
                            self.status.emit(f"Using sample rate {human_freq(self.fs)}")
                    if iq is None or iq.size < 2:
                        # advance even if failed capture
                        f_cur += self.step
                        continue

                    # Average power estimate
                    power_db = 10*np.log10(np.mean(np.abs(iq)**2) + 1e-12)
                    detected = power_db > self.threshold_db
                    writer.writerow([datetime.now(timezone.utc).isoformat(), int(f_cur), round(power_db, 2), int(detected)])
                    f.flush()

                    # Only refresh heavy plots when detected or when freq changed to reduce churn
                    if detected or (self.last_plotted_freq != f_cur):
                        self.canvas.update_plots(iq, f_cur, self.fs)
                        self.last_plotted_freq = f_cur

                    f_cur += self.step
        finally:
            if capture:
                capture.close()

    def stop(self):
        self.running = False

    def toggle_pause(self):
        self.paused = not self.paused

# ========================= UI ==============================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrum Scanner")
        self.setGeometry(100, 100, 1400, 720)

        self.canvas = PlotCanvas(self)
        self.setCentralWidget(self.canvas)

        # Toolbar controls
        self.status_label = QtWidgets.QLabel("Status: Idle")
        self.start_btn = QtWidgets.QPushButton("Start")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.setCheckable(True)

        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["Auto", "HackRF (hackrf_transfer)", "SoapySDR"])
        self.device_combo.setCurrentText("Auto")

        self.soapy_args_input = QtWidgets.QLineEdit(DEFAULT_SOAPY_ARGS)
        self.soapy_args_input.setPlaceholderText("Soapy args (e.g. driver=rtlsdr,serial=XXXX)")
        self.soapy_args_input.setFixedWidth(180)

        self.range_input = QtWidgets.QLineEdit()
        self.range_input.setPlaceholderText("Range (e.g. 100m-200m or 2.4g-2.5g)")
        self.range_input.setText("2.4g-2.5g")

        self.step_input = QtWidgets.QLineEdit(str(int(STEP_SIZE/1e6)))
        self.step_input.setFixedWidth(60)
        self.step_unit = QtWidgets.QComboBox()
        self.step_unit.addItems(["kHz","MHz"])
        self.step_unit.setCurrentText("MHz")

        self.rate_input = QtWidgets.QLineEdit(str(int(SAMPLE_RATE/1e6)))
        self.rate_input.setFixedWidth(60)
        self.rate_label = QtWidgets.QLabel("Msps")

        self.secs_input = QtWidgets.QLineEdit(str(RECORD_SECS))
        self.secs_input.setFixedWidth(60)
        self.secs_label = QtWidgets.QLabel("sec/step")

        self.gain_input = QtWidgets.QLineEdit(str(GAIN))
        self.gain_input.setFixedWidth(60)
        self.gain_label = QtWidgets.QLabel("gain")

        self.thresh_input = QtWidgets.QLineEdit(str(THRESHOLD_DB))
        self.thresh_input.setFixedWidth(70)
        self.thresh_label = QtWidgets.QLabel("dB thresh")

        self.toolbar = self.addToolBar("Controls")
        self.toolbar.addWidget(QtWidgets.QLabel(" Device: "))
        self.toolbar.addWidget(self.device_combo)
        self.toolbar.addWidget(QtWidgets.QLabel(" Soapy args: "))
        self.toolbar.addWidget(self.soapy_args_input)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(QtWidgets.QLabel(" Range: "))
        self.toolbar.addWidget(self.range_input)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(QtWidgets.QLabel(" Step: "))
        self.toolbar.addWidget(self.step_input)
        self.toolbar.addWidget(self.step_unit)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(QtWidgets.QLabel(" Rate: "))
        self.toolbar.addWidget(self.rate_input)
        self.toolbar.addWidget(self.rate_label)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(QtWidgets.QLabel(" Capture: "))
        self.toolbar.addWidget(self.secs_input)
        self.toolbar.addWidget(self.secs_label)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(QtWidgets.QLabel(" Gain: "))
        self.toolbar.addWidget(self.gain_input)
        self.toolbar.addWidget(self.gain_label)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(QtWidgets.QLabel(" Threshold: "))
        self.toolbar.addWidget(self.thresh_input)
        self.toolbar.addWidget(self.thresh_label)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.status_label)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.start_btn)
        self.toolbar.addWidget(self.pause_btn)

        # Signals
        self.start_btn.clicked.connect(self.start_scan)
        self.pause_btn.clicked.connect(self.toggle_pause)

        # Theme
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(40, 40, 40))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        self.setPalette(palette)

        self.thread = None

    def start_scan(self):
        if self.thread and self.thread.isRunning():
            return

        start_hz, end_hz = parse_range(self.range_input.text())
        backend_text = self.device_combo.currentText().lower()
        if backend_text.startswith("hackrf"):
            backend = "hackrf"
        elif backend_text.startswith("soapy"):
            backend = "soapy"
        else:
            backend = "auto"
        soapy_args = self.soapy_args_input.text()

        # Step (kHz/MHz)
        try:
            step_val = float(self.step_input.text())
        except Exception:
            step_val = STEP_SIZE / 1e6
        unit = self.step_unit.currentText().lower()
        step_hz = step_val * (1e6 if unit == "mhz" else 1e3)

        # Sample rate, seconds, gain, threshold
        try: fs = float(self.rate_input.text()) * 1e6
        except Exception: fs = SAMPLE_RATE
        try: secs = float(self.secs_input.text())
        except Exception: secs = RECORD_SECS
        try: gain = int(self.gain_input.text())
        except Exception: gain = GAIN
        try: thresh = float(self.thresh_input.text())
        except Exception: thresh = THRESHOLD_DB

        self.thread = SweepThread(self.canvas, start_freq=start_hz, end_freq=end_hz,
                                  step=step_hz, fs=fs, seconds=secs, gain=gain,
                                  threshold_db=thresh, log_csv=CSV_LOG_FILE,
                                  backend=backend, soapy_args=soapy_args)
        self.thread.status.connect(self.status_label.setText)
        self.thread.start()

    def toggle_pause(self):
        if not self.thread:
            return
        self.thread.toggle_pause()
        if self.thread.paused:
            self.status_label.setText("Status: Paused")
            self.pause_btn.setText("Resume")
        else:
            self.status_label.setText("Status: Scanning…")
            self.pause_btn.setText("Pause")

    def closeEvent(self, event):
        try:
            if self.thread:
                self.thread.stop()
                self.thread.wait(1000)
        finally:
            event.accept()

# ========================= MAIN ============================
if __name__ == "__main__":
    if not have_hackrf() and not have_soapy():
        print("[!] No SDR backend found. Install HackRF tools or SoapySDR.")
        sys.exit(1)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
