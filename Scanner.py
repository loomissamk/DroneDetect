#!/usr/bin/env python3

"""
DroneDetect spectrum inspector.

This scanner keeps the original HackRF/SoapySDR backends, but upgrades the UI
into a sweep-and-inspect workflow:
  - latest sweep overview across the configured band
  - sweep-history waterfall across the full scan range
  - detail PSD and in-capture waterfall for the current hit
  - IQ waveform and constellation views
  - automatic IQ clip recording when detections trigger
"""

import csv
import os
import shutil
import subprocess
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import signal

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor, QFont, QPalette
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

warnings.filterwarnings("ignore", message=r"sipPyTypeDict\(\) is deprecated", category=DeprecationWarning)

try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX, SOAPY_SDR_TIMEOUT

    HAVE_SOAPY = True
except Exception:
    SoapySDR = None
    SOAPY_SDR_CF32 = SOAPY_SDR_RX = SOAPY_SDR_TIMEOUT = None
    HAVE_SOAPY = False


# ========================== CONFIG ==========================
DEFAULT_START_FREQ = 10e6
DEFAULT_END_FREQ = 6e9
DEFAULT_RANGE_TEXT = "433.800m-434.200m"
STEP_SIZE = 250e3
SAMPLE_RATE = 2.4e6
RECORD_SECS = 0.2
GAIN = 20
DEFAULT_SNR_GATE_DB = 14.0
CSV_LOG_FILE = "spectrum_log.csv"
TMP_FILE = "temp_iq.raw"
DEFAULT_BACKEND = "auto"
DEFAULT_SOAPY_ARGS = ""
DEFAULT_RECORDINGS_DIR = "recordings"
DEFAULT_RECORD_COOLDOWN_SECS = 2.0
DEFAULT_DECODER_MODE = "auto_basic"
FIXED_SIDEBAR_WIDTH = 540
MIN_SIDEBAR_WIDTH = 420

MAX_SWEEP_STEPS = 100000
SWEEP_RENDER_BINS = 768
SWEEP_HISTORY = 80
MAX_ANALYSIS_SAMPLES = 131072
MAX_DETAIL_WATERFALL_SAMPLES = 65536
MAX_WAVEFORM_POINTS = 4096
MAX_CONSTELLATION_POINTS = 3000
MAX_DECODE_SAMPLES = 131072
MAX_DECODE_BITS = 512
DETAIL_FFT_BINS = 4096

DECODER_MODE_OPTIONS = [
    ("auto_basic", "Auto Basic"),
    ("auto_advanced", "Auto Advanced"),
    ("ask", "ASK / OOK"),
    ("fsk", "2-FSK"),
    ("manchester", "Manchester / OOK"),
    ("uart", "UART-like"),
    ("off", "Off"),
]

DEVICE_PROFILES = {
    "hackrf": {
        "display": "HackRF",
        "default_range": "2.400g-2.500g",
        "sample_rate": 20e6,
        "step_hz": 1e6,
        "min_freq_hz": 1e6,
        "max_freq_hz": 6e9,
        "hint": "Wideband sweeps and 2.4/5.8 GHz work well here.",
    },
    "rtlsdr": {
        "display": "RTL-SDR",
        "default_range": "433.800m-434.200m",
        "sample_rate": 2.4e6,
        "step_hz": 100e3,
        "min_freq_hz": 24e6,
        "max_freq_hz": 1766e6,
        "hint": "Use sub-GHz, VHF, or lower UHF bands. 2.4 GHz is out of range.",
    },
    "sdrplay": {
        "display": "SDRplay",
        "default_range": "433.800m-434.200m",
        "sample_rate": 8e6,
        "step_hz": 250e3,
        "min_freq_hz": 1e3,
        "max_freq_hz": 2e9,
        "hint": "Good middle ground for sub-GHz and lower microwave work.",
    },
    "generic": {
        "display": "Generic SDR",
        "default_range": DEFAULT_RANGE_TEXT,
        "sample_rate": SAMPLE_RATE,
        "step_hz": STEP_SIZE,
        "min_freq_hz": DEFAULT_START_FREQ,
        "max_freq_hz": DEFAULT_END_FREQ,
        "hint": "Profile is unknown. Start with a narrow band and moderate sample rate.",
    },
}


# ========================== DATA ============================
@dataclass
class DecodePreview:
    method: str
    symbol_rate_hz: float
    confidence: float
    bit_preview: str
    hex_preview: str
    ascii_preview: str
    notes: str


@dataclass
class BitstreamCandidate:
    method: str
    bits: np.ndarray
    byte_values: np.ndarray
    symbol_rate_hz: float
    symbol_samples: int
    byte_offset: int
    confidence: float
    printable_ratio: float
    unique_ratio: float
    idle_ratio: float
    notes: str


@dataclass
class SweepSnapshot:
    timestamp_utc: str
    center_freq_hz: float
    sample_rate: float
    decoder_mode: str
    step_index: int
    total_steps: int
    sweep_number: int
    avg_power_db: float
    peak_db: float
    noise_floor_db: float
    snr_db: float
    occupied_bw_hz: float
    dominant_offset_hz: float
    detected: bool
    recording_path: str
    freq_axis_hz: np.ndarray
    spectrum_db: np.ndarray
    detail_waterfall_freq_hz: np.ndarray
    detail_waterfall_time_s: np.ndarray
    detail_waterfall_db: np.ndarray
    waveform_time_ms: np.ndarray
    waveform_i: np.ndarray
    waveform_q: np.ndarray
    constellation: np.ndarray
    decode_preview: DecodePreview


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
                rates = [float(rate) for rate in rates]
                return min(rates, key=lambda rate: abs(rate - requested))
    except Exception:
        pass

    try:
        ranges = dev.getSampleRateRange(SOAPY_SDR_RX, 0)
        if ranges:
            best = None
            best_dist = float("inf")
            for rate_range in ranges:
                lo = rate_range.minimum() if callable(getattr(rate_range, "minimum", None)) else getattr(rate_range, "minimum", None)
                hi = rate_range.maximum() if callable(getattr(rate_range, "maximum", None)) else getattr(rate_range, "maximum", None)
                if lo is None or hi is None:
                    continue
                if lo <= requested <= hi:
                    return requested
                candidate = lo if requested < lo else hi
                dist = abs(candidate - requested)
                if dist < best_dist:
                    best = candidate
                    best_dist = dist
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
    raw = (text or "").strip()
    if not raw:
        return args
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            args[key.strip()] = value.strip()
        elif "driver" not in args:
            args["driver"] = part
        else:
            args[part] = ""
    return args


def select_soapy_device(user_args: dict):
    if user_args:
        return soapy_kwargs_to_dict(user_args)

    devices = [soapy_kwargs_to_dict(device) for device in soapy_enumerate()]
    if not devices:
        return None

    preferred = ["sdrplay", "rtlsdr", "hackrf", "airspy", "uhd", "bladerf", "lime"]
    for wanted in preferred:
        for device in devices:
            driver = str(device.get("driver", "")).lower()
            if wanted in driver:
                return device

    for device in devices:
        if str(device.get("driver", "")).lower() != "audio":
            return device

    return devices[0]


def normalize_driver_key(driver: str) -> str:
    raw = str(driver or "").strip().lower()
    for known in ("hackrf", "rtlsdr", "sdrplay"):
        if known in raw:
            return known
    return raw or "generic"


def device_profile_for_driver(driver: str) -> dict:
    return DEVICE_PROFILES.get(normalize_driver_key(driver), DEVICE_PROFILES["generic"])


def soapy_range_bounds(ranges) -> Optional[tuple]:
    lows = []
    highs = []
    if not ranges:
        return None
    for item in ranges:
        lo = item.minimum() if callable(getattr(item, "minimum", None)) else getattr(item, "minimum", None)
        hi = item.maximum() if callable(getattr(item, "maximum", None)) else getattr(item, "maximum", None)
        if lo is None or hi is None:
            continue
        lows.append(float(lo))
        highs.append(float(hi))
    if not lows or not highs:
        return None
    return min(lows), max(highs)


def detect_preferred_profile(backend_choice: str, soapy_args_text: str) -> tuple:
    backend = (backend_choice or DEFAULT_BACKEND).strip().lower()
    if backend == "hackrf":
        return "hackrf", DEVICE_PROFILES["hackrf"]

    if backend in ("auto", "soapy"):
        args = parse_soapy_args(soapy_args_text)
        driver = normalize_driver_key(args.get("driver", ""))
        if driver != "generic":
            return driver, device_profile_for_driver(driver)
        chosen = select_soapy_device(args if args else {})
        if chosen:
            driver = normalize_driver_key(chosen.get("driver", ""))
            return driver, device_profile_for_driver(driver)
        if backend == "auto" and have_hackrf():
            return "hackrf", DEVICE_PROFILES["hackrf"]

    if have_hackrf() and not have_soapy():
        return "hackrf", DEVICE_PROFILES["hackrf"]
    return "generic", DEVICE_PROFILES["generic"]


def recommended_step_for_profile(profile: dict) -> tuple:
    step_hz = float(profile.get("step_hz", STEP_SIZE))
    if step_hz >= 1e6:
        return step_hz / 1e6, "MHz"
    return step_hz / 1e3, "kHz"


def validate_frequency_plan(profile: dict, frequency_plan_hz: np.ndarray):
    min_freq_hz = float(profile.get("min_freq_hz", DEFAULT_START_FREQ))
    max_freq_hz = float(profile.get("max_freq_hz", DEFAULT_END_FREQ))
    start = float(frequency_plan_hz[0])
    end = float(frequency_plan_hz[-1])
    if start < min_freq_hz or end > max_freq_hz:
        raise ValueError(
            f"{profile['display']} supports {human_freq(min_freq_hz)} to {human_freq(max_freq_hz)}. "
            f"Requested sweep {human_freq(start)} to {human_freq(end)} is outside that range."
        )


def have_external_tool(tool_name: str) -> bool:
    return shutil.which(tool_name) is not None


def human_freq(freq_hz: float) -> str:
    if freq_hz >= 1e9:
        return f"{freq_hz / 1e9:.3f} GHz"
    if freq_hz >= 1e6:
        return f"{freq_hz / 1e6:.3f} MHz"
    if freq_hz >= 1e3:
        return f"{freq_hz / 1e3:.3f} kHz"
    return f"{freq_hz:.0f} Hz"


def human_rate(rate_hz: float) -> str:
    if rate_hz >= 1e6:
        return f"{rate_hz / 1e6:.2f} Msps"
    if rate_hz >= 1e3:
        return f"{rate_hz / 1e3:.2f} ksps"
    return f"{rate_hz:.0f} sps"


def human_bandwidth(bandwidth_hz: float) -> str:
    if bandwidth_hz >= 1e6:
        return f"{bandwidth_hz / 1e6:.2f} MHz"
    if bandwidth_hz >= 1e3:
        return f"{bandwidth_hz / 1e3:.1f} kHz"
    return f"{bandwidth_hz:.0f} Hz"


def parse_range(text: str):
    raw = (text or "").strip().lower().replace(" ", "")
    if "-" not in raw:
        return DEFAULT_START_FREQ, DEFAULT_END_FREQ

    def to_hz(part: str) -> float:
        mult = 1.0
        if part.endswith("g"):
            mult = 1e9
            part = part[:-1]
        elif part.endswith("m"):
            mult = 1e6
            part = part[:-1]
        elif part.endswith("k"):
            mult = 1e3
            part = part[:-1]
        return float(part) * mult

    try:
        left, right = raw.split("-", 1)
        return to_hz(left), to_hz(right)
    except Exception:
        return DEFAULT_START_FREQ, DEFAULT_END_FREQ


def build_frequency_plan(start_freq: float, end_freq: float, step_hz: float) -> np.ndarray:
    start = float(min(start_freq, end_freq))
    end = float(max(start_freq, end_freq))
    step = max(float(step_hz), 1.0)
    steps = int(np.floor((end - start) / step)) + 1
    if steps <= 0:
        raise ValueError("Frequency range produced no sweep steps.")
    if steps > MAX_SWEEP_STEPS:
        raise ValueError(f"Range is too dense ({steps} steps). Increase step size.")
    return start + np.arange(steps, dtype=np.float64) * step


def sanitize_filename(text: str) -> str:
    keep = []
    for char in text:
        keep.append(char if char.isalnum() or char in ("-", "_", ".") else "_")
    return "".join(keep)


def contiguous_view(iq: np.ndarray, max_samples: int) -> np.ndarray:
    if iq.size <= max_samples:
        return iq
    return iq[:max_samples]


def stride_view(iq: np.ndarray, max_samples: int, sample_rate: float):
    if iq.size <= max_samples:
        return iq, sample_rate
    stride = int(np.ceil(iq.size / max_samples))
    return iq[::stride][:max_samples], sample_rate / stride


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1 or values.size < window:
        return values.astype(np.float32, copy=False)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same").astype(np.float32, copy=False)


def estimate_symbol_samples(binary: np.ndarray) -> Optional[int]:
    if binary.size < 64:
        return None
    transitions = np.flatnonzero(binary[1:] != binary[:-1]) + 1
    if transitions.size < 8:
        return None
    intervals = np.diff(transitions)
    intervals = intervals[(intervals > 0) & (intervals < binary.size // 2)]
    if intervals.size < 4:
        return None
    return max(1, int(round(np.percentile(intervals, 25))))


def estimate_symbol_sample_candidates(binary: np.ndarray):
    base = estimate_symbol_samples(binary)
    if base is None:
        return []
    candidates = {int(base)}
    for factor in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        sample_count = int(round(base * factor))
        if 1 <= sample_count <= max(1, binary.size // 8):
            candidates.add(sample_count)
    return sorted(candidates)


def trim_leading_idle(bits: np.ndarray) -> np.ndarray:
    if bits.size > 8 and np.any(bits != bits[0]):
        first_change = int(np.flatnonzero(bits != bits[0])[0])
        return bits[max(0, first_change - 1) :]
    return bits


def derive_binary_series(series: np.ndarray, smooth_window: Optional[int] = None):
    if series.size < 512:
        return None, None
    if smooth_window is None:
        smooth_window = max(5, min(256, series.size // 4096))
    smoothed = moving_average(series.astype(np.float32, copy=False), smooth_window)
    low = float(np.percentile(smoothed, 20))
    high = float(np.percentile(smoothed, 80))
    if (high - low) <= 1e-5:
        return None, None
    threshold = float((low + high) * 0.5)
    binary = (smoothed > threshold).astype(np.uint8)
    return binary, smooth_window


def extract_symbol_bits(binary: np.ndarray, symbol_samples: int, start_index: Optional[int] = None) -> Optional[np.ndarray]:
    if symbol_samples <= 0 or binary.size < symbol_samples * 16:
        return None
    transitions = np.flatnonzero(binary[1:] != binary[:-1]) + 1
    if start_index is None:
        start_index = max(0, int(transitions[0] - symbol_samples)) if transitions.size else 0
    usable = binary[start_index:]
    bit_count = usable.size // symbol_samples
    if bit_count < 16:
        return None
    bit_matrix = usable[: bit_count * symbol_samples].reshape(bit_count, symbol_samples)
    bits = (np.mean(bit_matrix, axis=1) >= 0.5).astype(np.uint8)
    return trim_leading_idle(bits)


def choose_byte_alignment(bits: np.ndarray):
    best_offset = 0
    best_bytes = np.array([], dtype=np.uint8)
    best_score = float("-inf")
    max_offset = min(8, bits.size)
    for offset in range(max_offset):
        usable_len = ((bits.size - offset) // 8) * 8
        if usable_len < 16:
            continue
        raw = bits[offset : offset + usable_len].astype(np.uint8, copy=False)
        byte_values = np.packbits(raw)
        printable_ratio = float(np.mean((byte_values >= 32) & (byte_values < 127)))
        unique_ratio = float(len(np.unique(byte_values[: min(len(byte_values), 24)]))) / float(max(min(len(byte_values), 24), 1))
        idle_ratio = float(np.mean((byte_values == 0x00) | (byte_values == 0xFF)))
        score = (printable_ratio * 0.6) + (unique_ratio * 0.5) - (idle_ratio * 0.5)
        if score > best_score:
            best_score = score
            best_offset = offset
            best_bytes = byte_values
    return best_offset, best_bytes


def render_bit_preview(bits: np.ndarray) -> str:
    if bits.size == 0:
        return "—"
    preview = bits[:MAX_DECODE_BITS]
    groups = ["".join(str(int(bit)) for bit in preview[idx : idx + 8]) for idx in range(0, preview.size, 8)]
    return " ".join(groups)


def render_hex_preview(byte_values: np.ndarray) -> str:
    if byte_values.size == 0:
        return "—"
    return " ".join(f"{int(byte_val):02X}" for byte_val in byte_values[:24])


def render_ascii_preview(byte_values: np.ndarray) -> str:
    if byte_values.size == 0:
        return "—"
    chars = [chr(int(byte_val)) if 32 <= int(byte_val) < 127 else "." for byte_val in byte_values[:48]]
    return "".join(chars)


def decoder_mode_label(mode: str) -> str:
    for key, label in DECODER_MODE_OPTIONS:
        if key == mode:
            return label
    return mode.replace("_", " ").title()


def empty_decode_preview(notes: str, method: str = "No clean decode") -> DecodePreview:
    return DecodePreview(
        method=method,
        symbol_rate_hz=0.0,
        confidence=0.0,
        bit_preview="—",
        hex_preview="—",
        ascii_preview="—",
        notes=notes,
    )


def build_bitstream_candidate(
    bits: np.ndarray,
    sample_rate: float,
    symbol_samples: int,
    method: str,
    notes: str,
    confidence_bias: float = 0.0,
    min_confidence: float = 0.14,
) -> Optional[BitstreamCandidate]:
    bits = trim_leading_idle(bits.astype(np.uint8, copy=False))
    if bits.size < 16:
        return None

    ones_ratio = float(np.mean(bits))
    transition_ratio = float(np.mean(bits[1:] != bits[:-1])) if bits.size > 1 else 0.0
    if not (0.03 <= ones_ratio <= 0.97):
        return None

    byte_offset, byte_values = choose_byte_alignment(bits)
    if byte_values.size < 2:
        return None

    aligned_len = int(byte_values.size * 8)
    aligned_bits = bits[byte_offset : byte_offset + aligned_len]
    if aligned_bits.size < 16:
        return None

    printable_ratio = float(np.mean((byte_values >= 32) & (byte_values < 127)))
    unique_ratio = float(len(np.unique(byte_values[: min(len(byte_values), 24)]))) / float(max(min(len(byte_values), 24), 1))
    idle_ratio = float(np.mean((byte_values == 0x00) | (byte_values == 0xFF)))
    if idle_ratio > 0.94 and unique_ratio < 0.12:
        return None

    length_factor = min(aligned_bits.size / 160.0, 1.0)
    balance_factor = max(0.0, 1.0 - abs(ones_ratio - 0.5) * 2.0)
    transition_factor = min(transition_ratio * 1.8, 1.0)
    structure_factor = max(0.0, min((1.0 - idle_ratio) * 0.55 + unique_ratio * 0.30 + printable_ratio * 0.15, 1.0))
    confidence = max(
        0.05,
        min(length_factor * 0.28 + balance_factor * 0.24 + transition_factor * 0.18 + structure_factor * 0.30 + confidence_bias, 0.99),
    )
    if confidence < min_confidence:
        return None

    return BitstreamCandidate(
        method=method,
        bits=aligned_bits,
        byte_values=byte_values,
        symbol_rate_hz=float(sample_rate / symbol_samples) if symbol_samples else 0.0,
        symbol_samples=int(symbol_samples),
        byte_offset=int(byte_offset),
        confidence=confidence,
        printable_ratio=printable_ratio,
        unique_ratio=unique_ratio,
        idle_ratio=idle_ratio,
        notes=f"{notes} Byte alignment guessed with bit offset {byte_offset}.",
    )


def build_byte_candidate(
    byte_values: np.ndarray,
    bits: np.ndarray,
    sample_rate: float,
    symbol_samples: int,
    method: str,
    notes: str,
    confidence_base: float,
    min_confidence: float = 0.14,
) -> Optional[BitstreamCandidate]:
    if byte_values.size < 2 or bits.size < 16:
        return None

    printable_ratio = float(np.mean((byte_values >= 32) & (byte_values < 127)))
    unique_ratio = float(len(np.unique(byte_values[: min(len(byte_values), 24)]))) / float(max(min(len(byte_values), 24), 1))
    idle_ratio = float(np.mean((byte_values == 0x00) | (byte_values == 0xFF)))
    transition_ratio = float(np.mean(bits[1:] != bits[:-1])) if bits.size > 1 else 0.0
    balance_factor = max(0.0, 1.0 - abs(float(np.mean(bits)) - 0.5) * 2.0)
    structure_factor = max(0.0, min((1.0 - idle_ratio) * 0.50 + unique_ratio * 0.30 + printable_ratio * 0.20, 1.0))
    confidence = max(
        0.05,
        min(confidence_base * 0.60 + structure_factor * 0.25 + balance_factor * 0.10 + min(transition_ratio * 0.2, 0.05), 0.99),
    )
    if confidence < min_confidence:
        return None

    return BitstreamCandidate(
        method=method,
        bits=bits,
        byte_values=byte_values,
        symbol_rate_hz=float(sample_rate / symbol_samples) if symbol_samples else 0.0,
        symbol_samples=int(symbol_samples),
        byte_offset=0,
        confidence=confidence,
        printable_ratio=printable_ratio,
        unique_ratio=unique_ratio,
        idle_ratio=idle_ratio,
        notes=notes,
    )


def candidate_to_preview(candidate: BitstreamCandidate) -> DecodePreview:
    method = candidate.method
    if candidate.confidence < 0.35:
        method = f"Speculative {method}"
    elif candidate.confidence < 0.55:
        method = f"Weak {method}"

    return DecodePreview(
        method=method,
        symbol_rate_hz=candidate.symbol_rate_hz,
        confidence=candidate.confidence,
        bit_preview=render_bit_preview(candidate.bits),
        hex_preview=render_hex_preview(candidate.byte_values),
        ascii_preview=render_ascii_preview(candidate.byte_values),
        notes=(
            f"{candidate.notes} Symbol≈{candidate.symbol_samples} samples. "
            f"Printable={candidate.printable_ratio:.2f}, unique={candidate.unique_ratio:.2f}, idle={candidate.idle_ratio:.2f}."
        ),
    )


def isolate_signal_region(iq: np.ndarray, sample_rate: float, dominant_offset_hz: float, occupied_bw_hz: float):
    if iq.size < 1024:
        return iq, sample_rate

    sample_index = np.arange(iq.size, dtype=np.float32)
    rotator = np.exp(-1j * 2.0 * np.pi * float(dominant_offset_hz) * sample_index / float(sample_rate)).astype(np.complex64)
    shifted = iq.astype(np.complex64, copy=False) * rotator

    target_bw_hz = max(occupied_bw_hz * 4.0, 12e3)
    cutoff_hz = min(target_bw_hz, sample_rate * 0.45)
    normalized_cutoff = cutoff_hz / sample_rate
    taps = signal.firwin(129, normalized_cutoff)
    filtered = signal.lfilter(taps, [1.0], shifted).astype(np.complex64, copy=False)

    target_rate_hz = max(96e3, cutoff_hz * 10.0)
    decimation = max(1, int(sample_rate // target_rate_hz))
    if decimation > 1:
        filtered = filtered[::decimation]
        sample_rate = sample_rate / decimation

    return filtered, float(sample_rate)


def decode_binary_series(
    series: np.ndarray,
    sample_rate: float,
    method: str,
    smooth_window: Optional[int] = None,
    min_confidence: float = 0.16,
) -> Optional[BitstreamCandidate]:
    binary, smooth_window = derive_binary_series(series, smooth_window)
    if binary is None:
        return None

    candidates = []
    for symbol_samples in estimate_symbol_sample_candidates(binary):
        if symbol_samples < 2:
            continue
        bits = extract_symbol_bits(binary, symbol_samples)
        if bits is None:
            continue
        for inverted in (False, True):
            view_bits = (1 - bits) if inverted else bits
            candidate = build_bitstream_candidate(
                view_bits,
                sample_rate,
                symbol_samples,
                method,
                notes=(
                    f"NRZ slicer with smooth={smooth_window}."
                    + (" Inverted polarity applied." if inverted else "")
                ),
                min_confidence=min_confidence,
            )
            if candidate is not None:
                candidates.append(candidate)
    return max(candidates, key=lambda item: item.confidence) if candidates else None


def decode_manchester_series(
    series: np.ndarray,
    sample_rate: float,
    method: str,
    smooth_window: Optional[int] = None,
    min_confidence: float = 0.16,
) -> Optional[BitstreamCandidate]:
    binary, smooth_window = derive_binary_series(series, smooth_window)
    if binary is None:
        return None

    transitions = np.flatnonzero(binary[1:] != binary[:-1]) + 1
    start_index = max(0, int(transitions[0] - 2)) if transitions.size else 0
    candidates = []
    for half_symbol_samples in estimate_symbol_sample_candidates(binary):
        if half_symbol_samples < 2:
            continue
        usable = binary[start_index:]
        pair_samples = int(half_symbol_samples * 2)
        pair_count = usable.size // pair_samples
        if pair_count < 10:
            continue
        pair_view = usable[: pair_count * pair_samples].reshape(pair_count, 2, half_symbol_samples)
        first_half = np.mean(pair_view[:, 0, :], axis=1)
        second_half = np.mean(pair_view[:, 1, :], axis=1)
        zero_to_one = (first_half < 0.45) & (second_half > 0.55)
        one_to_zero = (first_half > 0.55) & (second_half < 0.45)
        valid_mask = zero_to_one | one_to_zero
        valid_ratio = float(np.mean(valid_mask))
        if valid_ratio < 0.42:
            continue
        bits = np.where(zero_to_one, 1, 0)[valid_mask].astype(np.uint8)
        if bits.size < 16:
            continue
        for inverted in (False, True):
            candidate = build_bitstream_candidate(
                (1 - bits) if inverted else bits,
                sample_rate,
                pair_samples,
                method,
                notes=(
                    f"Manchester pair slicer with smooth={smooth_window}, valid={valid_ratio:.2f}."
                    + (" Inverted polarity applied." if inverted else "")
                ),
                confidence_bias=0.08 + max(0.0, valid_ratio - 0.50) * 0.30,
                min_confidence=min_confidence,
            )
            if candidate is not None:
                candidates.append(candidate)
    return max(candidates, key=lambda item: item.confidence) if candidates else None


def decode_uart_like_series(
    series: np.ndarray,
    sample_rate: float,
    method: str,
    smooth_window: Optional[int] = None,
    min_confidence: float = 0.16,
) -> Optional[BitstreamCandidate]:
    binary, smooth_window = derive_binary_series(series, smooth_window)
    if binary is None:
        return None

    candidates = []
    for symbol_samples in estimate_symbol_sample_candidates(binary):
        if symbol_samples < 2:
            continue
        max_offsets = min(symbol_samples, 10)
        for offset in range(max_offsets):
            bits = extract_symbol_bits(binary[offset:], symbol_samples, start_index=0)
            if bits is None or bits.size < 30:
                continue
            for inverted in (False, True):
                stream = (1 - bits) if inverted else bits
                frames = []
                cursor = 1
                while cursor + 9 < stream.size:
                    if stream[cursor - 1] == 1 and stream[cursor] == 0 and stream[cursor + 9] == 1:
                        byte_val = 0
                        for bit_index, bit in enumerate(stream[cursor + 1 : cursor + 9]):
                            byte_val |= int(bit) << bit_index
                        frames.append(byte_val)
                        cursor += 10
                    else:
                        cursor += 1
                if len(frames) < 2:
                    continue
                byte_values = np.asarray(frames, dtype=np.uint8)
                display_bits = np.unpackbits(byte_values)
                frame_density = min(len(frames) / max(stream.size / 10.0, 1.0), 1.0)
                candidate = build_byte_candidate(
                    byte_values=byte_values,
                    bits=display_bits,
                    sample_rate=sample_rate,
                    symbol_samples=symbol_samples,
                    method=method,
                    notes=(
                        f"UART-like frame finder with smooth={smooth_window}, offset={offset}, frames={len(frames)}."
                        + (" Inverted polarity applied." if inverted else "")
                    ),
                    confidence_base=0.28 + min(len(frames) / 8.0, 1.0) * 0.30 + frame_density * 0.22,
                    min_confidence=min_confidence,
                )
                if candidate is not None:
                    candidates.append(candidate)
    return max(candidates, key=lambda item: item.confidence) if candidates else None


def decode_signal_preview(
    iq: np.ndarray,
    sample_rate: float,
    dominant_offset_hz: float,
    occupied_bw_hz: float,
    decoder_mode: str,
) -> DecodePreview:
    mode = (decoder_mode or DEFAULT_DECODER_MODE).strip().lower()
    if mode == "off":
        return empty_decode_preview("Decoder disabled by selected mode.", method="Decoder Off")

    decode_iq, decode_rate = isolate_signal_region(
        contiguous_view(iq, MAX_DECODE_SAMPLES),
        sample_rate,
        dominant_offset_hz,
        occupied_bw_hz,
    )
    envelope = np.abs(decode_iq)
    discriminator = np.angle(decode_iq[1:] * np.conj(decode_iq[:-1])) if decode_iq.size > 1 else np.array([], dtype=np.float32)

    candidates = []
    search_catalog = {
        "ask": [
            ("ASK / OOK envelope", envelope, decode_binary_series, (1, 2, 4, 8, 16), (3, 7, 15, 31, 63), 0.16),
        ],
        "fsk": [
            ("2-FSK discriminator", discriminator.astype(np.float32, copy=False), decode_binary_series, (1, 2, 4, 8, 16), (3, 7, 15, 31, 63), 0.16),
        ],
        "manchester": [
            ("Manchester / OOK", envelope, decode_manchester_series, (1, 2, 4, 8, 16), (3, 7, 15, 31), 0.15),
        ],
        "uart": [
            ("UART-like OOK", envelope, decode_uart_like_series, (1, 2, 4, 8, 16), (3, 7, 15, 31), 0.15),
            ("UART-like FSK", discriminator.astype(np.float32, copy=False), decode_uart_like_series, (1, 2, 4, 8, 16), (3, 7, 15, 31), 0.15),
        ],
    }
    if mode == "auto_basic":
        search_plan = search_catalog["ask"] + search_catalog["fsk"]
        presentation_gate = 0.18
    elif mode == "auto_advanced":
        search_plan = search_catalog["ask"] + search_catalog["fsk"] + search_catalog["manchester"] + search_catalog["uart"]
        presentation_gate = 0.15
    elif mode == "ask":
        search_plan = search_catalog["ask"]
        presentation_gate = 0.15
    elif mode == "fsk":
        search_plan = search_catalog["fsk"]
        presentation_gate = 0.15
    elif mode == "manchester":
        search_plan = search_catalog["manchester"]
        presentation_gate = 0.14
    elif mode == "uart":
        search_plan = search_catalog["uart"]
        presentation_gate = 0.14
    else:
        search_plan = search_catalog["ask"] + search_catalog["fsk"]
        presentation_gate = 0.18

    for method, series, decoder_fn, decimations, smooth_windows, min_confidence in search_plan:
        if series.size < 512:
            continue
        for decimation in decimations:
            if series.size // decimation < 768:
                continue
            view = series[::decimation]
            effective_rate = decode_rate / decimation
            for smooth_window in smooth_windows:
                candidate = decoder_fn(
                    view,
                    effective_rate,
                    method,
                    smooth_window=smooth_window,
                    min_confidence=min_confidence,
                )
                if candidate is None:
                    continue
                candidate.notes += f" Decim={decimation}."
                candidates.append(candidate)

    if not candidates:
        return empty_decode_preview(
            f"{decoder_mode_label(mode)} found no usable candidate after burst centering and narrowing.",
        )

    best = max(candidates, key=lambda candidate: candidate.confidence)
    if best.confidence < presentation_gate:
        return empty_decode_preview(
            f"{decoder_mode_label(mode)} saw a weak candidate, but confidence stayed below the presentation threshold.",
        )
    preview = candidate_to_preview(best)
    if mode.startswith("auto"):
        preview.notes = f"{preview.notes} Best of {len(candidates)} decoder candidates in {decoder_mode_label(mode)}."
    return preview


def compute_spectrum(iq: np.ndarray, sample_rate: float):
    analysis_iq = contiguous_view(iq, MAX_ANALYSIS_SAMPLES)
    if analysis_iq.size < 256:
        freq_axis = np.linspace(-sample_rate / 2.0, sample_rate / 2.0, 256, endpoint=False, dtype=np.float32)
        spectrum_db = np.full(256, -140.0, dtype=np.float32)
        return freq_axis, spectrum_db

    nperseg = min(DETAIL_FFT_BINS, int(analysis_iq.size))
    noverlap = nperseg // 2
    freq_axis, spectrum = signal.welch(
        analysis_iq,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        scaling="spectrum",
    )
    freq_axis = np.fft.fftshift(freq_axis)
    spectrum = np.fft.fftshift(spectrum)
    spectrum_db = 10.0 * np.log10(np.maximum(spectrum, 1e-15))
    return freq_axis.astype(np.float32), spectrum_db.astype(np.float32)


def estimate_occupied_bandwidth(freq_axis_hz: np.ndarray, spectrum_db: np.ndarray, peak_index: int, noise_floor_db: float, snr_db: float) -> float:
    if spectrum_db.size == 0:
        return 0.0
    gate_db = noise_floor_db + max(3.0, min(6.0, snr_db * 0.5))
    left = peak_index
    right = peak_index
    while left > 0 and spectrum_db[left] >= gate_db:
        left -= 1
    while right < spectrum_db.size - 1 and spectrum_db[right] >= gate_db:
        right += 1
    if right <= left:
        return 0.0
    return float(abs(freq_axis_hz[right] - freq_axis_hz[left]))


def extract_signal_metrics(freq_axis_hz: np.ndarray, spectrum_db: np.ndarray):
    if spectrum_db.size == 0:
        return -140.0, -140.0, 0.0, 0.0, 0.0

    noise_floor_db = float(np.percentile(spectrum_db, 60))
    peaks, props = signal.find_peaks(spectrum_db, prominence=3.0)
    if peaks.size:
        prominences = props.get("prominences", np.zeros(peaks.size, dtype=np.float32))
        best_peak = int(peaks[int(np.argmax(prominences))])
        prominence_db = float(prominences[int(np.argmax(prominences))])
    else:
        best_peak = int(np.argmax(spectrum_db))
        prominence_db = float(spectrum_db[best_peak] - noise_floor_db)

    peak_db = float(spectrum_db[best_peak])
    dominant_offset_hz = float(freq_axis_hz[best_peak])
    snr_db = max(float(peak_db - noise_floor_db), prominence_db, 0.0)
    occupied_bw_hz = estimate_occupied_bandwidth(freq_axis_hz, spectrum_db, best_peak, noise_floor_db, snr_db)
    return peak_db, noise_floor_db, snr_db, dominant_offset_hz, occupied_bw_hz


def compute_detail_waterfall(iq: np.ndarray, sample_rate: float):
    detail_iq, detail_rate = stride_view(contiguous_view(iq, MAX_DETAIL_WATERFALL_SAMPLES), MAX_DETAIL_WATERFALL_SAMPLES, sample_rate)
    if detail_iq.size < 512:
        return (
            np.linspace(-detail_rate / 2.0, detail_rate / 2.0, 128, endpoint=False, dtype=np.float32),
            np.linspace(0.0, 1.0, 4, dtype=np.float32),
            np.full((4, 128), -140.0, dtype=np.float32),
        )

    nperseg = min(1024, max(256, int(detail_iq.size // 24)))
    noverlap = int(nperseg * 0.75)
    freq_axis, time_axis, spectrum = signal.spectrogram(
        detail_iq,
        fs=detail_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,
        mode="magnitude",
    )
    freq_axis = np.fft.fftshift(freq_axis)
    spectrum = np.fft.fftshift(spectrum, axes=0)
    spectrum_db = 20.0 * np.log10(np.maximum(spectrum, 1e-12))
    return freq_axis.astype(np.float32), time_axis.astype(np.float32), spectrum_db.T.astype(np.float32)


def compute_waveform_view(iq: np.ndarray, sample_rate: float):
    waveform_iq, waveform_rate = stride_view(iq, MAX_WAVEFORM_POINTS, sample_rate)
    time_axis_ms = (np.arange(waveform_iq.size, dtype=np.float32) / waveform_rate) * 1e3
    return (
        time_axis_ms,
        waveform_iq.real.astype(np.float32, copy=False),
        waveform_iq.imag.astype(np.float32, copy=False),
    )


def compute_constellation_view(iq: np.ndarray):
    constellation_iq, _ = stride_view(iq, MAX_CONSTELLATION_POINTS, 1.0)
    return np.column_stack(
        (
            constellation_iq.real.astype(np.float32, copy=False),
            constellation_iq.imag.astype(np.float32, copy=False),
        )
    )


def analyze_capture(
    iq: np.ndarray,
    center_freq_hz: float,
    sample_rate: float,
    snr_gate_db: float,
    decoder_mode: str,
    step_index: int,
    total_steps: int,
    sweep_number: int,
) -> SweepSnapshot:
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    avg_power_db = 10.0 * np.log10(np.mean(np.abs(iq) ** 2) + 1e-15)
    freq_axis_hz, spectrum_db = compute_spectrum(iq, sample_rate)
    peak_db, noise_floor_db, snr_db, dominant_offset_hz, occupied_bw_hz = extract_signal_metrics(freq_axis_hz, spectrum_db)
    detail_freq_hz, detail_time_s, detail_waterfall_db = compute_detail_waterfall(iq, sample_rate)
    waveform_time_ms, waveform_i, waveform_q = compute_waveform_view(iq, sample_rate)
    constellation = compute_constellation_view(iq)
    if decoder_mode == "off":
        decode_preview = empty_decode_preview("Decoder disabled by selected mode.", method="Decoder Off")
    elif snr_db >= max(10.0, snr_gate_db * 0.7):
        decode_preview = decode_signal_preview(iq, sample_rate, dominant_offset_hz, occupied_bw_hz, decoder_mode)
    else:
        decode_preview = empty_decode_preview("Decoder skipped because the capture did not clear the confidence gate.")

    return SweepSnapshot(
        timestamp_utc=timestamp_utc,
        center_freq_hz=float(center_freq_hz),
        sample_rate=float(sample_rate),
        decoder_mode=str(decoder_mode or DEFAULT_DECODER_MODE),
        step_index=int(step_index),
        total_steps=int(total_steps),
        sweep_number=int(sweep_number),
        avg_power_db=float(avg_power_db),
        peak_db=float(peak_db),
        noise_floor_db=float(noise_floor_db),
        snr_db=float(snr_db),
        occupied_bw_hz=float(occupied_bw_hz),
        dominant_offset_hz=float(dominant_offset_hz),
        detected=bool(snr_db >= snr_gate_db),
        recording_path="",
        freq_axis_hz=freq_axis_hz,
        spectrum_db=spectrum_db,
        detail_waterfall_freq_hz=detail_freq_hz,
        detail_waterfall_time_s=detail_time_s,
        detail_waterfall_db=detail_waterfall_db,
        waveform_time_ms=waveform_time_ms,
        waveform_i=waveform_i,
        waveform_q=waveform_q,
        constellation=constellation,
        decode_preview=decode_preview,
    )


def save_detection_capture(iq: np.ndarray, snapshot: SweepSnapshot, output_dir: str) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_name = sanitize_filename(
        f"{stamp}_{snapshot.center_freq_hz / 1e6:.3f}MHz_snr{snapshot.snr_db:.1f}.npz"
    )
    target = out_dir / file_name
    cf32_target = target.with_suffix(".cf32")
    temp_target = target.with_name(f"{target.stem}.tmp.npz")
    temp_cf32_target = cf32_target.with_name(f"{cf32_target.name}.tmp")
    interleaved = np.empty(iq.size * 2, dtype=np.float32)
    interleaved[0::2] = iq.real.astype(np.float32, copy=False)
    interleaved[1::2] = iq.imag.astype(np.float32, copy=False)
    interleaved.tofile(temp_cf32_target)
    os.replace(temp_cf32_target, cf32_target)
    np.savez_compressed(
        temp_target,
        iq=iq.astype(np.complex64, copy=False),
        timestamp_utc=np.array(snapshot.timestamp_utc),
        center_freq_hz=np.float64(snapshot.center_freq_hz),
        sample_rate=np.float64(snapshot.sample_rate),
        decoder_mode=np.array(snapshot.decoder_mode),
        avg_power_db=np.float32(snapshot.avg_power_db),
        peak_db=np.float32(snapshot.peak_db),
        noise_floor_db=np.float32(snapshot.noise_floor_db),
        snr_db=np.float32(snapshot.snr_db),
        occupied_bw_hz=np.float32(snapshot.occupied_bw_hz),
        dominant_offset_hz=np.float32(snapshot.dominant_offset_hz),
        decoder_method=np.array(snapshot.decode_preview.method),
        decoder_symbol_rate_hz=np.float32(snapshot.decode_preview.symbol_rate_hz),
        decoder_confidence=np.float32(snapshot.decode_preview.confidence),
        decoder_hex_preview=np.array(snapshot.decode_preview.hex_preview),
        decoder_ascii_preview=np.array(snapshot.decode_preview.ascii_preview),
        cf32_path=np.array(str(cf32_target)),
    )
    os.replace(temp_target, target)
    return str(target)


# ========================== CAPTURE =========================
def record_iq_once(freq_hz: float, sample_rate: float, seconds: float, gain: int):
    samples = int(sample_rate * seconds)
    if samples <= 0:
        return None

    cmd = [
        "hackrf_transfer",
        "-r",
        TMP_FILE,
        "-f",
        str(int(freq_hz)),
        "-s",
        str(int(sample_rate)),
        "-g",
        str(int(gain)),
        "-n",
        str(samples),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if not os.path.exists(TMP_FILE):
            return None
        raw = np.fromfile(TMP_FILE, dtype=np.int8)
        os.remove(TMP_FILE)
        if raw.size < 2:
            return None
        i_vals = raw[0::2].astype(np.float32)
        q_vals = raw[1::2].astype(np.float32)
        return ((i_vals + 1j * q_vals) / 128.0).astype(np.complex64, copy=False)
    except Exception:
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
        self.driver_key = normalize_driver_key(getattr(self.dev, "getDriverKey", lambda: "")())
        self.profile = device_profile_for_driver(self.driver_key)
        self.frequency_bounds = self._read_frequency_bounds()

    def _read_frequency_bounds(self):
        try:
            bounds = soapy_range_bounds(self.dev.getFrequencyRange(SOAPY_SDR_RX, 0))
            if bounds is not None:
                return bounds
        except Exception:
            pass
        try:
            names = self.dev.listFrequencies(SOAPY_SDR_RX, 0)
            for name in names:
                bounds = soapy_range_bounds(self.dev.getFrequencyRange(SOAPY_SDR_RX, 0, name))
                if bounds is not None:
                    return bounds
        except Exception:
            pass
        return (
            float(self.profile.get("min_freq_hz", DEFAULT_START_FREQ)),
            float(self.profile.get("max_freq_hz", DEFAULT_END_FREQ)),
        )

    def validate_sweep(self, frequency_plan_hz: np.ndarray, requested_sample_rate: float) -> float:
        lo_hz, hi_hz = self.frequency_bounds
        start = float(frequency_plan_hz[0])
        end = float(frequency_plan_hz[-1])
        if start < lo_hz or end > hi_hz:
            raise ValueError(
                f"{self.profile['display']} supports {human_freq(lo_hz)} to {human_freq(hi_hz)}. "
                f"Requested sweep {human_freq(start)} to {human_freq(end)} is invalid for this radio."
            )
        return soapy_pick_sample_rate(self.dev, requested_sample_rate)

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
                except Exception:
                    pass
                self.gain = gain
            self.dev.setFrequency(SOAPY_SDR_RX, 0, freq_hz)
        except Exception:
            return None

        buff = np.empty(samples, np.complex64)
        read = 0
        timeout_us = 100000
        deadline = time.time() + max(1.0, seconds * 2.0)
        while read < samples and time.time() < deadline:
            result = self.dev.readStream(self.stream, [buff[read:]], samples - read, timeout_us)
            if result.ret > 0:
                read += result.ret
            elif result.ret == SOAPY_SDR_TIMEOUT:
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
class SpectrumCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(15.5, 9.5), facecolor="#07111a")
        grid = self.fig.add_gridspec(
            3,
            2,
            height_ratios=[1.0, 1.18, 1.06],
            hspace=0.26,
            wspace=0.14,
            left=0.055,
            right=0.992,
            top=0.975,
            bottom=0.065,
        )
        self.ax_overview = self.fig.add_subplot(grid[0, 0], facecolor="#0c1722")
        self.ax_detail_psd = self.fig.add_subplot(grid[0, 1], facecolor="#0c1722")
        self.ax_sweep_waterfall = self.fig.add_subplot(grid[1, 0], facecolor="#0c1722")
        self.ax_detail_waterfall = self.fig.add_subplot(grid[1, 1], facecolor="#0c1722")
        self.ax_waveform = self.fig.add_subplot(grid[2, 0], facecolor="#0c1722")
        self.ax_constellation = self.fig.add_subplot(grid[2, 1], facecolor="#0c1722")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self._axes = [
            self.ax_overview,
            self.ax_detail_psd,
            self.ax_sweep_waterfall,
            self.ax_detail_waterfall,
            self.ax_waveform,
            self.ax_constellation,
        ]
        for axis in self._axes:
            self._style_axis(axis)

        self.start_freq_hz = None
        self.end_freq_hz = None
        self.render_freqs_hz = np.array([], dtype=np.float32)
        self.current_sweep_number = None
        self.current_sweep_values = None
        self.current_sweep_hits = None
        self.peak_hold_values = None
        self.sweep_history = deque(maxlen=SWEEP_HISTORY)
        self.latest_snapshot: Optional[SweepSnapshot] = None
        self._last_redraw_ts = 0.0
        self.redraw_interval_s = 0.16
        self._draw_empty_state()

    def _style_axis(self, axis):
        axis.tick_params(colors="#cbd5e1", labelsize=9, pad=2)
        for spine in axis.spines.values():
            spine.set_color("#30465a")
        axis.grid(True, color="#223547", alpha=0.28, linewidth=0.7)
        axis.title.set_color("#f8fafc")
        axis.title.set_fontsize(11)
        axis.xaxis.label.set_color("#cbd5e1")
        axis.yaxis.label.set_color("#cbd5e1")
        axis.xaxis.labelpad = 4
        axis.yaxis.labelpad = 4

    def reset(self, frequency_plan_hz: np.ndarray):
        self.start_freq_hz = float(frequency_plan_hz[0])
        self.end_freq_hz = float(frequency_plan_hz[-1])
        render_bins = min(len(frequency_plan_hz), SWEEP_RENDER_BINS)
        self.render_freqs_hz = np.linspace(self.start_freq_hz, self.end_freq_hz, render_bins, dtype=np.float32)
        self.current_sweep_number = None
        self.current_sweep_values = np.full(render_bins, np.nan, dtype=np.float32)
        self.current_sweep_hits = np.zeros(render_bins, dtype=bool)
        self.peak_hold_values = np.full(render_bins, np.nan, dtype=np.float32)
        self.sweep_history.clear()
        self.latest_snapshot = None
        self._last_redraw_ts = 0.0
        self.redraw()

    def load_recording_snapshot(self, snapshot: SweepSnapshot):
        span_hz = max(float(snapshot.sample_rate), float(snapshot.occupied_bw_hz) * 12.0, 25e3)
        self.start_freq_hz = float(snapshot.center_freq_hz - span_hz * 0.5)
        self.end_freq_hz = float(snapshot.center_freq_hz + span_hz * 0.5)
        render_bins = 128
        self.render_freqs_hz = np.linspace(self.start_freq_hz, self.end_freq_hz, render_bins, dtype=np.float32)
        self.current_sweep_number = snapshot.sweep_number
        self.current_sweep_values = np.full(render_bins, float(snapshot.noise_floor_db), dtype=np.float32)
        self.current_sweep_hits = np.zeros(render_bins, dtype=bool)
        center_index = render_bins // 2
        self.current_sweep_values[center_index] = float(snapshot.peak_db)
        self.current_sweep_hits[center_index] = bool(snapshot.detected)
        self.peak_hold_values = self.current_sweep_values.copy()
        self.sweep_history.clear()
        self.sweep_history.append(self.current_sweep_values.copy())
        self.latest_snapshot = snapshot
        self._last_redraw_ts = 0.0
        self.redraw()

    def _render_index(self, center_freq_hz: float) -> int:
        if self.render_freqs_hz.size <= 1:
            return 0
        span = max(self.end_freq_hz - self.start_freq_hz, 1.0)
        ratio = (center_freq_hz - self.start_freq_hz) / span
        return int(np.clip(round(ratio * (self.render_freqs_hz.size - 1)), 0, self.render_freqs_hz.size - 1))

    def _push_completed_sweep(self):
        if self.current_sweep_values is None:
            return
        if not np.any(~np.isnan(self.current_sweep_values)):
            return
        sweep = self.current_sweep_values.copy()
        sweep_floor = float(np.nanpercentile(sweep, 10)) if np.any(~np.isnan(sweep)) else -140.0
        sweep = np.nan_to_num(sweep, nan=sweep_floor)
        self.sweep_history.append(sweep.astype(np.float32))

    def consume_snapshot(self, snapshot: SweepSnapshot):
        if self.render_freqs_hz.size == 0 or self.current_sweep_values is None:
            return

        if self.current_sweep_number is None:
            self.current_sweep_number = snapshot.sweep_number
        elif snapshot.sweep_number != self.current_sweep_number:
            self._push_completed_sweep()
            self.current_sweep_values = np.full(self.render_freqs_hz.size, np.nan, dtype=np.float32)
            self.current_sweep_hits = np.zeros(self.render_freqs_hz.size, dtype=bool)
            self.current_sweep_number = snapshot.sweep_number

        index = self._render_index(snapshot.center_freq_hz)
        if np.isnan(self.current_sweep_values[index]) or snapshot.peak_db > self.current_sweep_values[index]:
            self.current_sweep_values[index] = snapshot.peak_db
        self.current_sweep_hits[index] = self.current_sweep_hits[index] or snapshot.detected
        if self.peak_hold_values is not None:
            if np.isnan(self.peak_hold_values[index]) or snapshot.peak_db > self.peak_hold_values[index]:
                self.peak_hold_values[index] = snapshot.peak_db
        self.latest_snapshot = snapshot
        now = time.monotonic()
        should_redraw = (
            snapshot.detected
            or snapshot.step_index == snapshot.total_steps - 1
            or (now - self._last_redraw_ts) >= self.redraw_interval_s
        )
        if should_redraw:
            self.redraw()
            self._last_redraw_ts = now

    def _draw_empty_state(self):
        for axis in self._axes:
            axis.clear()
            self._style_axis(axis)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.text(
                0.5,
                0.5,
                "Start a sweep to populate this view",
                ha="center",
                va="center",
                color="#94a3b8",
                fontsize=11,
                transform=axis.transAxes,
            )
        self.draw_idle()

    def redraw(self):
        if self.render_freqs_hz.size == 0:
            self._draw_empty_state()
            return

        self._draw_overview()
        self._draw_sweep_waterfall()
        self._draw_detail_psd()
        self._draw_detail_waterfall()
        self._draw_waveform()
        self._draw_constellation()
        self.draw_idle()

    def _draw_overview(self):
        axis = self.ax_overview
        axis.clear()
        self._style_axis(axis)
        x_axis_mhz = self.render_freqs_hz / 1e6
        values = self.current_sweep_values
        peak_hold = self.peak_hold_values
        if peak_hold is not None and np.any(~np.isnan(peak_hold)):
            valid_hold = ~np.isnan(peak_hold)
            axis.plot(
                x_axis_mhz[valid_hold],
                peak_hold[valid_hold],
                color="#94a3b8",
                linewidth=0.95,
                linestyle=":",
                alpha=0.8,
                label="Peak hold",
            )
        if values is not None and np.any(~np.isnan(values)):
            valid = ~np.isnan(values)
            axis.plot(x_axis_mhz[valid], values[valid], color="#38bdf8", linewidth=1.3, label="Current sweep")
            hit_x = x_axis_mhz[self.current_sweep_hits]
            hit_y = values[self.current_sweep_hits]
            if hit_x.size:
                axis.scatter(hit_x, hit_y, s=28, color="#f59e0b", edgecolors="#fde68a", linewidths=0.5, zorder=3)
        if self.latest_snapshot is not None:
            axis.axvline(self.latest_snapshot.center_freq_hz / 1e6, color="#f97316", linestyle="--", linewidth=1.0, alpha=0.75)

        axis.set_title("Sweep Overview", pad=6)
        axis.set_xlabel("Center Frequency (MHz)")
        axis.set_ylabel("Peak Spectral Power (dB)")
        if (peak_hold is not None and np.any(~np.isnan(peak_hold))) or (values is not None and np.any(~np.isnan(values))):
            axis.legend(loc="lower left", facecolor="#07111a", edgecolor="#223547", labelcolor="#cbd5e1", fontsize=8)
        if self.latest_snapshot is not None:
            axis.text(
                0.01,
                0.95,
                f"Sweep {self.latest_snapshot.sweep_number + 1}  |  Step {self.latest_snapshot.step_index + 1}/{self.latest_snapshot.total_steps}",
                transform=axis.transAxes,
                va="top",
                color="#cbd5e1",
                fontsize=9,
                bbox={"facecolor": "#07111a", "edgecolor": "#223547", "boxstyle": "round,pad=0.25"},
            )

    def _draw_sweep_waterfall(self):
        axis = self.ax_sweep_waterfall
        axis.clear()
        self._style_axis(axis)
        rows = list(self.sweep_history)
        if self.current_sweep_values is not None and np.any(~np.isnan(self.current_sweep_values)):
            current = self.current_sweep_values.copy()
            current_floor = float(np.nanpercentile(current, 10)) if np.any(~np.isnan(current)) else -140.0
            rows.append(np.nan_to_num(current, nan=current_floor))

        if rows:
            matrix = np.vstack(rows)
            vmin = float(np.percentile(matrix, 5))
            vmax = float(np.percentile(matrix, 95))
            axis.imshow(
                matrix,
                aspect="auto",
                origin="lower",
                cmap="magma",
                extent=[self.start_freq_hz / 1e6, self.end_freq_hz / 1e6, 0, matrix.shape[0]],
                vmin=vmin,
                vmax=vmax,
            )
        else:
            axis.text(0.5, 0.5, "Waiting for sweep rows", ha="center", va="center", color="#94a3b8", transform=axis.transAxes)

        axis.set_title("Sweep Waterfall", pad=6)
        axis.set_xlabel("Center Frequency (MHz)")
        axis.set_ylabel("Completed Sweeps")

    def _draw_detail_psd(self):
        axis = self.ax_detail_psd
        axis.clear()
        self._style_axis(axis)
        if self.latest_snapshot is None:
            axis.text(0.5, 0.5, "No capture yet", ha="center", va="center", color="#94a3b8", transform=axis.transAxes)
            axis.set_title("Detail PSD", pad=6)
            return

        snapshot = self.latest_snapshot
        abs_axis_mhz = (snapshot.center_freq_hz + snapshot.freq_axis_hz) / 1e6
        axis.plot(abs_axis_mhz, snapshot.spectrum_db, color="#22c55e", linewidth=1.1)
        axis.axvline((snapshot.center_freq_hz + snapshot.dominant_offset_hz) / 1e6, color="#f59e0b", linestyle="--", linewidth=1.0)
        axis.set_title(f"Detail PSD @ {human_freq(snapshot.center_freq_hz)}", pad=6)
        axis.set_xlabel("Frequency (MHz)")
        axis.set_ylabel("Power (dB)")
        axis.text(
            0.01,
            0.98,
            f"SNR {snapshot.snr_db:.1f} dB\nBW {human_bandwidth(snapshot.occupied_bw_hz)}\nPeak {snapshot.peak_db:.1f} dB",
            transform=axis.transAxes,
            va="top",
            color="#e2e8f0",
            fontsize=9,
            bbox={"facecolor": "#07111a", "edgecolor": "#223547", "boxstyle": "round,pad=0.3"},
        )

    def _draw_detail_waterfall(self):
        axis = self.ax_detail_waterfall
        axis.clear()
        self._style_axis(axis)
        if self.latest_snapshot is None or self.latest_snapshot.detail_waterfall_db.size == 0:
            axis.text(0.5, 0.5, "No detail waterfall yet", ha="center", va="center", color="#94a3b8", transform=axis.transAxes)
            axis.set_title("Detail Waterfall", pad=6)
            return

        snapshot = self.latest_snapshot
        freq_axis_mhz = (snapshot.center_freq_hz + snapshot.detail_waterfall_freq_hz) / 1e6
        time_axis_ms = snapshot.detail_waterfall_time_s * 1e3
        axis.imshow(
            snapshot.detail_waterfall_db,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[freq_axis_mhz[0], freq_axis_mhz[-1], 0, time_axis_ms[-1] if time_axis_ms.size else 1],
        )
        axis.set_title("In-Capture Waterfall", pad=6)
        axis.set_xlabel("Frequency (MHz)")
        axis.set_ylabel("Time (ms)")

    def _draw_waveform(self):
        axis = self.ax_waveform
        axis.clear()
        self._style_axis(axis)
        if self.latest_snapshot is None or self.latest_snapshot.waveform_time_ms.size == 0:
            axis.text(0.5, 0.5, "No waveform yet", ha="center", va="center", color="#94a3b8", transform=axis.transAxes)
            axis.set_title("IQ Waveform", pad=6)
            return

        snapshot = self.latest_snapshot
        axis.plot(snapshot.waveform_time_ms, snapshot.waveform_i, color="#38bdf8", linewidth=0.95, label="I")
        axis.plot(snapshot.waveform_time_ms, snapshot.waveform_q, color="#f59e0b", linewidth=0.95, alpha=0.85, label="Q")
        axis.set_title("IQ Waveform", pad=6)
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("Amplitude")
        axis.legend(loc="upper right", facecolor="#07111a", edgecolor="#223547", labelcolor="#e2e8f0")

    def _draw_constellation(self):
        axis = self.ax_constellation
        axis.clear()
        self._style_axis(axis)
        if self.latest_snapshot is None or self.latest_snapshot.constellation.size == 0:
            axis.text(0.5, 0.5, "No constellation yet", ha="center", va="center", color="#94a3b8", transform=axis.transAxes)
            axis.set_title("Constellation", pad=6)
            return

        points = self.latest_snapshot.constellation
        axis.scatter(points[:, 0], points[:, 1], s=4, color="#22c55e", alpha=0.28)
        scale = max(0.25, min(np.max(np.abs(points)) * 1.25, 3.0))
        axis.set_xlim(-scale, scale)
        axis.set_ylim(-scale, scale)
        axis.set_title("Constellation", pad=6)
        axis.set_xlabel("I")
        axis.set_ylabel("Q")


# ====================== SCAN THREAD ========================
class SweepThread(QtCore.QThread):
    status = QtCore.pyqtSignal(str)
    snapshot = QtCore.pyqtSignal(object)
    detection = QtCore.pyqtSignal(object)
    pause_state = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        frequency_plan_hz: np.ndarray,
        sample_rate_hz: float,
        capture_secs: float,
        gain: int,
        snr_gate_db: float,
        log_csv: str,
        backend: str,
        soapy_args: str,
        record_detections: bool,
        record_dir: str,
        record_cooldown_s: float,
        pause_on_detection: bool,
        decoder_mode: str,
    ):
        super().__init__()
        self.frequency_plan_hz = frequency_plan_hz
        self.sample_rate_hz = float(sample_rate_hz)
        self.capture_secs = float(capture_secs)
        self.gain = int(gain)
        self.snr_gate_db = float(snr_gate_db)
        self.log_csv = log_csv
        self.backend = backend
        self.soapy_args = soapy_args
        self.record_detections = record_detections
        self.record_dir = record_dir
        self.record_cooldown_s = float(record_cooldown_s)
        self.pause_on_detection = pause_on_detection
        self.decoder_mode = str(decoder_mode or DEFAULT_DECODER_MODE)

        self.running = False
        self.paused = False
        self._recent_recordings = {}

    def _advance_position(self, step_index: int, sweep_number: int):
        step_index += 1
        if step_index >= len(self.frequency_plan_hz):
            step_index = 0
            sweep_number += 1
        return step_index, sweep_number

    def _should_record(self, center_freq_hz: float) -> bool:
        now = time.monotonic()
        bucket = int(round(center_freq_hz / max(self.sample_rate_hz, 1.0)))
        last = self._recent_recordings.get(bucket, 0.0)
        if now - last < self.record_cooldown_s:
            return False
        self._recent_recordings[bucket] = now
        return True

    def run(self):
        self.running = True
        backend_pref = (self.backend or DEFAULT_BACKEND).strip().lower()
        backend = None
        capture = None
        active_profile = DEVICE_PROFILES["generic"]

        if backend_pref == "hackrf":
            if not have_hackrf():
                self.status.emit("hackrf_transfer not found in PATH.")
                return
            backend = "hackrf"
            active_profile = DEVICE_PROFILES["hackrf"]
            validate_frequency_plan(active_profile, self.frequency_plan_hz)
            self.status.emit("Using HackRF backend")
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
                        active_profile = capture.profile
                        desc = chosen_args.get("label") or chosen_args.get("driver") or capture.profile["display"]
                        self.status.emit(f"Using SoapySDR backend ({desc})")
                    except Exception as exc:
                        capture = None
                        if backend_pref == "soapy":
                            self.status.emit(f"SoapySDR init failed: {exc}")
                            return

            if backend is None:
                if backend_pref == "soapy":
                    self.status.emit("No SoapySDR device found.")
                    return
                if have_hackrf():
                    backend = "hackrf"
                    active_profile = DEVICE_PROFILES["hackrf"]
                    validate_frequency_plan(active_profile, self.frequency_plan_hz)
                    self.status.emit("Using HackRF backend")
                else:
                    self.status.emit("No SDR backend found (HackRF or SoapySDR).")
                    return
        else:
            self.status.emit(f"Unknown backend selection: {self.backend}")
            return

        if capture is not None:
            try:
                adjusted_rate = capture.validate_sweep(self.frequency_plan_hz, self.sample_rate_hz)
            except ValueError as exc:
                self.status.emit(str(exc))
                return
            if abs(adjusted_rate - self.sample_rate_hz) > 1.0:
                self.sample_rate_hz = adjusted_rate
                self.status.emit(
                    f"{capture.profile['display']} adjusted sample rate to {human_rate(self.sample_rate_hz)}"
                )
        else:
            if active_profile["display"] != DEVICE_PROFILES["generic"]["display"]:
                try:
                    validate_frequency_plan(active_profile, self.frequency_plan_hz)
                except ValueError as exc:
                    self.status.emit(str(exc))
                    return

        step_index = 0
        sweep_number = 0
        total_steps = len(self.frequency_plan_hz)

        try:
            with open(self.log_csv, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "timestamp_utc",
                        "center_freq_hz",
                        "avg_power_db",
                        "peak_db",
                        "noise_floor_db",
                        "snr_db",
                        "occupied_bw_hz",
                        "dominant_offset_hz",
                        "detected",
                        "decoder_mode",
                        "decoder_method",
                        "decoder_symbol_rate_hz",
                        "decoder_confidence",
                        "decoder_hex_preview",
                        "recording_path",
                    ]
                )

                while self.running:
                    if self.paused:
                        self.msleep(100)
                        continue

                    center_freq_hz = float(self.frequency_plan_hz[step_index])
                    self.status.emit(
                        f"Sweeping {human_freq(center_freq_hz)}  |  {step_index + 1}/{total_steps}  |  {human_rate(self.sample_rate_hz)}"
                    )

                    if backend == "hackrf":
                        iq = record_iq_once(center_freq_hz, self.sample_rate_hz, self.capture_secs, self.gain)
                    else:
                        iq = capture.record(center_freq_hz, self.sample_rate_hz, self.capture_secs, self.gain)
                        if capture.sample_rate and capture.sample_rate != self.sample_rate_hz:
                            self.sample_rate_hz = capture.sample_rate
                            self.status.emit(f"Adjusted to device sample rate {human_rate(self.sample_rate_hz)}")

                    if iq is None or iq.size < 256:
                        self.status.emit(f"Capture failed @ {human_freq(center_freq_hz)}")
                        step_index, sweep_number = self._advance_position(step_index, sweep_number)
                        continue

                    snapshot = analyze_capture(
                        iq=iq,
                        center_freq_hz=center_freq_hz,
                        sample_rate=self.sample_rate_hz,
                        snr_gate_db=self.snr_gate_db,
                        decoder_mode=self.decoder_mode,
                        step_index=step_index,
                        total_steps=total_steps,
                        sweep_number=sweep_number,
                    )

                    if snapshot.detected and self.record_detections and self._should_record(center_freq_hz):
                        try:
                            snapshot.recording_path = save_detection_capture(iq, snapshot, self.record_dir)
                        except Exception as exc:
                            self.status.emit(f"Failed to save IQ clip: {exc}")

                    writer.writerow(
                        [
                            snapshot.timestamp_utc,
                            int(snapshot.center_freq_hz),
                            round(snapshot.avg_power_db, 3),
                            round(snapshot.peak_db, 3),
                            round(snapshot.noise_floor_db, 3),
                            round(snapshot.snr_db, 3),
                            round(snapshot.occupied_bw_hz, 3),
                            round(snapshot.dominant_offset_hz, 3),
                            int(snapshot.detected),
                            snapshot.decoder_mode,
                            snapshot.decode_preview.method,
                            round(snapshot.decode_preview.symbol_rate_hz, 3),
                            round(snapshot.decode_preview.confidence, 3),
                            snapshot.decode_preview.hex_preview,
                            snapshot.recording_path,
                        ]
                    )
                    handle.flush()

                    self.snapshot.emit(snapshot)

                    if snapshot.detected:
                        hit_message = (
                            f"Hit @ {human_freq(snapshot.center_freq_hz)}  |  "
                            f"SNR {snapshot.snr_db:.1f} dB  |  "
                            f"BW {human_bandwidth(snapshot.occupied_bw_hz)}"
                        )
                        if snapshot.recording_path:
                            hit_message += f"  |  saved {Path(snapshot.recording_path).name}"
                        self.status.emit(hit_message)
                        self.detection.emit(snapshot)
                        if self.pause_on_detection:
                            self.paused = True
                            self.pause_state.emit(True)
                            self.status.emit(f"Paused on detection @ {human_freq(snapshot.center_freq_hz)}")

                    step_index, sweep_number = self._advance_position(step_index, sweep_number)
        finally:
            if capture:
                capture.close()

    def stop(self):
        self.running = False
        self.paused = False
        self.pause_state.emit(False)

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_state.emit(self.paused)


# ========================= UI ==============================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.thread: Optional[SweepThread] = None
        self._backend_detected = have_hackrf() or have_soapy()
        self._sidebar_visible = True
        self._sidebar_width = FIXED_SIDEBAR_WIDTH
        self._workspace_mode = "all"
        self._detection_count = 0
        self.latest_snapshot: Optional[SweepSnapshot] = None
        self.latest_recording_path = ""
        self.latest_cf32_path = ""
        self._decode_banner_full_text = "Decoder: idle  |  Hex: —"

        self.setWindowTitle("DroneDetect Spectrum Inspector")
        self.setMinimumSize(1520, 940)
        self.resize(1680, 1020)
        self._apply_palette()
        self._apply_stylesheet()

        self.canvas = SpectrumCanvas(self)
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Idle")

        self._build_ui()
        self._set_decode_banner_text(self._decode_banner_full_text)
        self.apply_device_defaults(force=True)
        self.refresh_profile_hint()
        self.refresh_recordings_browser()
        self._refresh_scan_meta()
        self._set_running_state(False)
        QtWidgets.QShortcut("Ctrl+B", self, activated=self.toggle_sidebar)

        if not self._backend_detected:
            self._set_status("No SDR backend found. Install HackRF tools or SoapySDR to scan.")

    def _apply_palette(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#07111a"))
        palette.setColor(QPalette.WindowText, QColor("#f8fafc"))
        palette.setColor(QPalette.Base, QColor("#09141d"))
        palette.setColor(QPalette.AlternateBase, QColor("#0c1722"))
        palette.setColor(QPalette.Text, QColor("#f8fafc"))
        palette.setColor(QPalette.Button, QColor("#123047"))
        palette.setColor(QPalette.ButtonText, QColor("#f8fafc"))
        self.setPalette(palette)

    def _apply_stylesheet(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #07111a;
                color: #f8fafc;
            }
            QGroupBox {
                background-color: #0c1722;
                border: 1px solid #213649;
                border-radius: 12px;
                margin-top: 10px;
                padding: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #94a3b8;
            }
            QLabel#titleLabel {
                font-size: 24px;
                font-weight: 700;
                color: #f8fafc;
            }
            QLabel#subtleLabel {
                color: #94a3b8;
            }
            QLabel#metricCaption {
                color: #94a3b8;
                font-size: 10px;
                letter-spacing: 0.08em;
            }
            QLabel#metricValue {
                color: #f59e0b;
                font: 600 15px "DejaVu Sans Mono";
            }
            QLabel#statusPill {
                background-color: #0f2232;
                border: 1px solid #244863;
                border-radius: 10px;
                padding: 8px 10px;
                color: #e2e8f0;
            }
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QTableWidget, QPlainTextEdit {
                background-color: #09141d;
                border: 1px solid #213649;
                border-radius: 8px;
                padding: 6px 8px;
                color: #f8fafc;
            }
            QPushButton {
                background-color: #123047;
                border: 1px solid #244863;
                border-radius: 8px;
                padding: 8px 14px;
                color: #f8fafc;
            }
            QPushButton:hover {
                background-color: #18435f;
            }
            QPushButton:pressed {
                background-color: #0f2232;
            }
            QPushButton:checked {
                background-color: #9a3412;
                border-color: #fb923c;
            }
            QCheckBox {
                color: #e2e8f0;
            }
            QHeaderView::section {
                background-color: #0f2232;
                color: #cbd5e1;
                border: none;
                padding: 6px;
            }
            QTableWidget {
                gridline-color: #1e293b;
            }
            QStatusBar {
                background-color: #07111a;
                color: #cbd5e1;
                border-top: 1px solid #213649;
            }
            QTabWidget::pane {
                border: 1px solid #213649;
                border-radius: 10px;
                top: -1px;
                background: #09141d;
            }
            QTabBar::tab {
                background: #0f2232;
                color: #cbd5e1;
                padding: 8px 14px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
                background: #123047;
                color: #f8fafc;
            }
            QScrollArea {
                border: none;
            }
            QProgressBar {
                border: 1px solid #213649;
                border-radius: 8px;
                text-align: center;
                background: #09141d;
                color: #f8fafc;
                min-height: 20px;
            }
            QProgressBar::chunk {
                background: #38bdf8;
                border-radius: 7px;
            }
            """
        )

    def _build_ui(self):
        root = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        root_layout.addLayout(self._build_header_bar())

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setHandleWidth(12)
        self.sidebar_scroll = self._build_left_panel()
        self.right_panel = self._build_right_panel()
        self.splitter.addWidget(self.sidebar_scroll)
        self.splitter.addWidget(self.right_panel)
        self.splitter.setChildrenCollapsible(True)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([self._sidebar_width, 1100])
        self.splitter.splitterMoved.connect(self._remember_sidebar_width)

        root_layout.addWidget(self.splitter, 1)
        self.setCentralWidget(root)
        self.set_workspace_mode("all")

    def _build_header_bar(self):
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(5)

        self.decode_banner = QtWidgets.QLabel(self._decode_banner_full_text)
        self.decode_banner.setObjectName("subtleLabel")
        self.decode_banner.setWordWrap(False)
        self.decode_banner.setMinimumWidth(0)
        self.decode_banner.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.decode_banner.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.show_all_btn = QtWidgets.QPushButton("All")
        self.show_all_btn.setCheckable(True)
        self.show_all_btn.clicked.connect(lambda: self.set_workspace_mode("all"))
        self.show_controls_btn = QtWidgets.QPushButton("Inspect")
        self.show_controls_btn.setCheckable(True)
        self.show_controls_btn.clicked.connect(self.show_controls_view)
        self.show_hex_btn = QtWidgets.QPushButton("Hex")
        self.show_hex_btn.setCheckable(True)
        self.show_hex_btn.clicked.connect(self.show_hex_view)
        self.show_graphs_btn = QtWidgets.QPushButton("Graphs")
        self.show_graphs_btn.setCheckable(True)
        self.show_graphs_btn.clicked.connect(lambda: self.set_workspace_mode("graphs"))
        self.copy_hex_btn = QtWidgets.QPushButton("Copy Hex")
        self.copy_hex_btn.clicked.connect(self.copy_current_hex)
        self.header_pause_btn = QtWidgets.QPushButton("Pause")
        self.header_pause_btn.clicked.connect(self.toggle_pause)
        self.header_stop_btn = QtWidgets.QPushButton("Stop")
        self.header_stop_btn.clicked.connect(self.stop_scan)

        header.addWidget(self.decode_banner, 1)
        header.addWidget(self.show_all_btn, 0, QtCore.Qt.AlignRight)
        header.addWidget(self.show_controls_btn, 0, QtCore.Qt.AlignRight)
        header.addWidget(self.show_hex_btn, 0, QtCore.Qt.AlignRight)
        header.addWidget(self.show_graphs_btn, 0, QtCore.Qt.AlignRight)
        header.addWidget(self.copy_hex_btn, 0, QtCore.Qt.AlignRight)
        header.addWidget(self.header_pause_btn, 0, QtCore.Qt.AlignRight)
        header.addWidget(self.header_stop_btn, 0, QtCore.Qt.AlignRight)
        return header

    def _build_left_panel(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        scroll.setMinimumWidth(MIN_SIDEBAR_WIDTH)
        scroll.setMaximumWidth(16777215)

        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(MIN_SIDEBAR_WIDTH - 28)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("DroneDetect Inspector")
        title.setObjectName("titleLabel")
        subtitle = QtWidgets.QLabel("Sweep bands, inspect waterfalls, and auto-record IQ clips when hits trigger.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("subtleLabel")
        self.status_pill = QtWidgets.QLabel("Idle")
        self.status_pill.setObjectName("statusPill")
        self.status_pill.setWordWrap(True)
        self.profile_hint_label = QtWidgets.QLabel("")
        self.profile_hint_label.setObjectName("subtleLabel")
        self.profile_hint_label.setWordWrap(True)
        self.sweep_progress = QtWidgets.QProgressBar()
        self.sweep_progress.setRange(0, 100)
        self.sweep_progress.setValue(0)
        self.sweep_progress.setFormat("Sweep progress: idle")
        self.scan_meta_label = QtWidgets.QLabel("Hits: 0  |  Clips: off")
        self.scan_meta_label.setObjectName("subtleLabel")
        self.scan_meta_label.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.status_pill)
        layout.addWidget(self.sweep_progress)
        layout.addWidget(self.scan_meta_label)
        layout.addWidget(self.profile_hint_label)

        self.sidebar_tabs = QtWidgets.QTabWidget()
        self.sidebar_tabs.addTab(self._build_controls_tab(), "Control")
        self.sidebar_tabs.addTab(self._build_inspector_tab(), "Inspect")
        self.sidebar_tabs.currentChanged.connect(lambda _: self._refresh_workspace_buttons())
        self.signal_reader_tabs.currentChanged.connect(lambda _: self._refresh_workspace_buttons())
        layout.addWidget(self.sidebar_tabs, 1)

        scroll.setWidget(panel)
        return scroll

    def _build_controls_tab(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        layout.addWidget(self._build_scan_group())
        layout.addWidget(self._build_detection_group())
        layout.addStretch(1)
        return panel

    def _build_inspector_tab(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        layout.addWidget(self._build_metrics_group())
        layout.addWidget(self._build_detections_group(), 1)
        return panel

    def _build_right_panel(self):
        panel = QtWidgets.QWidget()
        panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, 1)
        return panel

    def _build_scan_group(self):
        group = QtWidgets.QGroupBox("Scan Controls")
        layout = QtWidgets.QVBoxLayout(group)
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignLeft)
        form.setFormAlignment(QtCore.Qt.AlignTop)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["Auto", "HackRF (hackrf_transfer)", "SoapySDR"])
        self.device_combo.setMinimumWidth(230)
        if have_hackrf():
            self.device_combo.setCurrentIndex(1)

        self.soapy_args_input = QtWidgets.QLineEdit(DEFAULT_SOAPY_ARGS)
        self.soapy_args_input.setPlaceholderText("driver=rtlsdr,serial=XXXX")
        self.soapy_args_input.setMinimumWidth(230)

        self.range_input = QtWidgets.QLineEdit(DEFAULT_RANGE_TEXT)
        self.range_input.setPlaceholderText("100m-200m or 2.4g-2.5g")
        self.range_input.setMinimumWidth(230)

        step_widget = QtWidgets.QWidget()
        step_layout = QtWidgets.QHBoxLayout(step_widget)
        step_layout.setContentsMargins(0, 0, 0, 0)
        step_layout.setSpacing(6)
        self.step_input = QtWidgets.QDoubleSpinBox()
        self.step_input.setDecimals(3)
        self.step_input.setRange(0.001, 1000000.0)
        self.step_input.setValue(STEP_SIZE / 1e6)
        self.step_unit = QtWidgets.QComboBox()
        self.step_unit.addItems(["MHz", "kHz"])
        step_layout.addWidget(self.step_input, 1)
        step_layout.addWidget(self.step_unit)
        self.step_input.setMinimumWidth(120)

        self.rate_input = QtWidgets.QDoubleSpinBox()
        self.rate_input.setDecimals(3)
        self.rate_input.setRange(0.25, 50.0)
        self.rate_input.setValue(SAMPLE_RATE / 1e6)
        self.rate_input.setSuffix(" Msps")

        self.capture_input = QtWidgets.QDoubleSpinBox()
        self.capture_input.setDecimals(3)
        self.capture_input.setRange(0.02, 5.0)
        self.capture_input.setSingleStep(0.05)
        self.capture_input.setValue(RECORD_SECS)
        self.capture_input.setSuffix(" s")

        self.gain_input = QtWidgets.QSpinBox()
        self.gain_input.setRange(0, 80)
        self.gain_input.setValue(GAIN)

        form.addRow("Backend", self.device_combo)
        form.addRow("Soapy args", self.soapy_args_input)
        form.addRow("Range", self.range_input)
        form.addRow("Step", step_widget)
        form.addRow("Sample rate", self.rate_input)
        form.addRow("Capture length", self.capture_input)
        form.addRow("Gain", self.gain_input)

        buttons = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start Sweep")
        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.pause_btn)
        buttons.addWidget(self.stop_btn)

        self.safe_defaults_btn = QtWidgets.QPushButton("Safe Defaults")
        self.safe_defaults_btn.setToolTip("Apply a sane range, step size, and sample rate for the selected radio profile.")

        layout.addLayout(form)
        layout.addWidget(self.safe_defaults_btn)
        layout.addLayout(buttons)

        self.start_btn.clicked.connect(self.start_scan)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.stop_btn.clicked.connect(self.stop_scan)
        self.safe_defaults_btn.clicked.connect(self.apply_device_defaults)
        self.device_combo.currentIndexChanged.connect(self.refresh_profile_hint)
        self.soapy_args_input.editingFinished.connect(self.refresh_profile_hint)

        self._run_locked_widgets = [
            self.device_combo,
            self.soapy_args_input,
            self.range_input,
            self.step_input,
            self.step_unit,
            self.rate_input,
            self.capture_input,
            self.gain_input,
        ]
        return group

    def _build_detection_group(self):
        group = QtWidgets.QGroupBox("Detection And Recording")
        layout = QtWidgets.QFormLayout(group)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(10)

        self.decoder_mode_combo = QtWidgets.QComboBox()
        for mode, label in DECODER_MODE_OPTIONS:
            self.decoder_mode_combo.addItem(label, mode)
        decoder_index = self.decoder_mode_combo.findData(DEFAULT_DECODER_MODE)
        if decoder_index >= 0:
            self.decoder_mode_combo.setCurrentIndex(decoder_index)
        self.decoder_mode_combo.setToolTip(
            "Auto Basic stays fast with ASK/OOK and 2-FSK. Auto Advanced also tries Manchester and UART-like framing."
        )

        self.snr_gate_input = QtWidgets.QDoubleSpinBox()
        self.snr_gate_input.setDecimals(1)
        self.snr_gate_input.setRange(1.0, 60.0)
        self.snr_gate_input.setValue(DEFAULT_SNR_GATE_DB)
        self.snr_gate_input.setSuffix(" dB")

        self.record_checkbox = QtWidgets.QCheckBox("Save IQ clips on detections")
        self.record_checkbox.setChecked(True)
        self.record_checkbox.toggled.connect(self._refresh_scan_meta)

        self.pause_on_hit_checkbox = QtWidgets.QCheckBox("Pause scan when a hit is found")
        self.pause_on_hit_checkbox.setChecked(False)

        self.cooldown_input = QtWidgets.QDoubleSpinBox()
        self.cooldown_input.setDecimals(1)
        self.cooldown_input.setRange(0.0, 60.0)
        self.cooldown_input.setValue(DEFAULT_RECORD_COOLDOWN_SECS)
        self.cooldown_input.setSuffix(" s")

        self.record_dir_input = QtWidgets.QLineEdit(DEFAULT_RECORDINGS_DIR)
        self.open_recordings_btn = QtWidgets.QPushButton("Open Recordings Folder")
        self.open_recordings_btn.clicked.connect(self.open_recordings_dir)
        self.clear_recordings_btn = QtWidgets.QPushButton("Clear Recordings")
        self.clear_recordings_btn.clicked.connect(self.clear_recordings_dir)
        recordings_actions = QtWidgets.QHBoxLayout()
        recordings_actions.addWidget(self.open_recordings_btn)
        recordings_actions.addWidget(self.clear_recordings_btn)

        layout.addRow("Decoder mode", self.decoder_mode_combo)
        layout.addRow("SNR gate", self.snr_gate_input)
        layout.addRow("Record hits", self.record_checkbox)
        layout.addRow("Pause on hit", self.pause_on_hit_checkbox)
        layout.addRow("Record cooldown", self.cooldown_input)
        layout.addRow("Clip directory", self.record_dir_input)
        layout.addRow("", recordings_actions)
        self._run_locked_widgets.append(self.decoder_mode_combo)
        return group

    def _build_metrics_group(self):
        group = QtWidgets.QGroupBox("Live Metrics")
        grid = QtWidgets.QGridLayout(group)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        self.metric_labels = {}
        fields = [
            ("Center", "center"),
            ("Avg Power", "avg_power"),
            ("Peak", "peak"),
            ("Noise Floor", "noise"),
            ("SNR", "snr"),
            ("Offset", "offset"),
            ("Est. Bandwidth", "bandwidth"),
            ("Sample Rate", "sample_rate"),
            ("Decoder", "decoder"),
            ("Last Clip", "clip"),
        ]
        for row, (caption, key) in enumerate(fields):
            cap_label = QtWidgets.QLabel(caption.upper())
            cap_label.setObjectName("metricCaption")
            value_label = QtWidgets.QLabel("—")
            value_label.setObjectName("metricValue")
            value_label.setWordWrap(True)
            self.metric_labels[key] = value_label
            grid.addWidget(cap_label, row, 0)
            grid.addWidget(value_label, row, 1)
        return group

    def _build_detections_group(self):
        group = QtWidgets.QGroupBox("Signal Reader")
        layout = QtWidgets.QVBoxLayout(group)

        self.signal_reader_tabs = QtWidgets.QTabWidget()

        detections_tab = QtWidgets.QWidget()
        detections_layout = QtWidgets.QVBoxLayout(detections_tab)
        detections_layout.setContentsMargins(0, 0, 0, 0)

        self.detections_table = QtWidgets.QTableWidget(0, 6)
        self.detections_table.setHorizontalHeaderLabels(["UTC", "Center", "SNR", "BW", "Mode", "Clip"])
        self.detections_table.verticalHeader().setVisible(False)
        self.detections_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.detections_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.detections_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.detections_table.horizontalHeader().setStretchLastSection(True)
        self.detections_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.detections_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.detections_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.detections_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        self.detections_table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        detections_layout.addWidget(self.detections_table)

        recordings_tab = QtWidgets.QWidget()
        recordings_layout = QtWidgets.QVBoxLayout(recordings_tab)
        recordings_layout.setContentsMargins(0, 0, 0, 0)
        recordings_hint = QtWidgets.QLabel("Browse saved IQ clips, then load one back into the analyzer to revisit its plots and hex.")
        recordings_hint.setWordWrap(True)
        recordings_hint.setObjectName("subtleLabel")
        self.recordings_table = QtWidgets.QTableWidget(0, 6)
        self.recordings_table.setHorizontalHeaderLabels(["UTC", "Center", "SNR", "Decoder", "Conf", "File"])
        self.recordings_table.verticalHeader().setVisible(False)
        self.recordings_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.recordings_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.recordings_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.recordings_table.horizontalHeader().setStretchLastSection(True)
        self.recordings_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.recordings_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.recordings_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.recordings_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        self.recordings_table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        self.recordings_table.itemSelectionChanged.connect(self._refresh_recordings_nav)
        self.recordings_table.cellDoubleClicked.connect(lambda *_args: self.load_selected_recording())
        recordings_actions = QtWidgets.QHBoxLayout()
        self.refresh_recordings_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_recordings_btn.clicked.connect(self.refresh_recordings_browser)
        self.prev_recording_btn = QtWidgets.QPushButton("Prev")
        self.prev_recording_btn.clicked.connect(lambda: self.browse_recording(-1))
        self.load_recording_btn = QtWidgets.QPushButton("Load Selected")
        self.load_recording_btn.clicked.connect(self.load_selected_recording)
        self.next_recording_btn = QtWidgets.QPushButton("Next")
        self.next_recording_btn.clicked.connect(lambda: self.browse_recording(1))
        recordings_actions.addWidget(self.refresh_recordings_btn)
        recordings_actions.addWidget(self.prev_recording_btn)
        recordings_actions.addWidget(self.load_recording_btn)
        recordings_actions.addWidget(self.next_recording_btn)
        recordings_layout.addWidget(recordings_hint)
        recordings_layout.addWidget(self.recordings_table, 1)
        recordings_layout.addLayout(recordings_actions)

        decode_tab = QtWidgets.QWidget()
        self.decode_tab_widget = decode_tab
        decode_layout = QtWidgets.QVBoxLayout(decode_tab)
        decode_layout.setContentsMargins(0, 0, 0, 0)
        decode_hint = QtWidgets.QLabel("Burst-centered decoder preview. Auto Basic tries ASK/OOK and 2-FSK; Auto Advanced also tries Manchester and UART-like framing.")
        decode_hint.setWordWrap(True)
        decode_hint.setObjectName("subtleLabel")
        self.decode_text = QtWidgets.QPlainTextEdit()
        self.decode_text.setReadOnly(True)
        self.decode_text.setPlaceholderText("Decoded bit and hex preview will appear here when a capture looks slicable.")
        self.decode_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.decode_text.setFont(QFont("DejaVu Sans Mono", 10))
        decode_actions = QtWidgets.QHBoxLayout()
        self.copy_hex_panel_btn = QtWidgets.QPushButton("Copy Hex")
        self.copy_hex_panel_btn.clicked.connect(self.copy_current_hex)
        self.open_inspectrum_btn = QtWidgets.QPushButton("Open In Inspectrum")
        self.open_inspectrum_btn.clicked.connect(lambda: self.open_latest_analyzer("inspectrum"))
        self.open_urh_btn = QtWidgets.QPushButton("Open In URH")
        self.open_urh_btn.clicked.connect(lambda: self.open_latest_analyzer("urh"))
        self.open_inspectrum_btn.setEnabled(have_external_tool("inspectrum"))
        self.open_urh_btn.setEnabled(have_external_tool("urh"))
        decode_actions.addWidget(self.copy_hex_panel_btn)
        decode_actions.addWidget(self.open_inspectrum_btn)
        decode_actions.addWidget(self.open_urh_btn)
        decode_layout.addWidget(decode_hint)
        decode_layout.addWidget(self.decode_text, 1)
        decode_layout.addLayout(decode_actions)

        self.signal_reader_tabs.addTab(detections_tab, "Hits")
        self.signal_reader_tabs.addTab(recordings_tab, "Recordings")
        self.signal_reader_tabs.addTab(decode_tab, "Hex View")
        layout.addWidget(self.signal_reader_tabs)
        return group

    def _set_status(self, text: str):
        display_text = text if len(text) <= 96 else f"{text[:93]}..."
        self.status_pill.setText(display_text)
        self.status_bar.showMessage(text)

    def _set_running_state(self, running: bool):
        self.start_btn.setEnabled(not running and self._backend_detected)
        self.stop_btn.setEnabled(running)
        self.pause_btn.setEnabled(running)
        self.header_pause_btn.setEnabled(running)
        self.header_stop_btn.setEnabled(running)
        if not running:
            self.pause_btn.setChecked(False)
            self.pause_btn.setText("Pause")
            self.header_pause_btn.setText("Pause")
            self.sweep_progress.setValue(0)
            self.sweep_progress.setFormat("Sweep progress: idle")
        for widget in self._run_locked_widgets:
            widget.setEnabled(not running)

    def _refresh_scan_meta(self):
        clip_state = "on" if self.record_checkbox.isChecked() else "off"
        self.scan_meta_label.setText(f"Hits: {self._detection_count}  |  Clips: {clip_state}")

    def _selected_backend(self) -> str:
        text = self.device_combo.currentText().lower()
        if text.startswith("hackrf"):
            return "hackrf"
        if text.startswith("soapy"):
            return "soapy"
        return "auto"

    def _selected_profile(self) -> tuple:
        return detect_preferred_profile(self._selected_backend(), self.soapy_args_input.text().strip())

    def refresh_profile_hint(self):
        driver, profile = self._selected_profile()
        rate_text = human_rate(float(profile.get("sample_rate", SAMPLE_RATE)))
        low = human_freq(float(profile.get("min_freq_hz", DEFAULT_START_FREQ)))
        high = human_freq(float(profile.get("max_freq_hz", DEFAULT_END_FREQ)))
        self.profile_hint_label.setText(
            f"Profile: {profile['display']} ({driver})\n"
            f"Safe range: {low} to {high}\n"
            f"Suggested rate: {rate_text}\n"
            f"{profile.get('hint', '')}"
        )

    def apply_device_defaults(self, force: bool = False):
        _, profile = self._selected_profile()
        self.range_input.setText(profile.get("default_range", DEFAULT_RANGE_TEXT))
        step_value, step_unit = recommended_step_for_profile(profile)
        self.step_input.setValue(step_value)
        self.step_unit.setCurrentText(step_unit)
        self.rate_input.setValue(float(profile.get("sample_rate", SAMPLE_RATE)) / 1e6)
        if force:
            self.capture_input.setValue(RECORD_SECS)
            self.gain_input.setValue(GAIN)
        self.refresh_profile_hint()

    def _remember_sidebar_width(self, *_args):
        if self._workspace_mode != "all":
            return
        left_size = self.splitter.sizes()[0]
        if left_size >= MIN_SIDEBAR_WIDTH:
            self._sidebar_width = left_size

    def _refresh_workspace_buttons(self):
        if not hasattr(self, "show_all_btn"):
            return
        self.show_all_btn.setChecked(self._workspace_mode == "all")
        self.show_graphs_btn.setChecked(self._workspace_mode == "graphs")
        self.show_controls_btn.setChecked(self._workspace_mode == "panel" and self.sidebar_tabs.currentIndex() == 0)
        self.show_hex_btn.setChecked(
            self._workspace_mode == "panel"
            and self.sidebar_tabs.currentIndex() == 1
            and getattr(self, "decode_tab_widget", None) is not None
            and self.signal_reader_tabs.currentWidget() is self.decode_tab_widget
        )

    def set_workspace_mode(self, mode: str):
        mode = (mode or "all").strip().lower()
        if mode not in {"all", "panel", "graphs"}:
            mode = "all"

        self._workspace_mode = mode
        self._sidebar_visible = mode != "graphs"
        if mode == "all":
            self.sidebar_scroll.show()
            self.right_panel.show()
            self.sidebar_scroll.setMinimumWidth(MIN_SIDEBAR_WIDTH)
            self.sidebar_scroll.setMaximumWidth(16777215)
            target_width = max(MIN_SIDEBAR_WIDTH, self._sidebar_width)
            self.splitter.setSizes([target_width, max(720, self.width() - target_width)])
        elif mode == "panel":
            self.sidebar_scroll.show()
            self.right_panel.hide()
            self.sidebar_scroll.setMinimumWidth(0)
            self.sidebar_scroll.setMaximumWidth(16777215)
            self.splitter.setSizes([max(1, self.width()), 0])
        else:
            self._remember_sidebar_width()
            self.sidebar_scroll.hide()
            self.right_panel.show()
            self.sidebar_scroll.setMinimumWidth(MIN_SIDEBAR_WIDTH)
            self.sidebar_scroll.setMaximumWidth(16777215)
            self.splitter.setSizes([0, max(1, self.width())])
        self._refresh_workspace_buttons()

    def set_sidebar_visible(self, visible: bool):
        self.set_workspace_mode("all" if visible else "graphs")

    def toggle_sidebar(self):
        self.set_workspace_mode("graphs" if self._workspace_mode != "graphs" else "all")

    def open_recordings_dir(self):
        target = self.record_dir_input.text().strip() or DEFAULT_RECORDINGS_DIR
        Path(target).mkdir(parents=True, exist_ok=True)
        QtCore.QProcess.startDetached("xdg-open", [target])

    def clear_recordings_dir(self):
        target = Path(self.record_dir_input.text().strip() or DEFAULT_RECORDINGS_DIR)
        target.mkdir(parents=True, exist_ok=True)
        for child in list(target.iterdir()):
            if child.is_file() or child.is_symlink():
                child.unlink()
        self.latest_recording_path = ""
        self.latest_cf32_path = ""
        self.refresh_recordings_browser()
        self._set_status("Cleared saved recordings from the current clip directory.")

    def _recordings_dir_path(self) -> Path:
        target = Path(self.record_dir_input.text().strip() or DEFAULT_RECORDINGS_DIR)
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _recording_entry_from_path(self, path: Path) -> dict:
        try:
            with np.load(path, allow_pickle=True) as data:
                timestamp = str(data["timestamp_utc"]) if "timestamp_utc" in data else path.stem
                center = human_freq(float(data["center_freq_hz"])) if "center_freq_hz" in data else "—"
                snr = f"{float(data['snr_db']):.1f} dB" if "snr_db" in data else "—"
                method = str(data["decoder_method"]) if "decoder_method" in data else "—"
                confidence = float(data["decoder_confidence"]) if "decoder_confidence" in data else 0.0
                return {
                    "path": str(path),
                    "timestamp": timestamp.split("T")[-1].replace("+00:00", ""),
                    "center": center,
                    "snr": snr,
                    "method": method,
                    "confidence": f"{confidence:.2f}",
                    "file": path.name,
                }
        except Exception:
            return {
                "path": str(path),
                "timestamp": path.stem.split("_")[0],
                "center": "—",
                "snr": "—",
                "method": "Corrupt file",
                "confidence": "0.00",
                "file": path.name,
            }

    def refresh_recordings_browser(self, selected_path: Optional[str] = None):
        if not hasattr(self, "recordings_table"):
            return
        entries = [
            self._recording_entry_from_path(path)
            for path in sorted(self._recordings_dir_path().glob("*.npz"), key=lambda item: item.stat().st_mtime, reverse=True)
        ]
        self.recording_browser_paths = [entry["path"] for entry in entries]
        self.recordings_table.setRowCount(0)
        for row, entry in enumerate(entries):
            self.recordings_table.insertRow(row)
            values = [
                entry["timestamp"],
                entry["center"],
                entry["snr"],
                entry["method"],
                entry["confidence"],
                entry["file"],
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setForeground(QColor("#f8fafc"))
                item.setBackground(QColor("#0f2232"))
                if col == 0:
                    item.setData(QtCore.Qt.UserRole, entry["path"])
                self.recordings_table.setItem(row, col, item)

        target_path = str(selected_path) if selected_path else None
        if target_path and target_path in self.recording_browser_paths:
            row = self.recording_browser_paths.index(target_path)
            self.recordings_table.selectRow(row)
        elif self.recording_browser_paths:
            self.recordings_table.selectRow(0)
        self._refresh_recordings_nav()

    def _selected_recording_row(self) -> Optional[int]:
        if not hasattr(self, "recordings_table"):
            return None
        selected = self.recordings_table.selectionModel().selectedRows()
        if not selected:
            return None
        return int(selected[0].row())

    def _refresh_recordings_nav(self):
        row = self._selected_recording_row()
        has_rows = bool(getattr(self, "recording_browser_paths", []))
        if hasattr(self, "load_recording_btn"):
            self.load_recording_btn.setEnabled(row is not None)
            self.prev_recording_btn.setEnabled(has_rows and row not in (None, 0))
            self.next_recording_btn.setEnabled(has_rows and row is not None and row < len(self.recording_browser_paths) - 1)

    def browse_recording(self, step: int):
        row = self._selected_recording_row()
        if row is None:
            return
        target_row = max(0, min(len(self.recording_browser_paths) - 1, row + int(step)))
        self.recordings_table.selectRow(target_row)

    def load_selected_recording(self):
        row = self._selected_recording_row()
        if row is None:
            self._set_status("Select a saved recording first.")
            return
        self.load_recording_at_row(row)

    def load_recording_at_row(self, row: int):
        if row < 0 or row >= len(getattr(self, "recording_browser_paths", [])):
            return
        path = Path(self.recording_browser_paths[row])
        try:
            with np.load(path, allow_pickle=True) as data:
                iq = np.asarray(data["iq"]).astype(np.complex64, copy=False)
                center_freq_hz = float(data["center_freq_hz"])
                sample_rate = float(data["sample_rate"])
                saved_timestamp = str(data["timestamp_utc"]) if "timestamp_utc" in data else path.stem
        except Exception as exc:
            self._set_status(f"Failed to load recording {path.name}: {exc}")
            return

        snapshot = analyze_capture(
            iq=iq,
            center_freq_hz=center_freq_hz,
            sample_rate=sample_rate,
            snr_gate_db=self.snr_gate_input.value(),
            decoder_mode=self._selected_decoder_mode(),
            step_index=0,
            total_steps=1,
            sweep_number=0,
        )
        snapshot.timestamp_utc = saved_timestamp
        snapshot.recording_path = str(path)
        self.latest_snapshot = snapshot
        self.latest_recording_path = str(path)
        cf32_path = path.with_suffix(".cf32")
        self.latest_cf32_path = str(cf32_path) if cf32_path.exists() else ""
        self.canvas.load_recording_snapshot(snapshot)
        self._update_metrics(snapshot)
        self.decode_text.setPlainText(self._format_decode_preview(snapshot))
        self._set_status(f"Loaded recording {path.name} with {decoder_mode_label(self._selected_decoder_mode())}.")

    def show_hex_view(self):
        self.set_workspace_mode("panel")
        self.sidebar_tabs.setCurrentIndex(1)
        if hasattr(self, "decode_tab_widget"):
            self.signal_reader_tabs.setCurrentWidget(self.decode_tab_widget)
        self._refresh_workspace_buttons()

    def show_controls_view(self):
        self.set_workspace_mode("panel")
        self.sidebar_tabs.setCurrentIndex(0)
        self._refresh_workspace_buttons()

    def copy_current_hex(self):
        if not self.latest_snapshot or self.latest_snapshot.decode_preview.hex_preview == "—":
            self._set_status("No decoded hex is available for the current capture.")
            return
        QtWidgets.QApplication.clipboard().setText(self.latest_snapshot.decode_preview.hex_preview)
        self._set_status("Copied current hex preview to the clipboard.")

    def open_latest_analyzer(self, tool_name: str):
        if not self.latest_cf32_path:
            self._set_status("No saved IQ clip is available yet.")
            return
        if not have_external_tool(tool_name):
            self._set_status(f"{tool_name} is not installed in PATH.")
            return
        if QtCore.QProcess.startDetached(tool_name, [self.latest_cf32_path]):
            self._set_status(f"Opened latest clip in {tool_name}.")
        else:
            self._set_status(f"Failed to launch {tool_name}.")

    def _selected_decoder_mode(self) -> str:
        return str(self.decoder_mode_combo.currentData() or DEFAULT_DECODER_MODE)

    def _set_decode_banner_text(self, text: str):
        self._decode_banner_full_text = text
        if hasattr(self, "decode_banner"):
            self.decode_banner.setToolTip(text)
            self._refresh_decode_banner()

    def _refresh_decode_banner(self):
        if not hasattr(self, "decode_banner"):
            return
        width = max(120, self.decode_banner.width() - 8)
        elided = self.decode_banner.fontMetrics().elidedText(
            self._decode_banner_full_text,
            QtCore.Qt.ElideRight,
            width,
        )
        self.decode_banner.setText(elided)

    def _step_hz(self) -> float:
        step = self.step_input.value()
        if self.step_unit.currentText().lower() == "khz":
            return step * 1e3
        return step * 1e6

    def start_scan(self):
        if self.thread and self.thread.isRunning():
            return
        if not (have_hackrf() or have_soapy()):
            self._set_status("No SDR backend found. Install HackRF tools or SoapySDR to scan.")
            return

        start_hz, end_hz = parse_range(self.range_input.text())
        try:
            frequency_plan_hz = build_frequency_plan(start_hz, end_hz, self._step_hz())
        except ValueError as exc:
            self._set_status(str(exc))
            return

        _, profile = self._selected_profile()
        try:
            validate_frequency_plan(profile, frequency_plan_hz)
        except ValueError as exc:
            self._set_status(str(exc))
            return

        self.canvas.reset(frequency_plan_hz)
        self.detections_table.setRowCount(0)
        self._detection_count = 0
        self.latest_snapshot = None
        self.latest_recording_path = ""
        self.latest_cf32_path = ""
        self._refresh_scan_meta()
        self._update_metrics(None)

        self.thread = SweepThread(
            frequency_plan_hz=frequency_plan_hz,
            sample_rate_hz=self.rate_input.value() * 1e6,
            capture_secs=self.capture_input.value(),
            gain=self.gain_input.value(),
            snr_gate_db=self.snr_gate_input.value(),
            log_csv=CSV_LOG_FILE,
            backend=self._selected_backend(),
            soapy_args=self.soapy_args_input.text().strip(),
            record_detections=self.record_checkbox.isChecked(),
            record_dir=self.record_dir_input.text().strip() or DEFAULT_RECORDINGS_DIR,
            record_cooldown_s=self.cooldown_input.value(),
            pause_on_detection=self.pause_on_hit_checkbox.isChecked(),
            decoder_mode=self._selected_decoder_mode(),
        )
        self.thread.status.connect(self._set_status)
        self.thread.snapshot.connect(self.handle_snapshot)
        self.thread.detection.connect(self.handle_detection)
        self.thread.pause_state.connect(self.handle_pause_state)
        self.thread.finished.connect(self.handle_thread_finished)
        self.thread.start()

        self._set_running_state(True)
        self._set_status(
            f"Starting sweep across {human_freq(frequency_plan_hz[0])} to {human_freq(frequency_plan_hz[-1])}"
        )

    def stop_scan(self):
        if not self.thread:
            return
        self.thread.stop()
        self.thread.wait(1500)
        self.thread = None
        self._set_running_state(False)
        self._set_status("Sweep stopped")

    def toggle_pause(self):
        if not self.thread:
            return
        self.thread.toggle_pause()

    def handle_pause_state(self, paused: bool):
        self.pause_btn.blockSignals(True)
        self.pause_btn.setChecked(paused)
        self.pause_btn.setText("Resume" if paused else "Pause")
        self.pause_btn.blockSignals(False)
        self.header_pause_btn.setText("Resume" if paused else "Pause")

    def handle_thread_finished(self):
        self.thread = None
        self._set_running_state(False)

    def _update_metrics(self, snapshot: Optional[SweepSnapshot]):
        if snapshot is None:
            for label in self.metric_labels.values():
                label.setText("—")
            if hasattr(self, "decode_text"):
                self.decode_text.clear()
            self._set_decode_banner_text("Decoder: idle  |  Hex: —")
            return

        self.metric_labels["center"].setText(human_freq(snapshot.center_freq_hz))
        self.metric_labels["avg_power"].setText(f"{snapshot.avg_power_db:.1f} dB")
        self.metric_labels["peak"].setText(f"{snapshot.peak_db:.1f} dB")
        self.metric_labels["noise"].setText(f"{snapshot.noise_floor_db:.1f} dB")
        self.metric_labels["snr"].setText(f"{snapshot.snr_db:.1f} dB")
        self.metric_labels["offset"].setText(f"{snapshot.dominant_offset_hz / 1e3:+.1f} kHz")
        self.metric_labels["bandwidth"].setText(human_bandwidth(snapshot.occupied_bw_hz))
        self.metric_labels["sample_rate"].setText(human_rate(snapshot.sample_rate))
        self.metric_labels["decoder"].setText(snapshot.decode_preview.method)
        self.metric_labels["clip"].setText(Path(snapshot.recording_path).name if snapshot.recording_path else "—")
        banner_hex = snapshot.decode_preview.hex_preview if snapshot.decode_preview.hex_preview != "—" else "—"
        self._set_decode_banner_text(f"Decoder: {snapshot.decode_preview.method}  |  Hex: {banner_hex}")

    def _format_decode_preview(self, snapshot: SweepSnapshot) -> str:
        preview = snapshot.decode_preview
        lines = [
            f"Center:        {human_freq(snapshot.center_freq_hz)}",
            f"Mode:          {decoder_mode_label(snapshot.decoder_mode)}",
            f"Decoder:       {preview.method}",
            f"Confidence:    {preview.confidence:.2f}",
            f"Symbol rate:   {preview.symbol_rate_hz:.1f} symbols/s" if preview.symbol_rate_hz > 0 else "Symbol rate:   —",
            f"SNR:           {snapshot.snr_db:.1f} dB",
            "",
            "Bits:",
            preview.bit_preview,
            "",
            "Hex:",
            preview.hex_preview,
            "",
            "ASCII:",
            preview.ascii_preview,
            "",
            "Notes:",
            preview.notes,
        ]
        return "\n".join(lines)

    def handle_snapshot(self, snapshot: SweepSnapshot):
        self.latest_snapshot = snapshot
        if snapshot.recording_path:
            self.latest_recording_path = snapshot.recording_path
            self.latest_cf32_path = str(Path(snapshot.recording_path).with_suffix(".cf32"))
        progress = int(round(((snapshot.step_index + 1) / max(snapshot.total_steps, 1)) * 100))
        self.sweep_progress.setValue(progress)
        self.sweep_progress.setFormat(
            f"Sweep {snapshot.sweep_number + 1}  |  Step {snapshot.step_index + 1}/{snapshot.total_steps}"
        )
        self.canvas.consume_snapshot(snapshot)
        self._update_metrics(snapshot)
        self.decode_text.setPlainText(self._format_decode_preview(snapshot))

    def handle_detection(self, snapshot: SweepSnapshot):
        self._detection_count += 1
        self._refresh_scan_meta()
        self.latest_snapshot = snapshot
        if snapshot.recording_path:
            self.latest_recording_path = snapshot.recording_path
            self.latest_cf32_path = str(Path(snapshot.recording_path).with_suffix(".cf32"))
        self.detections_table.insertRow(0)
        row_values = [
            snapshot.timestamp_utc.split("T")[-1].replace("+00:00", ""),
            human_freq(snapshot.center_freq_hz),
            f"{snapshot.snr_db:.1f} dB",
            human_bandwidth(snapshot.occupied_bw_hz),
            snapshot.decode_preview.method,
            Path(snapshot.recording_path).name if snapshot.recording_path else "live",
        ]
        for col, value in enumerate(row_values):
            item = QtWidgets.QTableWidgetItem(value)
            item.setForeground(QColor("#f8fafc"))
            item.setBackground(QColor("#0f2232"))
            self.detections_table.setItem(0, col, item)

        if snapshot.recording_path:
            self.detections_table.item(0, 5).setToolTip(snapshot.recording_path)

        while self.detections_table.rowCount() > 25:
            self.detections_table.removeRow(self.detections_table.rowCount() - 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "splitter"):
            if self._workspace_mode == "panel":
                self.splitter.setSizes([max(1, self.width()), 0])
            elif self._workspace_mode == "graphs":
                self.splitter.setSizes([0, max(1, self.width())])
        self._refresh_decode_banner()

    def closeEvent(self, event):
        try:
            if self.thread:
                self.thread.stop()
                self.thread.wait(1500)
        finally:
            event.accept()


# ========================= MAIN ============================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QFont("DejaVu Sans", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
