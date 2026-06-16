#!/usr/bin/env python3

"""
Offline frame analysis for saved DroneDetect recordings.

This tool re-decodes the full saved IQ clip instead of relying on the
24-byte live preview. It isolates the dominant signal, segments the clip into
burst windows, and attempts to recover a byte stream from each burst.
"""

from __future__ import annotations

import argparse
import json
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass
class DecodeCandidate:
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
class BurstReport:
    burst_index: int
    start_s: float
    end_s: float
    duration_s: float
    candidate: Optional[DecodeCandidate]


@dataclass
class AnalysisResult:
    recording_label: str
    timestamp_utc: str
    center_freq_hz: float
    sample_rate: float
    clip_seconds: float
    saved_decoder_method: str
    saved_decoder_confidence: float
    decode_rate: float
    requested_mode: str
    min_burst_ms: float
    padding_ms: float
    reports: list[BurstReport]


@dataclass
class BurstFamily:
    family_id: str
    method: str
    count: int
    burst_indexes: list[int]
    min_symbol_rate_hz: float
    max_symbol_rate_hz: float
    min_bytes: int
    max_bytes: int
    avg_confidence: float


def moving_average(series: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(series, dtype=np.float32)
    if values.size == 0 or window <= 1:
        return values
    kernel = np.ones(int(window), dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def lowpass_fir(cutoff_norm: float, taps: int = 129) -> np.ndarray:
    cutoff_norm = float(np.clip(cutoff_norm, 1e-4, 0.49))
    sample_index = np.arange(taps, dtype=np.float32)
    midpoint = (taps - 1) / 2.0
    kernel = 2.0 * cutoff_norm * np.sinc(2.0 * cutoff_norm * (sample_index - midpoint))
    window = 0.54 - 0.46 * np.cos((2.0 * np.pi * sample_index) / float(taps - 1))
    kernel = kernel * window
    kernel_sum = float(np.sum(kernel))
    if abs(kernel_sum) > 1e-12:
        kernel = kernel / kernel_sum
    return kernel.astype(np.float32)


def apply_fir_complex(iq: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    real = np.convolve(iq.real.astype(np.float32, copy=False), kernel, mode="same")
    imag = np.convolve(iq.imag.astype(np.float32, copy=False), kernel, mode="same")
    return (real + 1j * imag).astype(np.complex64)


def isolate_signal_region(
    iq: np.ndarray,
    sample_rate: float,
    dominant_offset_hz: float,
    occupied_bw_hz: float,
) -> tuple[np.ndarray, float]:
    if iq.size < 1024:
        return iq.astype(np.complex64, copy=False), float(sample_rate)

    sample_index = np.arange(iq.size, dtype=np.float32)
    rotator = np.exp(
        -1j * 2.0 * np.pi * float(dominant_offset_hz) * sample_index / float(sample_rate)
    ).astype(np.complex64)
    shifted = iq.astype(np.complex64, copy=False) * rotator

    target_bw_hz = max(float(occupied_bw_hz) * 4.0, 12e3)
    cutoff_hz = min(target_bw_hz, float(sample_rate) * 0.45)
    filtered = apply_fir_complex(shifted, lowpass_fir(cutoff_hz / float(sample_rate), taps=129))

    target_rate_hz = max(96e3, cutoff_hz * 10.0)
    decimation = max(1, int(sample_rate // target_rate_hz))
    if decimation > 1:
        filtered = filtered[::decimation]
        sample_rate = sample_rate / decimation

    return filtered.astype(np.complex64, copy=False), float(sample_rate)


def trim_leading_idle(bits: np.ndarray) -> np.ndarray:
    if bits.size > 8 and np.any(bits != bits[0]):
        first_change = int(np.flatnonzero(bits != bits[0])[0])
        return bits[max(0, first_change - 1) :]
    return bits


def derive_binary_series(series: np.ndarray, smooth_window: Optional[int] = None) -> tuple[Optional[np.ndarray], Optional[int]]:
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


def estimate_symbol_sample_candidates(binary: np.ndarray, hint_samples: Optional[float] = None) -> list[int]:
    candidates = set()
    base = estimate_symbol_samples(binary)
    if base is not None:
        for factor in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
            candidates.add(max(1, int(round(base * factor))))
    if hint_samples is not None:
        hint = max(1, int(round(hint_samples)))
        for delta in (-2, -1, 0, 1, 2):
            candidates.add(max(1, hint + delta))
    return sorted(value for value in candidates if 1 <= value <= max(1, binary.size // 8))


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


def choose_byte_alignment(bits: np.ndarray) -> tuple[int, np.ndarray]:
    best_offset = 0
    best_bytes = np.array([], dtype=np.uint8)
    best_score = float("-inf")
    for offset in range(min(8, bits.size)):
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


def decode_manchester_pairs(chips: np.ndarray, invert_output: bool = False) -> tuple[Optional[np.ndarray], float, int]:
    pair_count = chips.size // 2
    if pair_count < 16:
        return None, 0.0, 0
    pairs = chips[: pair_count * 2].reshape(pair_count, 2)
    valid = pairs[:, 0] != pairs[:, 1]
    valid_pairs = pairs[valid]
    valid_ratio = float(np.mean(valid)) if valid.size else 0.0
    if valid_pairs.shape[0] < 16 or valid_ratio < 0.72:
        return None, valid_ratio, int(pair_count - valid_pairs.shape[0])
    decoded = valid_pairs[:, 0].astype(np.uint8, copy=False)
    if invert_output:
        decoded = 1 - decoded
    return trim_leading_idle(decoded), valid_ratio, int(pair_count - valid_pairs.shape[0])


def decode_nrzi_bits(bits: np.ndarray, invert_output: bool = False) -> Optional[np.ndarray]:
    if bits.size < 17:
        return None
    decoded = (bits[1:] != bits[:-1]).astype(np.uint8, copy=False)
    if invert_output:
        decoded = 1 - decoded
    return trim_leading_idle(decoded)


def build_bitstream_candidate(
    bits: np.ndarray,
    sample_rate: float,
    symbol_samples: int,
    method: str,
    notes: str,
    min_confidence: float = 0.14,
) -> Optional[DecodeCandidate]:
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
    printable_ratio = float(np.mean((byte_values >= 32) & (byte_values < 127)))
    unique_ratio = float(len(np.unique(byte_values[: min(len(byte_values), 24)]))) / float(max(min(len(byte_values), 24), 1))
    idle_ratio = float(np.mean((byte_values == 0x00) | (byte_values == 0xFF)))
    if idle_ratio > 0.94 and unique_ratio < 0.12:
        return None

    length_factor = min(aligned_bits.size / 160.0, 1.0)
    balance_factor = max(0.0, 1.0 - abs(ones_ratio - 0.5) * 2.0)
    transition_factor = min(transition_ratio * 1.8, 1.0)
    structure_factor = max(
        0.0,
        min((1.0 - idle_ratio) * 0.55 + unique_ratio * 0.30 + printable_ratio * 0.15, 1.0),
    )
    confidence = max(
        0.05,
        min(length_factor * 0.28 + balance_factor * 0.24 + transition_factor * 0.18 + structure_factor * 0.30, 0.99),
    )
    if confidence < min_confidence:
        return None

    return DecodeCandidate(
        method=method,
        bits=aligned_bits,
        byte_values=byte_values,
        symbol_rate_hz=float(sample_rate / symbol_samples) if symbol_samples else 0.0,
        symbol_samples=int(symbol_samples),
        byte_offset=int(byte_offset),
        confidence=float(confidence),
        printable_ratio=printable_ratio,
        unique_ratio=unique_ratio,
        idle_ratio=idle_ratio,
        notes=f"{notes} Byte alignment guessed with bit offset {byte_offset}.",
    )


def build_uart_candidate(
    binary: np.ndarray,
    sample_rate: float,
    symbol_samples: int,
    smooth_window: int,
) -> Optional[DecodeCandidate]:
    candidates = []
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
            bits_display = np.unpackbits(byte_values)
            printable_ratio = float(np.mean((byte_values >= 32) & (byte_values < 127)))
            unique_ratio = float(len(np.unique(byte_values[: min(len(byte_values), 24)]))) / float(max(min(len(byte_values), 24), 1))
            idle_ratio = float(np.mean((byte_values == 0x00) | (byte_values == 0xFF)))
            transition_ratio = float(np.mean(bits_display[1:] != bits_display[:-1])) if bits_display.size > 1 else 0.0
            balance_factor = max(0.0, 1.0 - abs(float(np.mean(bits_display)) - 0.5) * 2.0)
            structure_factor = max(0.0, min((1.0 - idle_ratio) * 0.50 + unique_ratio * 0.30 + printable_ratio * 0.20, 1.0))
            frame_density = min(len(frames) / max(stream.size / 10.0, 1.0), 1.0)
            confidence_base = 0.28 + min(len(frames) / 8.0, 1.0) * 0.30 + frame_density * 0.22
            confidence = max(
                0.05,
                min(confidence_base * 0.60 + structure_factor * 0.25 + balance_factor * 0.10 + min(transition_ratio * 0.2, 0.05), 0.99),
            )
            if confidence < 0.16:
                continue
            notes = (
                f"UART-like frame finder with smooth={smooth_window}, offset={offset}, frames={len(frames)}."
                + (" Inverted polarity applied." if inverted else "")
            )
            candidates.append(
                DecodeCandidate(
                    method="UART-like frame finder",
                    bits=bits_display,
                    byte_values=byte_values,
                    symbol_rate_hz=float(sample_rate / symbol_samples) if symbol_samples else 0.0,
                    symbol_samples=int(symbol_samples),
                    byte_offset=0,
                    confidence=float(confidence),
                    printable_ratio=printable_ratio,
                    unique_ratio=unique_ratio,
                    idle_ratio=idle_ratio,
                    notes=notes,
                )
            )
    return max(candidates, key=lambda item: item.confidence) if candidates else None


def build_manchester_candidates(
    binary: np.ndarray,
    sample_rate: float,
    hint_samples: Optional[float],
    smooth_window: int,
    source_method: str,
) -> list[DecodeCandidate]:
    candidates = []
    for chip_samples in estimate_symbol_sample_candidates(binary, hint_samples):
        max_offsets = min(chip_samples, 10)
        for offset in range(max_offsets):
            chips = extract_symbol_bits(binary[offset:], chip_samples, start_index=0)
            if chips is None or chips.size < 32:
                continue
            for inverted in (False, True):
                stream = (1 - chips) if inverted else chips
                for phase in (0, 1):
                    phase_stream = stream[phase:]
                    for invert_output in (False, True):
                        decoded, valid_ratio, invalid_pairs = decode_manchester_pairs(
                            phase_stream,
                            invert_output=invert_output,
                        )
                        if decoded is None or decoded.size < 16:
                            continue
                        notes = (
                            f"smooth={smooth_window}, offset={offset}, phase={phase}, chips/sym={chip_samples}, "
                            f"valid_pairs={valid_ratio:.2f}, invalid_pairs={invalid_pairs}"
                            + (" Inverted polarity applied." if inverted else "")
                            + (" Decoded Manchester polarity flipped." if invert_output else "")
                        )
                        candidate = build_bitstream_candidate(
                            bits=decoded,
                            sample_rate=sample_rate,
                            symbol_samples=max(1, chip_samples * 2),
                            method=f"{source_method} + Manchester",
                            notes=notes,
                        )
                        if candidate is not None:
                            candidates.append(candidate)
    return candidates


def build_nrzi_candidates(
    binary: np.ndarray,
    sample_rate: float,
    hint_samples: Optional[float],
    smooth_window: int,
    source_method: str,
) -> list[DecodeCandidate]:
    candidates = []
    for symbol_samples in estimate_symbol_sample_candidates(binary, hint_samples):
        max_offsets = min(symbol_samples, 10)
        for offset in range(max_offsets):
            bits = extract_symbol_bits(binary[offset:], symbol_samples, start_index=0)
            if bits is None or bits.size < 30:
                continue
            for inverted in (False, True):
                stream = (1 - bits) if inverted else bits
                for invert_output in (False, True):
                    decoded = decode_nrzi_bits(stream, invert_output=invert_output)
                    if decoded is None or decoded.size < 16:
                        continue
                    notes = (
                        f"smooth={smooth_window}, offset={offset}, differential decode"
                        + (" Inverted polarity applied." if inverted else "")
                        + (" Output polarity flipped." if invert_output else "")
                    )
                    candidate = build_bitstream_candidate(
                        bits=decoded,
                        sample_rate=sample_rate,
                        symbol_samples=symbol_samples,
                        method=f"{source_method} + Differential / NRZI",
                        notes=notes,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
    return candidates


def decode_fsk(iq: np.ndarray, sample_rate: float, symbol_rate_hint_hz: Optional[float]) -> Optional[DecodeCandidate]:
    discriminator = np.angle(iq[1:] * np.conj(iq[:-1])) if iq.size > 1 else np.array([], dtype=np.float32)
    binary, smooth_window = derive_binary_series(discriminator.astype(np.float32, copy=False))
    if binary is None or smooth_window is None:
        return None
    hint_samples = (sample_rate / symbol_rate_hint_hz) if symbol_rate_hint_hz else None
    candidates = []
    for symbol_samples in estimate_symbol_sample_candidates(binary, hint_samples):
        max_offsets = min(symbol_samples, 10)
        for offset in range(max_offsets):
            bits = extract_symbol_bits(binary[offset:], symbol_samples, start_index=0)
            if bits is None or bits.size < 30:
                continue
            for inverted in (False, True):
                stream = (1 - bits) if inverted else bits
                candidate = build_bitstream_candidate(
                    bits=stream,
                    sample_rate=sample_rate,
                    symbol_samples=symbol_samples,
                    method="2-FSK discriminator",
                    notes=f"smooth={smooth_window}, offset={offset}" + (" Inverted polarity applied." if inverted else ""),
                )
                if candidate is not None:
                    candidates.append(candidate)
    return max(candidates, key=lambda item: item.confidence) if candidates else None


def decode_ask(iq: np.ndarray, sample_rate: float, symbol_rate_hint_hz: Optional[float]) -> Optional[DecodeCandidate]:
    envelope = np.abs(iq).astype(np.float32, copy=False)
    binary, smooth_window = derive_binary_series(envelope)
    if binary is None or smooth_window is None:
        return None
    hint_samples = (sample_rate / symbol_rate_hint_hz) if symbol_rate_hint_hz else None
    candidates = []
    for symbol_samples in estimate_symbol_sample_candidates(binary, hint_samples):
        max_offsets = min(symbol_samples, 10)
        for offset in range(max_offsets):
            bits = extract_symbol_bits(binary[offset:], symbol_samples, start_index=0)
            if bits is None or bits.size < 30:
                continue
            for inverted in (False, True):
                stream = (1 - bits) if inverted else bits
                candidate = build_bitstream_candidate(
                    bits=stream,
                    sample_rate=sample_rate,
                    symbol_samples=symbol_samples,
                    method="ASK / OOK envelope",
                    notes=f"smooth={smooth_window}, offset={offset}" + (" Inverted polarity applied." if inverted else ""),
                )
                if candidate is not None:
                    candidates.append(candidate)
    return max(candidates, key=lambda item: item.confidence) if candidates else None


def decode_uart(iq: np.ndarray, sample_rate: float, symbol_rate_hint_hz: Optional[float]) -> Optional[DecodeCandidate]:
    discriminator = np.angle(iq[1:] * np.conj(iq[:-1])) if iq.size > 1 else np.array([], dtype=np.float32)
    binary, smooth_window = derive_binary_series(discriminator.astype(np.float32, copy=False))
    if binary is None or smooth_window is None:
        return None
    hint_samples = (sample_rate / symbol_rate_hint_hz) if symbol_rate_hint_hz else None
    candidates = []
    for symbol_samples in estimate_symbol_sample_candidates(binary, hint_samples):
        candidate = build_uart_candidate(binary, sample_rate, symbol_samples, smooth_window)
        if candidate is not None:
            candidates.append(candidate)
    return max(candidates, key=lambda item: item.confidence) if candidates else None


def decode_manchester(iq: np.ndarray, sample_rate: float, symbol_rate_hint_hz: Optional[float]) -> Optional[DecodeCandidate]:
    hint_samples = (sample_rate / symbol_rate_hint_hz) if symbol_rate_hint_hz else None
    candidates = []
    series_options = [
        ("2-FSK discriminator", np.angle(iq[1:] * np.conj(iq[:-1])) if iq.size > 1 else np.array([], dtype=np.float32)),
        ("ASK / OOK envelope", np.abs(iq).astype(np.float32, copy=False)),
    ]
    for source_method, series in series_options:
        binary, smooth_window = derive_binary_series(np.asarray(series, dtype=np.float32))
        if binary is None or smooth_window is None:
            continue
        candidates.extend(
            build_manchester_candidates(
                binary=binary,
                sample_rate=sample_rate,
                hint_samples=hint_samples,
                smooth_window=smooth_window,
                source_method=source_method,
            )
        )
    return max(candidates, key=lambda item: item.confidence) if candidates else None


def decode_nrzi(iq: np.ndarray, sample_rate: float, symbol_rate_hint_hz: Optional[float]) -> Optional[DecodeCandidate]:
    hint_samples = (sample_rate / symbol_rate_hint_hz) if symbol_rate_hint_hz else None
    candidates = []
    series_options = [
        ("2-FSK discriminator", np.angle(iq[1:] * np.conj(iq[:-1])) if iq.size > 1 else np.array([], dtype=np.float32)),
        ("ASK / OOK envelope", np.abs(iq).astype(np.float32, copy=False)),
    ]
    for source_method, series in series_options:
        binary, smooth_window = derive_binary_series(np.asarray(series, dtype=np.float32))
        if binary is None or smooth_window is None:
            continue
        candidates.extend(
            build_nrzi_candidates(
                binary=binary,
                sample_rate=sample_rate,
                hint_samples=hint_samples,
                smooth_window=smooth_window,
                source_method=source_method,
            )
        )
    return max(candidates, key=lambda item: item.confidence) if candidates else None


def detect_bursts(
    iq: np.ndarray,
    sample_rate: float,
    min_burst_ms: float,
    padding_ms: float,
) -> list[tuple[int, int]]:
    envelope = np.abs(iq).astype(np.float32, copy=False)
    smooth_window = max(8, min(2048, envelope.size // 2000))
    smoothed = moving_average(envelope, smooth_window)
    low = float(np.percentile(smoothed, 70))
    high = float(np.percentile(smoothed, 99))
    threshold = low + (high - low) * 0.22
    active = smoothed > threshold

    pad_samples = max(4, int(round((padding_ms / 1e3) * sample_rate)))
    min_burst_samples = max(8, int(round((min_burst_ms / 1e3) * sample_rate)))

    transitions = np.diff(active.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)

    runs: list[tuple[int, int]] = []
    for start, end in zip(starts, ends):
        start = max(0, int(start - pad_samples))
        end = min(envelope.size, int(end + pad_samples))
        if (end - start) < min_burst_samples:
            continue
        if runs and start <= runs[-1][1] + pad_samples:
            runs[-1] = (runs[-1][0], max(runs[-1][1], end))
        else:
            runs.append((start, end))

    if not runs:
        runs.append((0, envelope.size))
    return runs


def infer_mode_order(requested_mode: str, saved_method: str) -> list[str]:
    requested_mode = (requested_mode or "auto").strip().lower()
    saved_method = (saved_method or "").strip().lower()
    if requested_mode != "auto":
        return [requested_mode]
    if "fsk" in saved_method:
        return ["fsk", "manchester", "nrzi", "uart", "ask"]
    if "ask" in saved_method or "ook" in saved_method:
        return ["ask", "manchester", "nrzi", "uart", "fsk"]
    if "uart" in saved_method:
        return ["uart", "fsk", "manchester", "nrzi", "ask"]
    return ["fsk", "ask", "manchester", "nrzi", "uart"]


def decode_burst(
    iq: np.ndarray,
    sample_rate: float,
    mode_order: Iterable[str],
    symbol_rate_hint_hz: Optional[float],
) -> Optional[DecodeCandidate]:
    candidates = []
    for mode in mode_order:
        if mode == "fsk":
            candidate = decode_fsk(iq, sample_rate, symbol_rate_hint_hz)
        elif mode == "ask":
            candidate = decode_ask(iq, sample_rate, symbol_rate_hint_hz)
        elif mode == "uart":
            candidate = decode_uart(iq, sample_rate, symbol_rate_hint_hz)
        elif mode == "manchester":
            candidate = decode_manchester(iq, sample_rate, symbol_rate_hint_hz)
        elif mode == "nrzi":
            candidate = decode_nrzi(iq, sample_rate, symbol_rate_hint_hz)
        else:
            candidate = None
        if candidate is not None:
            candidates.append(candidate)
    return max(candidates, key=lambda item: item.confidence) if candidates else None


def format_hex(byte_values: np.ndarray, max_bytes: int) -> str:
    if byte_values.size == 0:
        return "—"
    preview = byte_values if max_bytes <= 0 else byte_values[:max_bytes]
    text = " ".join(f"{int(byte_val):02X}" for byte_val in preview)
    if max_bytes > 0 and byte_values.size > max_bytes:
        text += f" ... (+{byte_values.size - max_bytes} bytes)"
    return text


def format_hex_lines(byte_values: np.ndarray, max_bytes: int, line_bytes: int = 24) -> list[str]:
    if byte_values.size == 0:
        return ["—"]
    preview = byte_values if max_bytes <= 0 else byte_values[:max_bytes]
    lines = [
        " ".join(f"{int(byte_val):02X}" for byte_val in preview[start : start + line_bytes])
        for start in range(0, preview.size, line_bytes)
    ]
    if max_bytes > 0 and byte_values.size > max_bytes:
        lines.append(f"... (+{byte_values.size - max_bytes} bytes)")
    return lines


def format_ascii_lines(byte_values: np.ndarray, max_bytes: int, line_bytes: int = 48) -> list[str]:
    if byte_values.size == 0:
        return ["—"]
    preview = byte_values if max_bytes <= 0 else byte_values[:max_bytes]
    chars = "".join(chr(int(byte_val)) if 32 <= int(byte_val) < 127 else "." for byte_val in preview)
    lines = [chars[start : start + line_bytes] for start in range(0, len(chars), line_bytes)]
    if max_bytes > 0 and byte_values.size > max_bytes:
        lines.append(f"... (+{byte_values.size - max_bytes} bytes)")
    return lines


def format_hex_dump_lines(byte_values: np.ndarray, max_bytes: int, line_bytes: int = 16) -> list[str]:
    if byte_values.size == 0:
        return ["—"]
    preview = byte_values if max_bytes <= 0 else byte_values[:max_bytes]
    lines = []
    for offset in range(0, int(preview.size), line_bytes):
        chunk = preview[offset : offset + line_bytes]
        hex_text = " ".join(f"{int(byte_val):02X}" for byte_val in chunk)
        ascii_text = "".join(chr(int(byte_val)) if 32 <= int(byte_val) < 127 else "." for byte_val in chunk)
        lines.append(f"{offset:04X}  {hex_text:<47}  {ascii_text}")
    if max_bytes > 0 and byte_values.size > max_bytes:
        lines.append(f"... (+{byte_values.size - max_bytes} bytes)")
    return lines


def analysis_mode_label(mode: str) -> str:
    mode = (mode or "auto").strip().lower()
    return {
        "auto": "Auto",
        "fsk": "2-FSK",
        "ask": "ASK / OOK",
        "uart": "UART-like",
        "manchester": "Manchester",
        "nrzi": "Differential / NRZI",
    }.get(mode, mode.upper())


def format_hex_preview(byte_values: np.ndarray, max_bytes: int = 8) -> str:
    if byte_values.size == 0:
        return "—"
    preview = byte_values[:max_bytes]
    text = " ".join(f"{int(byte_val):02X}" for byte_val in preview)
    if byte_values.size > max_bytes:
        text += " ..."
    return text


def format_ascii_preview(byte_values: np.ndarray, max_bytes: int = 16) -> str:
    if byte_values.size == 0:
        return "—"
    preview = byte_values[:max_bytes]
    text = "".join(chr(int(value)) if 32 <= int(value) < 127 else "." for value in preview)
    if byte_values.size > max_bytes:
        text += "..."
    return text


def byte_entropy(byte_values: np.ndarray) -> float:
    if byte_values.size == 0:
        return 0.0
    counts = np.bincount(byte_values.astype(np.uint8, copy=False), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0.0] / float(byte_values.size)
    return float(-(probabilities * np.log2(probabilities)).sum())


def top_byte_values(byte_values: np.ndarray, limit: int = 4) -> list[tuple[int, int]]:
    if byte_values.size == 0:
        return []
    counts = np.bincount(byte_values.astype(np.uint8, copy=False), minlength=256)
    indexes = np.flatnonzero(counts)
    if indexes.size == 0:
        return []
    order = sorted(indexes, key=lambda idx: int(counts[idx]), reverse=True)
    return [(int(idx), int(counts[idx])) for idx in order[:limit]]


def common_prefix_len(left: np.ndarray, right: np.ndarray) -> int:
    limit = min(int(left.size), int(right.size))
    idx = 0
    while idx < limit and int(left[idx]) == int(right[idx]):
        idx += 1
    return idx


def common_suffix_len(left: np.ndarray, right: np.ndarray) -> int:
    limit = min(int(left.size), int(right.size))
    idx = 0
    while idx < limit and int(left[-(idx + 1)]) == int(right[-(idx + 1)]):
        idx += 1
    return idx


def stream_metrics(byte_values: np.ndarray) -> dict:
    if byte_values.size == 0:
        return {
            "entropy": 0.0,
            "printable_ratio": 0.0,
            "unique_ratio": 0.0,
            "idle_ratio": 0.0,
        }
    window = byte_values[: min(int(byte_values.size), 24)]
    return {
        "entropy": byte_entropy(byte_values),
        "printable_ratio": float(np.mean((byte_values >= 32) & (byte_values < 127))),
        "unique_ratio": float(len(np.unique(window))) / float(max(window.size, 1)),
        "idle_ratio": float(np.mean((byte_values == 0x00) | (byte_values == 0xFF))),
    }


def reverse_bits_per_byte(byte_values: np.ndarray) -> np.ndarray:
    values = byte_values.astype(np.uint8, copy=True)
    values = ((values & 0xF0) >> 4) | ((values & 0x0F) << 4)
    values = ((values & 0xCC) >> 2) | ((values & 0x33) << 2)
    values = ((values & 0xAA) >> 1) | ((values & 0x55) << 1)
    return values.astype(np.uint8, copy=False)


def pn9_mask_bytes(length: int, seed: int = 0x1FF, msb_first: bool = False) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=np.uint8)
    state = int(seed) & 0x1FF
    mask = np.zeros(int(length), dtype=np.uint8)
    for index in range(int(length)):
        byte_val = 0
        for bit_index in range(8):
            if msb_first:
                out_bit = (state >> 8) & 0x1
                feedback = ((state >> 8) ^ (state >> 4)) & 0x1
                state = ((state << 1) & 0x1FF) | feedback
                shift = 7 - bit_index
            else:
                out_bit = state & 0x1
                feedback = ((state >> 0) ^ (state >> 5)) & 0x1
                state = ((state >> 1) & 0x0FF) | (feedback << 8)
                shift = bit_index
            byte_val |= int(out_bit) << shift
        mask[index] = byte_val
    return mask


def crc16_generic(
    data: bytes,
    poly: int,
    init: int,
    xorout: int = 0,
    refin: bool = False,
    refout: bool = False,
) -> int:
    crc = int(init) & 0xFFFF
    for byte in data:
        value = int(byte) & 0xFF
        if refin:
            crc ^= value
            for _ in range(8):
                if crc & 0x0001:
                    crc = ((crc >> 1) ^ poly) & 0xFFFF
                else:
                    crc = (crc >> 1) & 0xFFFF
        else:
            crc ^= (value << 8) & 0xFFFF
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ poly) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
    if refout and not refin:
        reflected = 0
        for bit_index in range(16):
            reflected = (reflected << 1) | ((crc >> bit_index) & 0x1)
        crc = reflected & 0xFFFF
    return (crc ^ int(xorout)) & 0xFFFF


def detect_crc_matches(byte_values: np.ndarray) -> list[dict]:
    matches = []
    payload = byte_values.astype(np.uint8, copy=False).tobytes()
    if len(payload) >= 2:
        body = payload[:-2]
        trailer = payload[-2:]
        trailer_be = int.from_bytes(trailer, "big")
        trailer_le = int.from_bytes(trailer, "little")
        crc16_algorithms = [
            ("CRC-16/ARC", lambda data: crc16_generic(data, poly=0xA001, init=0x0000, refin=True)),
            ("CRC-16/XMODEM", lambda data: crc16_generic(data, poly=0x1021, init=0x0000)),
            ("CRC-16/CCITT-FALSE", lambda data: crc16_generic(data, poly=0x1021, init=0xFFFF)),
            ("CRC-16/X25", lambda data: crc16_generic(data, poly=0x8408, init=0xFFFF, xorout=0xFFFF, refin=True)),
            ("CRC-16/KERMIT", lambda data: crc16_generic(data, poly=0x8408, init=0x0000, refin=True)),
        ]
        for name, func in crc16_algorithms:
            value = int(func(body)) & 0xFFFF
            if value == trailer_be:
                matches.append({"name": name, "length": 2, "endian": "big"})
            if value == trailer_le and trailer_le != trailer_be:
                matches.append({"name": name, "length": 2, "endian": "little"})
    if len(payload) >= 4:
        body = payload[:-4]
        trailer = payload[-4:]
        trailer_be = int.from_bytes(trailer, "big")
        trailer_le = int.from_bytes(trailer, "little")
        crc32_value = zlib.crc32(body) & 0xFFFFFFFF
        if crc32_value == trailer_be:
            matches.append({"name": "CRC-32", "length": 4, "endian": "big"})
        if crc32_value == trailer_le and trailer_le != trailer_be:
            matches.append({"name": "CRC-32", "length": 4, "endian": "little"})
    unique_matches = []
    seen = set()
    for match in matches:
        key = (match["name"], match["length"], match["endian"])
        if key in seen:
            continue
        seen.add(key)
        unique_matches.append(match)
    return unique_matches


def build_postprocess_hints(byte_values: np.ndarray, limit: int = 3) -> list[dict]:
    raw_metrics = stream_metrics(byte_values)
    raw_entropy = float(raw_metrics["entropy"])
    raw_printable = float(raw_metrics["printable_ratio"])
    raw_idle = float(raw_metrics["idle_ratio"])

    transforms = [
        ("byte_invert", "Byte invert", np.bitwise_xor(byte_values.astype(np.uint8, copy=False), 0xFF)),
        ("bit_reverse", "Bit-reverse bytes", reverse_bits_per_byte(byte_values)),
        ("pn9_lsb", "PN9 descramble (LSB-first)", np.bitwise_xor(byte_values.astype(np.uint8, copy=False), pn9_mask_bytes(int(byte_values.size), msb_first=False))),
        ("pn9_msb", "PN9 descramble (MSB-first)", np.bitwise_xor(byte_values.astype(np.uint8, copy=False), pn9_mask_bytes(int(byte_values.size), msb_first=True))),
    ]

    hints = []
    seen = {byte_values.astype(np.uint8, copy=False).tobytes()}
    for key, label, transformed in transforms:
        transformed = transformed.astype(np.uint8, copy=False)
        blob = transformed.tobytes()
        if blob in seen:
            continue
        seen.add(blob)
        metrics = stream_metrics(transformed)
        crc_matches = detect_crc_matches(transformed)
        entropy_delta = raw_entropy - float(metrics["entropy"])
        printable_delta = float(metrics["printable_ratio"]) - raw_printable
        idle_delta = raw_idle - float(metrics["idle_ratio"])
        score = (
            max(entropy_delta, 0.0) * 0.30
            + max(printable_delta, 0.0) * 0.55
            + max(idle_delta, 0.0) * 0.10
            + min(len(crc_matches), 2) * 0.35
        )
        notes = []
        if printable_delta >= 0.12:
            notes.append(f"printable +{printable_delta:.2f}")
        if entropy_delta >= 0.35:
            notes.append(f"entropy {raw_entropy:.2f}->{metrics['entropy']:.2f}")
        if idle_delta >= 0.10:
            notes.append(f"idle {raw_idle:.2f}->{metrics['idle_ratio']:.2f}")
        if crc_matches:
            crc_summary = ", ".join(f"{item['name']} ({item['endian']})" for item in crc_matches[:3])
            notes.append(f"CRC match: {crc_summary}")
        if not notes:
            continue
        hints.append(
            {
                "key": key,
                "name": label,
                "score": round(float(score), 3),
                "entropy": round(float(metrics["entropy"]), 3),
                "printable_ratio": round(float(metrics["printable_ratio"]), 3),
                "unique_ratio": round(float(metrics["unique_ratio"]), 3),
                "idle_ratio": round(float(metrics["idle_ratio"]), 3),
                "hex_preview": format_hex_preview(transformed, max_bytes=12),
                "ascii_preview": format_ascii_preview(transformed, max_bytes=24),
                "crc_matches": [f"{item['name']} ({item['endian']})" for item in crc_matches],
                "notes": "; ".join(notes),
            }
        )
    hints.sort(key=lambda item: (len(item["crc_matches"]), item["score"]), reverse=True)
    return hints[:limit]


def find_family_peers(result: AnalysisResult, report: BurstReport) -> list[BurstReport]:
    candidate = report.candidate
    if candidate is None:
        return []
    peers = []
    for other in result.reports:
        if other is report or other.candidate is None:
            continue
        other_candidate = other.candidate
        rate_floor = max(min(float(candidate.symbol_rate_hz), float(other_candidate.symbol_rate_hz)), 1.0)
        rate_ratio = max(float(candidate.symbol_rate_hz), float(other_candidate.symbol_rate_hz)) / rate_floor
        byte_delta = abs(int(candidate.byte_values.size) - int(other_candidate.byte_values.size))
        if str(other_candidate.method) != str(candidate.method):
            continue
        if rate_ratio > 1.25:
            continue
        if byte_delta > max(8, int(round(candidate.byte_values.size * 0.18))):
            continue
        peers.append(other)
    return peers


def cluster_burst_families(result: AnalysisResult) -> list[BurstFamily]:
    families: list[dict] = []
    for report in result.reports:
        if report.candidate is None:
            continue
        candidate = report.candidate
        best_index = None
        best_score = None
        for family_index, family in enumerate(families):
            if family["method"] != str(candidate.method):
                continue
            avg_rate = family["rate_sum"] / max(family["count"], 1)
            avg_bytes = family["byte_sum"] / max(family["count"], 1)
            rate_floor = max(min(avg_rate, float(candidate.symbol_rate_hz)), 1.0)
            rate_ratio = max(avg_rate, float(candidate.symbol_rate_hz)) / rate_floor
            byte_delta = abs(avg_bytes - float(candidate.byte_values.size))
            if rate_ratio > 1.18:
                continue
            if byte_delta > max(8.0, avg_bytes * 0.16):
                continue
            score = abs(np.log(rate_ratio)) + (byte_delta / max(avg_bytes, 1.0))
            if best_score is None or score < best_score:
                best_score = score
                best_index = family_index

        if best_index is None:
            families.append(
                {
                    "method": str(candidate.method),
                    "count": 1,
                    "burst_indexes": [report.burst_index],
                    "rate_sum": float(candidate.symbol_rate_hz),
                    "byte_sum": float(candidate.byte_values.size),
                    "min_rate": float(candidate.symbol_rate_hz),
                    "max_rate": float(candidate.symbol_rate_hz),
                    "min_bytes": int(candidate.byte_values.size),
                    "max_bytes": int(candidate.byte_values.size),
                    "confidence_sum": float(candidate.confidence),
                }
            )
            continue

        family = families[best_index]
        family["count"] += 1
        family["burst_indexes"].append(report.burst_index)
        family["rate_sum"] += float(candidate.symbol_rate_hz)
        family["byte_sum"] += float(candidate.byte_values.size)
        family["min_rate"] = min(family["min_rate"], float(candidate.symbol_rate_hz))
        family["max_rate"] = max(family["max_rate"], float(candidate.symbol_rate_hz))
        family["min_bytes"] = min(family["min_bytes"], int(candidate.byte_values.size))
        family["max_bytes"] = max(family["max_bytes"], int(candidate.byte_values.size))
        family["confidence_sum"] += float(candidate.confidence)

    built = [
        BurstFamily(
            family_id=f"F{index + 1}",
            method=str(family["method"]),
            count=int(family["count"]),
            burst_indexes=list(family["burst_indexes"]),
            min_symbol_rate_hz=float(family["min_rate"]),
            max_symbol_rate_hz=float(family["max_rate"]),
            min_bytes=int(family["min_bytes"]),
            max_bytes=int(family["max_bytes"]),
            avg_confidence=float(family["confidence_sum"] / max(family["count"], 1)),
        )
        for index, family in enumerate(families)
    ]
    return sorted(built, key=lambda family: family.burst_indexes[0] if family.burst_indexes else 10**9)


def family_reports(result: AnalysisResult, family: BurstFamily) -> list[BurstReport]:
    burst_index_set = set(int(index) for index in family.burst_indexes)
    return [report for report in result.reports if report.burst_index in burst_index_set and report.candidate is not None]


def shared_prefix_len_for_reports(reports: list[BurstReport]) -> int:
    if not reports:
        return 0
    prefix = reports[0].candidate.byte_values
    length = int(prefix.size)
    for report in reports[1:]:
        length = min(length, common_prefix_len(prefix[:length], report.candidate.byte_values))
        if length <= 0:
            return 0
    return length


def shared_suffix_len_for_reports(reports: list[BurstReport]) -> int:
    if not reports:
        return 0
    suffix = reports[0].candidate.byte_values
    length = int(suffix.size)
    for report in reports[1:]:
        length = min(length, common_suffix_len(suffix[-length:], report.candidate.byte_values))
        if length <= 0:
            return 0
    return length


def describe_burst_family(result: AnalysisResult, family: BurstFamily) -> str:
    reports = family_reports(result, family)
    if not reports:
        return "no decoded bursts"

    notes = []
    shared_prefix = shared_prefix_len_for_reports(reports)
    shared_suffix = shared_suffix_len_for_reports(reports)
    unique_lengths = {int(report.candidate.byte_values.size) for report in reports}
    if shared_prefix >= 2:
        notes.append(f"shared {shared_prefix}-byte prefix")
    if len(unique_lengths) == 1:
        notes.append("same-length frames")
    else:
        notes.append("variable-length frames")
    if shared_suffix >= 2:
        notes.append(f"shared {shared_suffix}-byte trailer")
    if family.avg_confidence >= 0.95:
        notes.append("strong slice quality")
    return ", ".join(notes)


def detect_length_fields(byte_values: np.ndarray) -> list[dict]:
    size = int(byte_values.size)
    hints = []
    for offset in range(min(size, 8)):
        for width in (1, 2):
            if offset + width > size:
                continue
            value = int.from_bytes(byte_values[offset : offset + width].tobytes(), "big")
            remaining = size - (offset + width)
            if value == size:
                hints.append(
                    {
                        "offset": offset,
                        "length": width,
                        "value": value,
                        "note": "matches total byte count",
                    }
                )
            elif value == remaining:
                hints.append(
                    {
                        "offset": offset,
                        "length": width,
                        "value": value,
                        "note": "matches bytes remaining after this field",
                    }
                )
            elif value in {max(size - 2, 0), max(size - 4, 0)}:
                hints.append(
                    {
                        "offset": offset,
                        "length": width,
                        "value": value,
                        "note": "close to total bytes minus a small trailer",
                    }
                )
    unique_hints = []
    seen = set()
    for hint in hints:
        key = (hint["offset"], hint["length"], hint["value"], hint["note"])
        if key in seen:
            continue
        seen.add(key)
        unique_hints.append(hint)
    return unique_hints[:4]


def build_decode_tree_entries(result: AnalysisResult, report: BurstReport) -> list[dict]:
    candidate = report.candidate
    if candidate is None:
        return [
            {
                "label": "No decode candidate",
                "offset": "",
                "length": "",
                "value": "",
                "notes": "Burst was segmented, but no decoder family produced a stable byte stream.",
                "children": [],
            }
        ]

    byte_values = candidate.byte_values
    size = int(byte_values.size)
    peers = find_family_peers(result, report)
    peer_prefix = []
    peer_suffix = []
    for peer in peers:
        peer_bytes = peer.candidate.byte_values
        peer_prefix.append((common_prefix_len(byte_values, peer_bytes), peer))
        peer_suffix.append((common_suffix_len(byte_values, peer_bytes), peer))
    best_prefix_len, best_prefix_peer = max(peer_prefix, default=(0, None), key=lambda item: item[0])
    best_suffix_len, best_suffix_peer = max(peer_suffix, default=(0, None), key=lambda item: item[0])
    same_length_peers = [peer for peer in peers if int(peer.candidate.byte_values.size) == size]
    length_hints = detect_length_fields(byte_values)
    crc_matches = detect_crc_matches(byte_values)
    postprocess_hints = build_postprocess_hints(byte_values)

    entries = [
        {
            "label": "Candidate Decode",
            "offset": "",
            "length": "",
            "value": candidate.method,
            "notes": (
                f"conf={candidate.confidence:.3f}, symbol_rate={candidate.symbol_rate_hz:.1f} Hz, "
                f"samples/sym={candidate.symbol_samples}, byte_offset={candidate.byte_offset}"
            ),
            "children": [
                {
                    "label": "Recovered Byte Stream",
                    "offset": "0x0000",
                    "length": str(size),
                    "value": format_hex_preview(byte_values, max_bytes=12),
                    "notes": "heuristic byte alignment applied after symbol slicing",
                    "children": [],
                },
                {
                    "label": "Byte Statistics",
                    "offset": "",
                    "length": "",
                    "value": f"entropy={byte_entropy(byte_values):.2f}",
                    "notes": (
                        f"printable={candidate.printable_ratio:.2f}, "
                        f"idle={candidate.idle_ratio:.2f}, unique24={candidate.unique_ratio:.2f}"
                    ),
                    "children": [],
                },
            ],
        }
    ]

    header_len = best_prefix_len if best_prefix_len >= 2 else 0
    if header_len:
        entries.append(
            {
                "label": "Probable Header / Shared Prefix",
                "offset": "0x0000",
                "length": str(header_len),
                "value": format_hex_preview(byte_values[:header_len], max_bytes=min(header_len, 12)),
                "notes": f"shared with burst {best_prefix_peer.burst_index}; likely sync/header candidate",
                "children": [
                    {
                        "label": "ASCII Preview",
                        "offset": "",
                        "length": "",
                        "value": format_ascii_preview(byte_values[:header_len], max_bytes=min(header_len, 16)),
                        "notes": "usually non-printable for RF sync/header bytes",
                        "children": [],
                    }
                ],
            }
        )

    for hint in length_hints:
        hint_bytes = byte_values[hint["offset"] : hint["offset"] + hint["length"]]
        entries.append(
            {
                "label": "Possible Length Field",
                "offset": f"0x{hint['offset']:04X}",
                "length": str(hint["length"]),
                "value": format_hex_preview(hint_bytes, max_bytes=hint["length"]),
                "notes": f"{hint['note']} (value={hint['value']})",
                "children": [
                    {
                        "label": "Big-endian Value",
                        "offset": "",
                        "length": "",
                        "value": str(hint["value"]),
                        "notes": "candidate interpretation only",
                        "children": [],
                    }
                ],
            }
        )

    trailer_len = 0
    trailer_note = ""
    if best_suffix_len >= 2:
        trailer_len = best_suffix_len
        trailer_note = f"shared with burst {best_suffix_peer.burst_index}; may be a trailer or terminator"
    elif same_length_peers and header_len >= 3 and size - header_len >= 2:
        trailer_len = min(4, size - header_len)
        trailer_note = "same-length family with varying body; last bytes may include CRC/trailer"

    payload_start = header_len
    payload_end = size - trailer_len if trailer_len else size
    if payload_end > payload_start:
        payload_len = payload_end - payload_start
        payload_preview = byte_values[payload_start:payload_end]
        entries.append(
            {
                "label": "Probable Payload",
                "offset": f"0x{payload_start:04X}",
                "length": str(payload_len),
                "value": format_hex_preview(payload_preview, max_bytes=12),
                "notes": f"entropy={byte_entropy(payload_preview):.2f} bits/byte",
                "children": [
                    {
                        "label": "ASCII Preview",
                        "offset": "",
                        "length": "",
                        "value": format_ascii_preview(payload_preview, max_bytes=24),
                        "notes": "low printable ratio usually means binary payload, not text",
                        "children": [],
                    }
                ],
            }
        )

    if trailer_len:
        trailer_bytes = byte_values[size - trailer_len :]
        entries.append(
            {
                "label": "Possible Trailer / CRC",
                "offset": f"0x{size - trailer_len:04X}",
                "length": str(trailer_len),
                "value": format_hex_preview(trailer_bytes, max_bytes=trailer_len),
                "notes": trailer_note,
                "children": [
                    {
                        "label": "Interpretation",
                        "offset": "",
                        "length": "",
                        "value": "CRC / trailer candidate",
                        "notes": "verify only after body length and header are stable across a family",
                        "children": [],
                    }
                ],
            }
        )

    if crc_matches:
        entries.append(
            {
                "label": "CRC Checks",
                "offset": "",
                "length": "",
                "value": ", ".join(f"{item['name']} ({item['endian']})" for item in crc_matches[:4]),
                "notes": "raw frame body matches a common CRC trailer candidate",
                "children": [],
            }
        )

    if postprocess_hints:
        entries.append(
            {
                "label": "Post-Decode Transform Hints",
                "offset": "",
                "length": "",
                "value": postprocess_hints[0]["name"],
                "notes": "transforms that make the frame look less random or expose a CRC",
                "children": [
                    {
                        "label": hint["name"],
                        "offset": "",
                        "length": "",
                        "value": hint["hex_preview"],
                        "notes": hint["notes"],
                        "children": [
                            {
                                "label": "ASCII Preview",
                                "offset": "",
                                "length": "",
                                "value": hint["ascii_preview"],
                                "notes": (
                                    f"entropy={hint['entropy']:.2f}, printable={hint['printable_ratio']:.2f}, "
                                    f"idle={hint['idle_ratio']:.2f}"
                                ),
                                "children": [],
                            }
                        ],
                    }
                    for hint in postprocess_hints
                ],
            }
        )

    top_bytes = ", ".join(f"0x{byte_val:02X} x{count}" for byte_val, count in top_byte_values(byte_values))
    if top_bytes:
        entries.append(
            {
                "label": "Byte Histogram Hints",
                "offset": "",
                "length": "",
                "value": top_bytes,
                "notes": f"overall entropy={byte_entropy(byte_values):.2f} bits/byte",
                "children": [],
            }
        )

    if peers:
        families = cluster_burst_families(result)
        family_note = ""
        for family in families:
            if report.burst_index in family.burst_indexes:
                family_note = describe_burst_family(result, family)
                break
        entries.append(
            {
                "label": "Burst Family",
                "offset": "",
                "length": "",
                "value": f"{len(peers) + 1} similar bursts",
                "notes": f"{len(same_length_peers)} peers share the same byte count",
                "children": [
                    {
                        "label": "Peer Bursts",
                        "offset": "",
                        "length": "",
                        "value": ", ".join(str(peer.burst_index) for peer in peers[:12]),
                        "notes": family_note or "same-method bursts with similar symbol rate and byte count",
                        "children": [],
                    }
                ],
            }
        )

    return entries


def build_reports(
    iq: np.ndarray,
    sample_rate: float,
    dominant_offset_hz: float,
    occupied_bw_hz: float,
    saved_method: str,
    symbol_rate_hint_hz: Optional[float],
    requested_mode: str,
    min_burst_ms: float,
    padding_ms: float,
) -> tuple[np.ndarray, float, list[BurstReport]]:
    decode_iq, decode_rate = isolate_signal_region(iq, sample_rate, dominant_offset_hz, occupied_bw_hz)
    bursts = detect_bursts(decode_iq, decode_rate, min_burst_ms=min_burst_ms, padding_ms=padding_ms)
    mode_order = infer_mode_order(requested_mode, saved_method)
    reports = []
    for burst_index, (start, end) in enumerate(bursts, start=1):
        burst_iq = decode_iq[start:end]
        candidate = decode_burst(burst_iq, decode_rate, mode_order, symbol_rate_hint_hz)
        reports.append(
            BurstReport(
                burst_index=burst_index,
                start_s=float(start / decode_rate),
                end_s=float(end / decode_rate),
                duration_s=float((end - start) / decode_rate),
                candidate=candidate,
            )
        )
    return decode_iq, decode_rate, reports


def build_analysis_result(
    recording_label: str,
    metadata: dict,
    decode_rate: float,
    reports: list[BurstReport],
    requested_mode: str,
    min_burst_ms: float,
    padding_ms: float,
) -> AnalysisResult:
    iq = metadata["iq"]
    clip_seconds = float(iq.size / max(float(metadata["sample_rate"]), 1.0))
    return AnalysisResult(
        recording_label=str(recording_label),
        timestamp_utc=str(metadata["timestamp_utc"]),
        center_freq_hz=float(metadata["center_freq_hz"]),
        sample_rate=float(metadata["sample_rate"]),
        clip_seconds=clip_seconds,
        saved_decoder_method=str(metadata["decoder_method"]),
        saved_decoder_confidence=float(metadata["decoder_confidence"]),
        decode_rate=float(decode_rate),
        requested_mode=str(requested_mode or "auto"),
        min_burst_ms=float(min_burst_ms),
        padding_ms=float(padding_ms),
        reports=list(reports),
    )


def load_recording(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {
            "iq": np.asarray(data["iq"]).astype(np.complex64, copy=False),
            "timestamp_utc": str(data["timestamp_utc"]) if "timestamp_utc" in data else path.stem,
            "center_freq_hz": float(data["center_freq_hz"]) if "center_freq_hz" in data else 0.0,
            "sample_rate": float(data["sample_rate"]) if "sample_rate" in data else 0.0,
            "occupied_bw_hz": float(data["occupied_bw_hz"]) if "occupied_bw_hz" in data else 0.0,
            "dominant_offset_hz": float(data["dominant_offset_hz"]) if "dominant_offset_hz" in data else 0.0,
            "decoder_method": str(data["decoder_method"]) if "decoder_method" in data else "",
            "decoder_symbol_rate_hz": float(data["decoder_symbol_rate_hz"]) if "decoder_symbol_rate_hz" in data else 0.0,
            "decoder_confidence": float(data["decoder_confidence"]) if "decoder_confidence" in data else 0.0,
        }


def json_ready_reports(reports: list[BurstReport]) -> list[dict]:
    ready = []
    for report in reports:
        candidate = report.candidate
        ready.append(
            {
                "burst_index": report.burst_index,
                "start_s": report.start_s,
                "end_s": report.end_s,
                "duration_s": report.duration_s,
                "candidate": None
                if candidate is None
                else {
                    "method": candidate.method,
                    "confidence": candidate.confidence,
                    "symbol_rate_hz": candidate.symbol_rate_hz,
                    "symbol_samples": candidate.symbol_samples,
                    "byte_offset": candidate.byte_offset,
                    "byte_count": int(candidate.byte_values.size),
                    "hex": " ".join(f"{int(byte_val):02X}" for byte_val in candidate.byte_values),
                    "hex_preview": format_hex_preview(candidate.byte_values, max_bytes=16),
                    "ascii_preview": format_ascii_preview(candidate.byte_values, max_bytes=24),
                    "printable_ratio": candidate.printable_ratio,
                    "unique_ratio": candidate.unique_ratio,
                    "idle_ratio": candidate.idle_ratio,
                    "crc_matches": [f"{item['name']} ({item['endian']})" for item in detect_crc_matches(candidate.byte_values)],
                    "postprocess_hints": build_postprocess_hints(candidate.byte_values),
                    "notes": candidate.notes,
                },
            }
        )
    return ready


def json_ready_families(result: AnalysisResult) -> list[dict]:
    families = []
    for family in cluster_burst_families(result):
        families.append(
            {
                "family_id": family.family_id,
                "method": family.method,
                "count": family.count,
                "burst_indexes": list(family.burst_indexes),
                "min_symbol_rate_hz": family.min_symbol_rate_hz,
                "max_symbol_rate_hz": family.max_symbol_rate_hz,
                "min_bytes": family.min_bytes,
                "max_bytes": family.max_bytes,
                "avg_confidence": family.avg_confidence,
                "notes": describe_burst_family(result, family),
            }
        )
    return families


def json_ready_combined_output(result: AnalysisResult) -> dict:
    decoded_reports = [report for report in result.reports if report.candidate is not None]
    undecoded_reports = [report for report in result.reports if report.candidate is None]

    combined_hex_lines = []
    combined_ascii_lines = []
    combined_decode_lines = []
    total_decoded_bytes = 0

    for report in decoded_reports:
        candidate = report.candidate
        total_decoded_bytes += int(candidate.byte_values.size)
        crc_matches = detect_crc_matches(candidate.byte_values)
        postprocess_hints = build_postprocess_hints(candidate.byte_values, limit=1)
        hint_suffix = ""
        if crc_matches:
            hint_suffix = " | crc=" + ", ".join(f"{item['name']} ({item['endian']})" for item in crc_matches[:2])
        elif postprocess_hints:
            hint_suffix = f" | hint={postprocess_hints[0]['name']}"
        combined_hex_lines.append(f"[B{report.burst_index}] {format_hex(candidate.byte_values, 0)}")
        combined_ascii_lines.append(
            f"[B{report.burst_index}] {' '.join(format_ascii_lines(candidate.byte_values, 0))}"
        )
        combined_decode_lines.append(
            f"[B{report.burst_index}] {candidate.method} | conf={candidate.confidence:.3f} | "
            f"sym_rate={candidate.symbol_rate_hz:.1f} | bytes={candidate.byte_values.size} | {candidate.notes}{hint_suffix}"
        )

    for report in undecoded_reports:
        combined_decode_lines.append(
            f"[B{report.burst_index}] no candidate | start={report.start_s:.6f}s | dur={report.duration_s:.6f}s"
        )

    return {
        "decoded_burst_count": len(decoded_reports),
        "undecoded_burst_count": len(undecoded_reports),
        "decoded_burst_indexes": [report.burst_index for report in decoded_reports],
        "undecoded_burst_indexes": [report.burst_index for report in undecoded_reports],
        "total_decoded_bytes": total_decoded_bytes,
        "combined_hex": "\n".join(combined_hex_lines) if combined_hex_lines else "",
        "combined_ascii": "\n".join(combined_ascii_lines) if combined_ascii_lines else "",
        "combined_decode": "\n".join(combined_decode_lines) if combined_decode_lines else "",
    }


def recording_track_signature(result: AnalysisResult, file_name: str, path: str) -> dict:
    decoded_reports = [report for report in result.reports if report.candidate is not None]
    families = cluster_burst_families(result)
    methods: dict[str, int] = {}
    symbol_rates = []
    byte_counts = []
    header_signatures = []
    for report in decoded_reports:
        candidate = report.candidate
        methods[candidate.method] = methods.get(candidate.method, 0) + 1
        symbol_rates.append(float(candidate.symbol_rate_hz))
        byte_counts.append(int(candidate.byte_values.size))
    for family in families[:6]:
        reports = family_reports(result, family)
        if not reports:
            continue
        prefix_len = shared_prefix_len_for_reports(reports)
        sample = reports[0].candidate.byte_values
        if prefix_len >= 2:
            signature_bytes = sample[: min(prefix_len, 8)]
        else:
            signature_bytes = sample[: min(int(sample.size), 6)]
        if signature_bytes.size:
            header_signatures.append(format_hex_preview(signature_bytes, max_bytes=int(signature_bytes.size)))

    timestamp = None
    try:
        timestamp = datetime.fromisoformat(result.timestamp_utc)
    except ValueError:
        timestamp = None

    dominant_method = max(methods.items(), key=lambda item: item[1])[0] if methods else result.saved_decoder_method
    median_symbol_rate_hz = float(np.median(symbol_rates)) if symbol_rates else 0.0
    return {
        "file": str(file_name),
        "path": str(path),
        "timestamp_utc": result.timestamp_utc,
        "timestamp_dt": timestamp,
        "center_freq_hz": float(result.center_freq_hz),
        "decoded_burst_count": len(decoded_reports),
        "family_count": len(families),
        "dominant_method": str(dominant_method),
        "method_counts": methods,
        "median_symbol_rate_hz": median_symbol_rate_hz,
        "min_symbol_rate_hz": min(symbol_rates) if symbol_rates else 0.0,
        "max_symbol_rate_hz": max(symbol_rates) if symbol_rates else 0.0,
        "min_byte_count": min(byte_counts) if byte_counts else 0,
        "max_byte_count": max(byte_counts) if byte_counts else 0,
        "header_signatures": header_signatures[:4],
    }


def score_track_link(left: dict, right: dict) -> tuple[float, list[str]]:
    reasons = []
    score = 0.0
    left_time = left.get("timestamp_dt")
    right_time = right.get("timestamp_dt")
    if left_time is not None and right_time is not None:
        gap_s = abs((right_time - left_time).total_seconds())
        if gap_s <= 2.0:
            score += 1.4
            reasons.append(f"time gap {gap_s:.2f}s")
        elif gap_s <= 6.0:
            score += 1.0
            reasons.append(f"time gap {gap_s:.2f}s")
        elif gap_s <= 12.0:
            score += 0.5
            reasons.append(f"time gap {gap_s:.2f}s")
        else:
            return 0.0, [f"time gap {gap_s:.2f}s too large"]

    if left["dominant_method"] == right["dominant_method"]:
        score += 0.9
        reasons.append("same dominant method")

    left_rate = float(left.get("median_symbol_rate_hz") or 0.0)
    right_rate = float(right.get("median_symbol_rate_hz") or 0.0)
    if left_rate > 0.0 and right_rate > 0.0:
        rate_ratio = max(left_rate, right_rate) / max(min(left_rate, right_rate), 1.0)
        if rate_ratio <= 1.08:
            score += 1.2
            reasons.append("symbol rate nearly identical")
        elif rate_ratio <= 1.18:
            score += 0.8
            reasons.append("symbol rate similar")
        elif rate_ratio <= 1.35:
            score += 0.3
            reasons.append("symbol rate loosely similar")

    if left["decoded_burst_count"] and right["decoded_burst_count"]:
        left_min = int(left.get("min_byte_count") or 0)
        left_max = int(left.get("max_byte_count") or 0)
        right_min = int(right.get("min_byte_count") or 0)
        right_max = int(right.get("max_byte_count") or 0)
        if min(left_max, right_max) >= max(left_min, right_min):
            score += 0.5
            reasons.append("byte-count ranges overlap")

    left_headers = set(left.get("header_signatures") or [])
    right_headers = set(right.get("header_signatures") or [])
    if left_headers and right_headers:
        shared_headers = sorted(left_headers.intersection(right_headers))
        if shared_headers:
            score += 1.6
            reasons.append(f"shared header {shared_headers[0]}")

    return score, reasons


def cluster_recording_tracks(signatures: list[dict]) -> list[dict]:
    ordered = sorted(
        [signature for signature in signatures if signature.get("timestamp_dt") is not None],
        key=lambda item: item["timestamp_dt"],
    )
    tracks: list[dict] = []
    for signature in ordered:
        best_index = None
        best_score = 0.0
        best_reasons = []
        for track_index, track in enumerate(tracks):
            score, reasons = score_track_link(track["last_signature"], signature)
            if score > best_score:
                best_index = track_index
                best_score = score
                best_reasons = reasons
        if best_index is None or best_score < 2.1:
            tracks.append(
                {
                    "signatures": [signature],
                    "last_signature": signature,
                    "link_reasons": [],
                }
            )
            continue
        tracks[best_index]["signatures"].append(signature)
        tracks[best_index]["last_signature"] = signature
        tracks[best_index]["link_reasons"].append(best_reasons)

    output = []
    for track_index, track in enumerate(tracks, start=1):
        signatures_in_track = track["signatures"]
        center_freqs = [float(item["center_freq_hz"]) for item in signatures_in_track]
        methods: dict[str, int] = {}
        headers = []
        for item in signatures_in_track:
            for method, count in (item.get("method_counts") or {}).items():
                methods[method] = methods.get(method, 0) + int(count)
            headers.extend(item.get("header_signatures") or [])
        start_time = signatures_in_track[0]["timestamp_dt"]
        end_time = signatures_in_track[-1]["timestamp_dt"]
        duration_s = max((end_time - start_time).total_seconds(), 0.0)
        unique_headers = list(dict.fromkeys(headers))
        dominant_methods = ", ".join(f"{method} x{count}" for method, count in sorted(methods.items(), key=lambda item: item[1], reverse=True)[:4])
        ordered_obs = []
        last_item = None
        hop_steps_hz = []
        time_gaps_s = []
        for seq_index, item in enumerate(signatures_in_track, start=1):
            delta_freq_hz = None
            delta_time_s = None
            if last_item is not None:
                delta_freq_hz = float(item["center_freq_hz"]) - float(last_item["center_freq_hz"])
                hop_steps_hz.append(abs(delta_freq_hz))
                if item.get("timestamp_dt") is not None and last_item.get("timestamp_dt") is not None:
                    delta_time_s = (item["timestamp_dt"] - last_item["timestamp_dt"]).total_seconds()
                    time_gaps_s.append(abs(delta_time_s))
            ordered_obs.append(
                {
                    "sequence_index": seq_index,
                    "file": item["file"],
                    "path": item["path"],
                    "timestamp_utc": item["timestamp_utc"],
                    "center_freq_mhz": round(float(item["center_freq_hz"]) / 1e6, 6),
                    "dominant_method": item["dominant_method"],
                    "decoded_burst_count": int(item["decoded_burst_count"]),
                    "header_signatures": list(item.get("header_signatures") or []),
                    "delta_freq_mhz": None if delta_freq_hz is None else round(delta_freq_hz / 1e6, 6),
                    "delta_time_s": None if delta_time_s is None else round(float(delta_time_s), 6),
                }
            )
            last_item = item
        output.append(
            {
                "track_id": f"T{track_index}",
                "recording_count": len(signatures_in_track),
                "time_start_utc": signatures_in_track[0]["timestamp_utc"],
                "time_end_utc": signatures_in_track[-1]["timestamp_utc"],
                "duration_s": round(float(duration_s), 3),
                "center_freqs_mhz": [round(freq / 1e6, 3) for freq in sorted(set(center_freqs))],
                "hop_span_mhz": round((max(center_freqs) - min(center_freqs)) / 1e6, 3) if center_freqs else 0.0,
                "median_hop_step_mhz": round(float(np.median(hop_steps_hz)) / 1e6, 6) if hop_steps_hz else 0.0,
                "median_gap_s": round(float(np.median(time_gaps_s)), 6) if time_gaps_s else 0.0,
                "dominant_methods": dominant_methods,
                "header_signatures": unique_headers[:6],
                "files": [item["file"] for item in signatures_in_track],
                "paths": [item["path"] for item in signatures_in_track],
                "recordings": ordered_obs,
                "notes": (
                    "possible frequency-hopping / retuning activity"
                    if len(set(center_freqs)) > 1 and len(signatures_in_track) > 1
                    else "time-correlated recording cluster"
                ),
            }
        )
    return output


def render_summary_text(result: AnalysisResult) -> str:
    decoded_reports = [report for report in result.reports if report.candidate is not None]
    undecoded_count = len(result.reports) - len(decoded_reports)
    total_bytes = sum(int(report.candidate.byte_values.size) for report in decoded_reports)
    families = cluster_burst_families(result)
    average_confidence = (
        sum(float(report.candidate.confidence) for report in decoded_reports) / float(len(decoded_reports))
        if decoded_reports
        else 0.0
    )
    symbol_rates = [float(report.candidate.symbol_rate_hz) for report in decoded_reports]
    method_counts: dict[str, int] = {}
    for report in decoded_reports:
        method = str(report.candidate.method)
        method_counts[method] = method_counts.get(method, 0) + 1

    lines = [
        f"Source:         {result.recording_label}",
        f"Timestamp:      {result.timestamp_utc}",
        f"Center:         {result.center_freq_hz / 1e6:.3f} MHz",
        f"Sample rate:    {result.sample_rate / 1e6:.3f} MS/s",
        f"Duration:       {result.clip_seconds:.6f} s",
        f"Saved decoder:  {result.saved_decoder_method} (confidence {result.saved_decoder_confidence:.2f})",
        f"Decode rate:    {result.decode_rate / 1e6:.3f} MS/s",
        f"Analysis mode:  {analysis_mode_label(result.requested_mode)}",
        f"Burst gate:     min {result.min_burst_ms:.2f} ms  |  padding {result.padding_ms:.2f} ms",
        f"Bursts:         {len(result.reports)} total  |  {len(decoded_reports)} decoded  |  {undecoded_count} no-candidate",
        f"Families:       {len(families)} decoded families",
        f"Payload bytes:  {total_bytes}",
    ]

    if decoded_reports:
        lines.append(
            f"Confidence:     avg {average_confidence:.3f}"
        )
        lines.append(
            f"Symbol rates:   min {min(symbol_rates):.1f} Hz  |  max {max(symbol_rates):.1f} Hz"
        )
        method_summary = ", ".join(f"{method} x{count}" for method, count in sorted(method_counts.items()))
        lines.append(f"Methods:        {method_summary}")
        lines.append("Interpretation: recovered bursts look like binary packet frames, not plaintext payloads.")
        if len({round(rate, 1) for rate in symbol_rates}) > 1:
            lines.append("Observation:    multiple symbol-rate groups were detected, which may indicate different packet types or estimator drift.")
        if average_confidence >= 0.9:
            lines.append("Observation:    decoder confidence is consistently high enough to justify protocol-level follow-up.")
        if families:
            lines.append("Family view:")
            for family in families[:6]:
                byte_text = (
                    f"{family.min_bytes}"
                    if family.min_bytes == family.max_bytes
                    else f"{family.min_bytes}-{family.max_bytes}"
                )
                rate_text = (
                    f"{family.min_symbol_rate_hz / 1e3:.1f}k"
                    if abs(family.max_symbol_rate_hz - family.min_symbol_rate_hz) < 1.0
                    else f"{family.min_symbol_rate_hz / 1e3:.1f}-{family.max_symbol_rate_hz / 1e3:.1f}k"
                )
                lines.append(
                    f"  {family.family_id}: {family.method} x{family.count}  |  {rate_text} sym/s  |  "
                    f"{byte_text} bytes  |  {describe_burst_family(result, family)}"
                )

    return "\n".join(lines)


def render_interpretation_text(result: AnalysisResult, report: BurstReport) -> str:
    lines = [
        f"Source: {result.recording_label}",
        f"Burst {report.burst_index}/{len(result.reports)} interpretation",
    ]

    if report.candidate is None:
        lines.extend(
            [
                "",
                "This burst had enough energy to be segmented, but none of the selected decoder families",
                "produced a stable byte stream.",
                "",
                "Try:",
                "- Auto mode if you forced a single decoder",
                "- a slightly smaller burst gate or a bit more padding",
                "- comparing nearby bursts for a stronger packet of the same type",
            ]
        )
        return "\n".join(lines)

    candidate = report.candidate
    byte_values = candidate.byte_values
    peers = find_family_peers(result, report)
    peer_prefix = []
    peer_suffix = []
    for peer in peers:
        peer_bytes = peer.candidate.byte_values
        peer_prefix.append(common_prefix_len(byte_values, peer_bytes))
        peer_suffix.append(common_suffix_len(byte_values, peer_bytes))
    shared_prefix = max(peer_prefix, default=0)
    shared_suffix = max(peer_suffix, default=0)
    length_hints = detect_length_fields(byte_values)
    entropy = byte_entropy(byte_values)
    crc_matches = detect_crc_matches(byte_values)
    postprocess_hints = build_postprocess_hints(byte_values)
    method_lower = candidate.method.lower()
    lines.append("")
    lines.append(
        f"This looks like a real packet-shaped burst: about {candidate.byte_values.size} recovered bytes over "
        f"{report.duration_s * 1e3:.3f} ms at roughly {candidate.symbol_rate_hz / 1e3:.1f} ksym/s."
    )

    if candidate.printable_ratio < 0.15:
        lines.append("The payload does not look like plaintext or ASCII.")
    else:
        lines.append("There is a small amount of printable content, but it still does not look like plain text.")

    if candidate.unique_ratio > 0.85 and candidate.idle_ratio < 0.05:
        lines.append("Byte diversity is high and idle bytes are rare, which usually means scrambling, coding, or encryption.")
    elif candidate.idle_ratio > 0.15:
        lines.append("There are noticeable idle bytes, so this may include padding, framing gaps, or weak slicing regions.")
    else:
        lines.append("The byte stream looks structured enough to be a protocol frame, but not yet field-decoded.")

    if shared_prefix >= 2:
        lines.append(f"A repeated prefix of about {shared_prefix} byte(s) appears across similar bursts, which is a strong header or sync-word hint.")
    if length_hints:
        hint = length_hints[0]
        lines.append(
            f"Bytes near offset 0x{hint['offset']:02X} may act like a length field because {hint['note']}."
        )
    if shared_suffix >= 2:
        lines.append(f"The last {shared_suffix} byte(s) repeat across similar bursts, which often points to a trailer or CRC region.")
    if crc_matches:
        crc_summary = ", ".join(f"{item['name']} ({item['endian']})" for item in crc_matches[:3])
        lines.append(f"A common CRC already matches the trailer on the raw frame: {crc_summary}.")
    elif postprocess_hints:
        hint = postprocess_hints[0]
        lines.append(
            f"A post-process transform stands out: {hint['name']} makes the bytes look less random ({hint['notes']})."
        )

    if candidate.byte_offset != 0:
        lines.append(
            f"Byte alignment is still heuristic: the current best framing starts at bit offset {candidate.byte_offset}."
        )
    else:
        lines.append("Byte alignment did not require an extra bit offset, which is a stronger sign of a true frame boundary.")

    if "manchester" in method_lower:
        lines.append("Manchester line coding is a plausible fit here, but whitening, FEC, CRC verification, and protocol fields are still unresolved.")
    elif "nrzi" in method_lower or "differential" in method_lower:
        lines.append("Differential / NRZI decoding looks plausible, but higher-layer framing and descrambling still need confirmation.")
    elif "fsk" in method_lower:
        lines.append("This is still only symbol slicing. Whitening, FEC, CRCs, and higher-level protocol fields are not decoded yet.")
    elif "ask" in method_lower or "ook" in method_lower:
        lines.append("This came from an envelope-based decoder, so polarity, thresholding, and pulse shaping still matter.")
    elif "uart" in method_lower:
        lines.append("This looks UART-like at the bit level, but framing confidence still depends on start/stop-bit assumptions.")

    if report.start_s <= (1.0 / max(result.decode_rate, 1.0)) * 8.0:
        lines.append("The capture begins at or extremely close to this burst, so the clip likely opened on packet start.")

    lines.extend(
        [
            "",
            "Decode / decrypt guidance:",
        ]
    )
    if peers:
        lines.append(f"- Compare this burst against {len(peers)} similar burst(s) first; repeated bytes are more useful than a single frame.")
    else:
        lines.append("- Capture more bursts of the same type first; de-framing a single burst is much harder.")
    if shared_prefix >= 2:
        lines.append(f"- Treat the first {shared_prefix} byte(s) as the tentative sync/header and align other bursts to it.")
    else:
        lines.append("- Confirm the true frame start by finding a repeated prefix across same-rate bursts.")
    if length_hints:
        lines.append("- Test the candidate length field before touching payload semantics or CRC guesses.")
    else:
        lines.append("- Search the first 8 bytes for a field that tracks total length or payload length.")
    if entropy >= 7.2 and candidate.printable_ratio < 0.12:
        lines.append("- The payload is high-entropy. Try dewhitening / descrambling first; if it stays random after that, it is probably encrypted and not realistically decryptable without keys.")
    else:
        lines.append("- The payload is not maximally random, so try whitening, interleaving, and CRC removal before assuming encryption.")
    if postprocess_hints:
        hint = postprocess_hints[0]
        lines.append(f"- Start with {hint['name']} before trying speculative decryption; it already improved structure in the recovered bytes.")
    if shared_suffix >= 2:
        lines.append("- Test the repeated trailer bytes as a CRC or footer once the frame body is stable.")
    else:
        lines.append("- Once families are stable, sweep common 16-bit and 32-bit CRCs across the last bytes.")

    lines.extend(
        [
            "",
            "Next useful interpretation steps:",
            "- compare prefixes across bursts for a repeated sync word or header",
            "- cluster bursts by byte length and symbol rate to separate packet types",
            "- test for CRC/length fields before attempting payload semantics",
            "- only consider real encryption after whitening and CRC hypotheses fail",
        ]
    )
    return "\n".join(lines)


def render_burst_detail(result: AnalysisResult, report: BurstReport, max_hex_bytes: int) -> str:
    lines = [
        f"Packet: B{report.burst_index} / {len(result.reports)}",
        f"Time:   start {report.start_s:.6f} s  |  dur {report.duration_s:.6f} s  |  end {report.end_s:.6f} s",
    ]

    if report.candidate is None:
        lines.extend(
            [
                "Decode: no candidate",
                "",
                "Bytes:",
                "No stable byte stream recovered from this burst.",
            ]
        )
        return "\n".join(lines)

    candidate = report.candidate
    crc_matches = detect_crc_matches(candidate.byte_values)
    postprocess_hints = build_postprocess_hints(candidate.byte_values)
    lines.extend(
        [
            f"Decode: {candidate.method}  |  conf={candidate.confidence:.3f}  |  rate={candidate.symbol_rate_hz:.1f} Hz  |  bytes={candidate.byte_values.size}",
            f"Frame:  byte_offset={candidate.byte_offset}  |  idle={candidate.idle_ratio:.2f}  |  unique24={candidate.unique_ratio:.2f}",
            "",
            "Bytes:",
        ]
    )
    for line in format_hex_dump_lines(candidate.byte_values, max_hex_bytes):
        lines.append(line)
    lines.extend(["", "Notes:", candidate.notes])
    if crc_matches:
        lines.extend(
            [
                "",
                "CRC hints:",
                ", ".join(f"{item['name']} ({item['endian']})" for item in crc_matches),
            ]
        )
    if postprocess_hints:
        lines.extend(["", "Post-process hints:"])
        for hint in postprocess_hints:
            lines.append(f"- {hint['name']}: {hint['notes']}")
    return "\n".join(lines)


def render_text_report(
    recording_label: str,
    metadata: dict,
    decode_rate: float,
    reports: list[BurstReport],
    max_hex_bytes: int,
    requested_mode: str = "auto",
    min_burst_ms: float = 0.35,
    padding_ms: float = 0.25,
) -> str:
    result = build_analysis_result(
        recording_label=recording_label,
        metadata=metadata,
        decode_rate=decode_rate,
        reports=reports,
        requested_mode=requested_mode,
        min_burst_ms=min_burst_ms,
        padding_ms=padding_ms,
    )
    lines = [render_summary_text(result)]
    for report in result.reports:
        lines.append("")
        lines.append(render_burst_detail(result, report, max_hex_bytes))
    return "\n".join(lines)


def print_text_report(
    recording_path: Path,
    metadata: dict,
    decode_rate: float,
    reports: list[BurstReport],
    max_hex_bytes: int,
    requested_mode: str = "auto",
    min_burst_ms: float = 0.35,
    padding_ms: float = 0.25,
) -> None:
    print(
        render_text_report(
            str(recording_path),
            metadata,
            decode_rate,
            reports,
            max_hex_bytes,
            requested_mode=requested_mode,
            min_burst_ms=min_burst_ms,
            padding_ms=padding_ms,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze full DroneDetect recordings burst-by-burst.")
    parser.add_argument("recording", type=Path, help="Path to a saved .npz recording.")
    parser.add_argument(
        "--mode",
        choices=("auto", "fsk", "ask", "manchester", "nrzi", "uart"),
        default="auto",
        help="Decoder family to try. 'auto' follows the saved detector method.",
    )
    parser.add_argument(
        "--min-burst-ms",
        type=float,
        default=0.35,
        help="Minimum burst duration in milliseconds before decoding.",
    )
    parser.add_argument(
        "--padding-ms",
        type=float,
        default=0.25,
        help="Padding added before and after each detected burst.",
    )
    parser.add_argument(
        "--max-hex-bytes",
        type=int,
        default=64,
        help="Maximum bytes to print per burst. Use 0 to print the full byte stream.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recording_path = args.recording.expanduser().resolve()
    metadata = load_recording(recording_path)
    _, decode_rate, reports = build_reports(
        iq=metadata["iq"],
        sample_rate=metadata["sample_rate"],
        dominant_offset_hz=metadata["dominant_offset_hz"],
        occupied_bw_hz=metadata["occupied_bw_hz"],
        saved_method=metadata["decoder_method"],
        symbol_rate_hint_hz=metadata["decoder_symbol_rate_hz"] or None,
        requested_mode=args.mode,
        min_burst_ms=args.min_burst_ms,
        padding_ms=args.padding_ms,
    )

    if args.json:
        payload = {
            "recording": str(recording_path),
            "timestamp_utc": metadata["timestamp_utc"],
            "center_freq_hz": metadata["center_freq_hz"],
            "sample_rate": metadata["sample_rate"],
            "saved_decoder_method": metadata["decoder_method"],
            "saved_decoder_confidence": metadata["decoder_confidence"],
            "requested_mode": args.mode,
            "min_burst_ms": args.min_burst_ms,
            "padding_ms": args.padding_ms,
            "reports": json_ready_reports(reports),
        }
        print(json.dumps(payload, indent=2))
    else:
        print_text_report(
            recording_path,
            metadata,
            decode_rate,
            reports,
            args.max_hex_bytes,
            requested_mode=args.mode,
            min_burst_ms=args.min_burst_ms,
            padding_ms=args.padding_ms,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
