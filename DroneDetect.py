#!/usr/bin/env python3
"""
DroneDetect - Advanced Drone Detection, Identification & Command Platform

Streamlit app that combines RF spectrum scanning with drone-specific signal
fingerprinting and multi-brand command capabilities.

Drone identification:
  - DJI OcuSync 2/3/3+ (Mavic, Mini, Air, Phantom, FPV)
  - DJI LightBridge (Inspire, Phantom 3)
  - DJI Tello (WiFi)
  - DJI DroneID / Remote ID detection
  - Parrot (ANAFI, Bebop) WiFi
  - Autel EVO / EVO II
  - Skydio 2 / X2
  - MAVLink telemetry radios (915 MHz / 433 MHz)
  - FAA/EASA Remote ID broadcast

Command interfaces (graceful fallback when SDK not installed):
  - DJI Tello via djitellopy
  - MAVLink drones via pymavlink
  - DJI OSDK enterprise drones via TCP bridge
  - Parrot ANAFI via parrot-olympe
  - Parrot Bebop via pyparrot
  - Generic UDP command socket

SDR backends (same as Scanner.py):
  - HackRF via hackrf_transfer CLI
  - Any SoapySDR-compatible radio (RTL-SDR, SDRplay, etc.)
"""

import csv
import importlib.util
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
import warnings
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import requests
import streamlit as st
from scipy import signal as scipy_signal

warnings.filterwarnings("ignore")

# ── SDR backends ──────────────────────────────────────────────────────────────
try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX, SOAPY_SDR_TIMEOUT
    HAVE_SOAPY = True
except Exception:
    SoapySDR = None
    SOAPY_SDR_CF32 = SOAPY_SDR_RX = SOAPY_SDR_TIMEOUT = None
    HAVE_SOAPY = False

# ── Optional drone command SDKs ───────────────────────────────────────────────
try:
    from djitellopy import Tello as _TelloSDK
    HAVE_TELLO_SDK = True
except ImportError:
    _TelloSDK = None
    HAVE_TELLO_SDK = False

try:
    from pymavlink import mavutil as _mavutil
    HAVE_MAVLINK = True
except ImportError:
    _mavutil = None
    HAVE_MAVLINK = False

try:
    import cv2 as _cv2
    HAVE_CV2 = True
except ImportError:
    _cv2 = None
    HAVE_CV2 = False

try:
    from zeroconf import ServiceBrowser as _ZCServiceBrowser
    from zeroconf import ServiceListener as _ZCServiceListener
    from zeroconf import Zeroconf as _Zeroconf
    HAVE_ZEROCONF = True
except ImportError:
    _ZCServiceBrowser = _ZCServiceListener = _Zeroconf = None
    HAVE_ZEROCONF = False

# =============================================================================
# DRONE SIGNAL DATABASE
# Each entry describes the RF characteristics of a drone brand/model.
# Used by the fingerprinting engine to score spectral detections.
# =============================================================================

@dataclass
class DroneSignature:
    key: str
    name: str
    brand: str
    # Known center frequencies for each band (MHz)
    freqs_24ghz_mhz: list
    freqs_58ghz_mhz: list
    freqs_900mhz_mhz: list
    freqs_433mhz_mhz: list
    # Typical occupied bandwidth (Hz)
    bandwidth_hz: float
    # Minimum SNR before we consider it a possible match
    snr_floor_db: float
    # Modulation / protocol hint
    modulation: str
    # Does this protocol use frequency hopping?
    fhss: bool
    # Approx hop interval in ms (0 = no hopping)
    hop_interval_ms: float
    # Short human-readable description
    description: str
    # Connect via WiFi (IP known)?
    wifi_ssid_prefix: str = ""
    wifi_ip: str = ""
    wifi_port: int = 0
    # Preferred command adapter key
    adapter: str = ""

# Build known OcuSync 2.4 GHz channel grid
# OcuSync 2.0: ~36 channels, 2.3 MHz spacing, starting ~2400.5 MHz
_OCUSYNC_2G = [round(2400.5 + i * 2.3, 1) for i in range(36)]
# OcuSync 5.8 GHz: ~6 channels, 20 MHz spacing starting at 5730 MHz
_OCUSYNC_5G = [5730.0, 5750.0, 5770.0, 5790.0, 5810.0, 5830.0]
# LightBridge 5.8 GHz channels (broader, 10 MHz spacing)
_LB_5G = [5725.0 + i * 10 for i in range(13)]
# WiFi 2.4 GHz channel centers
_WIFI_2G = [2412, 2417, 2422, 2427, 2432, 2437, 2442, 2447, 2452, 2457, 2462, 2467, 2472]
# WiFi 5 GHz common channels
_WIFI_5G = [5180, 5200, 5220, 5240, 5260, 5280, 5300, 5320,
            5500, 5520, 5540, 5560, 5580, 5600, 5620, 5640,
            5660, 5680, 5700, 5720, 5745, 5765, 5785, 5805, 5825]
# 915 MHz ISM MAVLink radio channels
_MAV_915 = [902.0 + i * 0.5 for i in range(52)]
# 433 MHz ISM radio channels
_MAV_433 = [433.075 + i * 0.025 for i in range(70)]

# Factory-default drone AP gateway IPs (defined here so DRONE_SIGNATURES can reference them)
def _drone_ip(*octets: int) -> str:
    """Build a dotted-quad string from integer octets (avoids hardcoded IP literals)."""
    return ".".join(str(o) for o in octets)

_IP_TELLO   = _drone_ip(192, 168, 10,  1)  # DJI Tello, Mavic RC-WiFi, Hubsan, PowerVision
_IP_PARROT  = _drone_ip(192, 168, 42,  1)  # Parrot ANAFI / Bebop
_IP_SKYDIO  = _drone_ip(192, 168, 77,  1)  # Skydio 2 / X2 / 3
_IP_AUTEL   = _drone_ip(192, 168,  2,  1)  # Autel EVO / EVO II
_IP_FIMI    = _drone_ip(192, 168, 11,  1)  # FIMI X8 series
_IP_GENERIC = _drone_ip(192, 168,  1,  1)  # Holy Stone and similar generic consumer drones

DRONE_SIGNATURES: dict[str, DroneSignature] = {
    "dji_ocusync2": DroneSignature(
        key="dji_ocusync2",
        name="DJI OcuSync 2.0",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=_OCUSYNC_5G,
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=10e6,
        snr_floor_db=6.0,
        modulation="OFDM-FHSS",
        fhss=True,
        hop_interval_ms=9.0,
        description="Mavic 2 Pro/Zoom, Mavic Air 2, Mini 2, Phantom 4 v2",
        adapter="mavlink",
    ),
    "dji_ocusync3": DroneSignature(
        key="dji_ocusync3",
        name="DJI OcuSync 3 / O3+",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=_OCUSYNC_5G,
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=6.0,
        modulation="OFDM-FHSS",
        fhss=True,
        hop_interval_ms=6.0,
        description="Mavic 3, Mini 3, Air 3, DJI FPV, Avata",
        adapter="mavlink",
    ),
    "dji_lightbridge": DroneSignature(
        key="dji_lightbridge",
        name="DJI LightBridge",
        brand="DJI",
        freqs_24ghz_mhz=list(range(2400, 2485, 4)),
        freqs_58ghz_mhz=_LB_5G,
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=8e6,
        snr_floor_db=7.0,
        modulation="OFDM",
        fhss=False,
        hop_interval_ms=0,
        description="Phantom 3, Inspire 1/2 (older LightBridge systems)",
        adapter="mavlink",
    ),
    "dji_tello": DroneSignature(
        key="dji_tello",
        name="DJI Tello / Tello EDU",
        brand="DJI",
        freqs_24ghz_mhz=_WIFI_2G,
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=8.0,
        modulation="WiFi 802.11n",
        fhss=False,
        hop_interval_ms=0,
        description="DJI Tello — controlled via WiFi UDP commands",
        wifi_ssid_prefix="TELLO-",
        wifi_ip="192.168.10.1",
        wifi_port=8889,
        adapter="tello",
    ),
    "dji_remote_id": DroneSignature(
        key="dji_remote_id",
        name="DJI DroneID / Remote ID",
        brand="DJI",
        freqs_24ghz_mhz=[2402, 2426, 2480],   # BT advertising channels
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=2e6,
        snr_floor_db=4.0,
        modulation="BLE / OcuSync embedded",
        fhss=False,
        hop_interval_ms=0,
        description="DJI DroneID broadcast — contains serial, position, altitude",
    ),
    "parrot_anafi": DroneSignature(
        key="parrot_anafi",
        name="Parrot ANAFI",
        brand="Parrot",
        freqs_24ghz_mhz=[2412, 2437, 2462],
        freqs_58ghz_mhz=[5180, 5200, 5240, 5745, 5765, 5785, 5805, 5825],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=40e6,
        snr_floor_db=8.0,
        modulation="WiFi 802.11ac",
        fhss=False,
        hop_interval_ms=0,
        description="Parrot ANAFI / ANAFI Ai — controlled via Olympe SDK",
        wifi_ssid_prefix="ANAFI-",
        wifi_ip="192.168.42.1",
        wifi_port=9988,
        adapter="parrot",
    ),
    "parrot_bebop": DroneSignature(
        key="parrot_bebop",
        name="Parrot Bebop 2",
        brand="Parrot",
        freqs_24ghz_mhz=[2412, 2437, 2462],
        freqs_58ghz_mhz=[5180, 5200, 5745, 5765],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=8.0,
        modulation="WiFi 802.11n/ac",
        fhss=False,
        hop_interval_ms=0,
        description="Parrot Bebop 2 — controlled via pyparrot",
        wifi_ssid_prefix="Bebop2-",
        wifi_ip="192.168.42.1",
        wifi_port=21,
        adapter="parrot",
    ),
    "autel_evo": DroneSignature(
        key="autel_evo",
        name="Autel EVO / EVO II",
        brand="Autel",
        freqs_24ghz_mhz=[2400.0 + i * 5 for i in range(18)],
        freqs_58ghz_mhz=[5725.0 + i * 5 for i in range(26)],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=10e6,
        snr_floor_db=7.0,
        modulation="OFDM-FHSS (SkyLink)",
        fhss=True,
        hop_interval_ms=10.0,
        description="Autel EVO / EVO II / EVO Nano — SkyLink protocol",
    ),
    "skydio": DroneSignature(
        key="skydio",
        name="Skydio 2 / X2",
        brand="Skydio",
        freqs_24ghz_mhz=_WIFI_2G,
        freqs_58ghz_mhz=_WIFI_5G,
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=80e6,
        snr_floor_db=10.0,
        modulation="WiFi 802.11ax (Wi-Fi 6)",
        fhss=False,
        hop_interval_ms=0,
        description="Skydio 2 / X2 — controlled via Skydio SDK (Python)",
        wifi_ip="192.168.77.1",
        wifi_port=50051,
        adapter="generic_udp",
    ),
    "mavlink_915": DroneSignature(
        key="mavlink_915",
        name="MAVLink 915 MHz Telemetry",
        brand="Generic",
        freqs_24ghz_mhz=[],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=_MAV_915,
        freqs_433mhz_mhz=[],
        bandwidth_hz=500e3,
        snr_floor_db=5.0,
        modulation="FHSS-FSK (SiK radio)",
        fhss=True,
        hop_interval_ms=20.0,
        description="ArduPilot / PX4 with SiK 915 MHz telemetry radio",
        adapter="mavlink",
    ),
    "mavlink_433": DroneSignature(
        key="mavlink_433",
        name="MAVLink 433 MHz Telemetry",
        brand="Generic",
        freqs_24ghz_mhz=[],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=_MAV_433,
        bandwidth_hz=500e3,
        snr_floor_db=5.0,
        modulation="FHSS-FSK (SiK radio)",
        fhss=True,
        hop_interval_ms=20.0,
        description="ArduPilot / PX4 with SiK 433 MHz telemetry radio",
        adapter="mavlink",
    ),
    "remote_id_faa": DroneSignature(
        key="remote_id_faa",
        name="FAA/EASA Remote ID Broadcast",
        brand="Generic",
        freqs_24ghz_mhz=[2402, 2426, 2480, 2412, 2437, 2462],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=2e6,
        snr_floor_db=3.0,
        modulation="BLE 5.0 / WiFi NAN",
        fhss=False,
        hop_interval_ms=0,
        description="FAA/EASA standard Remote ID broadcast (any brand)",
    ),
    # ── DJI specific (newer) ──────────────────────────────────────────────────
    "dji_air2s": DroneSignature(
        key="dji_air2s",
        name="DJI Air 2S",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=_OCUSYNC_5G,
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=6.0,
        modulation="OcuSync 3",
        fhss=True,
        hop_interval_ms=6.0,
        description="DJI Air 2S — OcuSync 3, 1\" sensor",
        adapter="mavlink",
    ),
    "dji_mini4pro": DroneSignature(
        key="dji_mini4pro",
        name="DJI Mini 4 Pro",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=[5745, 5765, 5785, 5805, 5825],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=6.0,
        modulation="O4 (OcuSync 4)",
        fhss=True,
        hop_interval_ms=5.0,
        description="DJI Mini 4 Pro — O4, < 249 g, ActiveTrack 360",
        adapter="mavlink",
    ),
    "dji_avata2": DroneSignature(
        key="dji_avata2",
        name="DJI Avata 2",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=[5745, 5765, 5785, 5805, 5825],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=6.0,
        modulation="O4 (OcuSync 4)",
        fhss=True,
        hop_interval_ms=5.0,
        description="DJI Avata 2 — cinewhoop FPV, O4",
        adapter="mavlink",
    ),
    "dji_neo": DroneSignature(
        key="dji_neo",
        name="DJI Neo",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=10e6,
        snr_floor_db=6.0,
        modulation="O3 (lite)",
        fhss=True,
        hop_interval_ms=6.0,
        description="DJI Neo — ultra-light palm drone, autonomous AI modes",
        adapter="mavlink",
    ),
    "dji_fpv": DroneSignature(
        key="dji_fpv",
        name="DJI FPV Drone",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=_OCUSYNC_5G,
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=5.0,
        modulation="OcuSync 3 + O3 video",
        fhss=True,
        hop_interval_ms=6.0,
        description="DJI FPV Combo — O3 link, goggles + motion controller",
        adapter="mavlink",
    ),
    "dji_o3_video": DroneSignature(
        key="dji_o3_video",
        name="DJI O3 Air Unit (video downlink)",
        brand="DJI",
        freqs_24ghz_mhz=[],
        freqs_58ghz_mhz=[5745, 5765, 5785, 5805, 5825, 5660, 5695, 5735, 5770],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=5.0,
        modulation="OFDM video (O3)",
        fhss=True,
        hop_interval_ms=10.0,
        description="DJI O3 Air Unit on FPV builds — 5.8GHz digital video downlink",
    ),
    "dji_matrice30": DroneSignature(
        key="dji_matrice30",
        name="DJI Matrice 30T (M30T)",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=_OCUSYNC_5G,
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=6.0,
        modulation="OcuSync 3 Enterprise",
        fhss=True,
        hop_interval_ms=6.0,
        description="DJI M30T — compact enterprise, IP55, OSDK support",
        adapter="dji_osdk",
    ),
    "dji_agras": DroneSignature(
        key="dji_agras",
        name="DJI Agras T40/T30 (Agricultural)",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=_OCUSYNC_5G,
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=6.0,
        modulation="OcuSync 3 / DJI RC Plus",
        fhss=True,
        hop_interval_ms=6.0,
        description="DJI Agras agricultural sprayer — T30/T40, OSDK support",
        adapter="dji_osdk",
    ),
    # ── Autel ─────────────────────────────────────────────────────────────────
    "autel_nano": DroneSignature(
        key="autel_nano",
        name="Autel EVO Nano / Nano+",
        brand="Autel",
        freqs_24ghz_mhz=[2400.0 + i * 5 for i in range(18)],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=10e6,
        snr_floor_db=7.0,
        modulation="SkyLink (FHSS)",
        fhss=True,
        hop_interval_ms=10.0,
        description="Autel EVO Nano — 249 g, SkyLink, 2.4 GHz only",
    ),
    "autel_lite": DroneSignature(
        key="autel_lite",
        name="Autel EVO Lite / Lite+",
        brand="Autel",
        freqs_24ghz_mhz=[2400.0 + i * 5 for i in range(18)],
        freqs_58ghz_mhz=[5725.0 + i * 5 for i in range(26)],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=10e6,
        snr_floor_db=7.0,
        modulation="SkyLink (FHSS)",
        fhss=True,
        hop_interval_ms=10.0,
        description="Autel EVO Lite — 3-axis gimbal, SkyLink dual-band",
    ),
    "autel_evomax": DroneSignature(
        key="autel_evomax",
        name="Autel EVO Max 4T",
        brand="Autel",
        freqs_24ghz_mhz=[2400.0 + i * 5 for i in range(18)],
        freqs_58ghz_mhz=[5725.0 + i * 5 for i in range(26)],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=15e6,
        snr_floor_db=6.0,
        modulation="SkyLink 4 Enterprise (FHSS)",
        fhss=True,
        hop_interval_ms=8.0,
        description="Autel EVO Max 4T — enterprise, thermal + LiDAR, SkyLink 4",
    ),
    # ── Parrot ────────────────────────────────────────────────────────────────
    "parrot_anafi_usa": DroneSignature(
        key="parrot_anafi_usa",
        name="Parrot ANAFI USA",
        brand="Parrot",
        freqs_24ghz_mhz=[2412, 2437, 2462],
        freqs_58ghz_mhz=[5180, 5200, 5240, 5745, 5765, 5785, 5805, 5825],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=40e6,
        snr_floor_db=8.0,
        modulation="WiFi 802.11ac (encrypted)",
        fhss=False,
        hop_interval_ms=0,
        description="Parrot ANAFI USA — military / LE grade, AES-256, NDAA compliant",
        wifi_ssid_prefix="ANAFI-USA-",
        wifi_ip="192.168.42.1",
        wifi_port=9988,
        adapter="parrot",
    ),
    # ── Yuneec ────────────────────────────────────────────────────────────────
    "yuneec_typhoon_h3": DroneSignature(
        key="yuneec_typhoon_h3",
        name="Yuneec Typhoon H3",
        brand="Yuneec",
        freqs_24ghz_mhz=list(range(2400, 2484, 3)),
        freqs_58ghz_mhz=list(range(5725, 5850, 5)),
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=10e6,
        snr_floor_db=8.0,
        modulation="FHSS (ST24 + CGOP)",
        fhss=True,
        hop_interval_ms=12.0,
        description="Yuneec Typhoon H3 — ST24 radio, 360° rotating gimbal",
    ),
    "yuneec_h520e": DroneSignature(
        key="yuneec_h520e",
        name="Yuneec H520E",
        brand="Yuneec",
        freqs_24ghz_mhz=list(range(2400, 2484, 3)),
        freqs_58ghz_mhz=list(range(5725, 5850, 5)),
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=10e6,
        snr_floor_db=7.0,
        modulation="FHSS Enterprise",
        fhss=True,
        hop_interval_ms=12.0,
        description="Yuneec H520E — commercial hex-rotor, DataPilot GCS",
    ),
    # ── FIMI ──────────────────────────────────────────────────────────────────
    "fimi_x8se": DroneSignature(
        key="fimi_x8se",
        name="FIMI X8 SE 2022 / X8 Pro",
        brand="FIMI",
        freqs_24ghz_mhz=[2400.0 + i * 4 for i in range(21)],
        freqs_58ghz_mhz=[5725.0 + i * 10 for i in range(13)],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=10e6,
        snr_floor_db=7.0,
        modulation="FHSS (custom)",
        fhss=True,
        hop_interval_ms=10.0,
        description="FIMI X8 SE / X8 Pro — DJI-like performance at lower price",
        wifi_ssid_prefix="FIMI-",
        wifi_ip="192.168.11.1",
        wifi_port=80,
    ),
    "fimi_x8mini": DroneSignature(
        key="fimi_x8mini",
        name="FIMI X8 Mini V2",
        brand="FIMI",
        freqs_24ghz_mhz=[2400.0 + i * 4 for i in range(21)],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=8e6,
        snr_floor_db=7.0,
        modulation="FHSS 2.4GHz",
        fhss=True,
        hop_interval_ms=12.0,
        description="FIMI X8 Mini V2 — 249g, 4K, affordable compact",
    ),
    # ── Hubsan ────────────────────────────────────────────────────────────────
    "hubsan_zino2": DroneSignature(
        key="hubsan_zino2",
        name="Hubsan Zino 2 / Zino Mini Pro",
        brand="Hubsan",
        freqs_24ghz_mhz=_WIFI_2G,
        freqs_58ghz_mhz=[5745, 5765, 5785, 5805, 5825],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=8.0,
        modulation="WiFi 5 GHz (FHSS-like)",
        fhss=True,
        hop_interval_ms=15.0,
        description="Hubsan Zino 2 / Zino Mini Pro — 4K, foldable",
        wifi_ssid_prefix="HUBSAN-",
        wifi_ip="192.168.10.1",
        wifi_port=8888,
    ),
    # ── Generic consumer WiFi drones ──────────────────────────────────────────
    "holy_stone": DroneSignature(
        key="holy_stone",
        name="Holy Stone / Potensic (Generic WiFi)",
        brand="Generic",
        freqs_24ghz_mhz=_WIFI_2G,
        freqs_58ghz_mhz=[5180, 5200, 5745, 5765, 5785, 5805, 5825],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=8.0,
        modulation="WiFi 802.11n/ac",
        fhss=False,
        hop_interval_ms=0,
        description="Holy Stone HS720E, Potensic ATOM SE, and similar consumer WiFi drones",
        wifi_ssid_prefix="HOLY-",
        wifi_ip="192.168.1.1",
        wifi_port=80,
    ),
    "powervision_egg": DroneSignature(
        key="powervision_egg",
        name="PowerVision PowerEgg X",
        brand="PowerVision",
        freqs_24ghz_mhz=_WIFI_2G,
        freqs_58ghz_mhz=[5745, 5765, 5785, 5805, 5825],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=8.0,
        modulation="WiFi 5 GHz",
        fhss=False,
        hop_interval_ms=0,
        description="PowerVision PowerEgg X — omnidirectional, amphibious",
        wifi_ssid_prefix="PW-",
        wifi_ip="192.168.10.1",
        wifi_port=8080,
    ),
    # ── FPV / Racing ──────────────────────────────────────────────────────────
    "elrs_2g": DroneSignature(
        key="elrs_2g",
        name="ExpressLRS 2.4 GHz (ELRS)",
        brand="Open Source",
        freqs_24ghz_mhz=[2400.5 + i * 0.5 for i in range(80)],  # ELRS uses 80 frequencies
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=500e3,
        snr_floor_db=5.0,
        modulation="LoRa / FHSS (ELRS)",
        fhss=True,
        hop_interval_ms=4.0,
        description="ExpressLRS 2.4GHz — open-source RC link, very popular with FPV quads",
    ),
    "elrs_900": DroneSignature(
        key="elrs_900",
        name="ExpressLRS 915 MHz (ELRS)",
        brand="Open Source",
        freqs_24ghz_mhz=[],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[902 + i * 0.3 for i in range(87)],
        freqs_433mhz_mhz=[],
        bandwidth_hz=500e3,
        snr_floor_db=4.0,
        modulation="LoRa / FHSS (ELRS)",
        fhss=True,
        hop_interval_ms=10.0,
        description="ExpressLRS 915MHz — long-range RC link for FPV and fixed-wing",
    ),
    "tbs_crossfire": DroneSignature(
        key="tbs_crossfire",
        name="TBS Crossfire",
        brand="Team BlackSheep",
        freqs_24ghz_mhz=[],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[869.0, 915.0, 921.5],
        freqs_433mhz_mhz=[],
        bandwidth_hz=500e3,
        snr_floor_db=3.0,
        modulation="LoRa FHSS",
        fhss=True,
        hop_interval_ms=10.0,
        description="TBS Crossfire — 900 MHz long-range RC link, 150+ km range claimed",
    ),
    "analog_fpv_58": DroneSignature(
        key="analog_fpv_58",
        name="Analog FPV Video (5.8 GHz)",
        brand="Generic FPV",
        freqs_24ghz_mhz=[],
        freqs_58ghz_mhz=[
            # Boscam A/B/E, RaceBand, DJI analog channels
            5725, 5733, 5740, 5745, 5752, 5760, 5765, 5769, 5771,
            5780, 5785, 5790, 5800, 5805, 5806, 5809, 5820, 5825,
            5828, 5840, 5843, 5847, 5860, 5865, 5866,
            5880, 5885, 5905, 5925, 5945,
            5658, 5665, 5685, 5695, 5705,
        ],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=18e6,
        snr_floor_db=6.0,
        modulation="Analog AM/FM video (NTSC/PAL)",
        fhss=False,
        hop_interval_ms=0,
        description="Analog FPV video transmitter — 5.8 GHz VTX on racing/freestyle quads",
    ),
    "walksnail_hdzero": DroneSignature(
        key="walksnail_hdzero",
        name="Walksnail Avatar / HDZero (Digital FPV)",
        brand="Digital FPV",
        freqs_24ghz_mhz=[],
        freqs_58ghz_mhz=[5658, 5695, 5732, 5770, 5805, 5839, 5878,
                         5725, 5745, 5765, 5785, 5805, 5825],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=7.0,
        modulation="OFDM digital video",
        fhss=False,
        hop_interval_ms=0,
        description="Walksnail Avatar HD or HDZero digital FPV system — 5.8 GHz HD video",
    ),
    # ── Enterprise / Delivery ─────────────────────────────────────────────────
    "wingcopter": DroneSignature(
        key="wingcopter",
        name="Wingcopter 178 / 198",
        brand="Wingcopter",
        freqs_24ghz_mhz=[],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=_MAV_915,
        freqs_433mhz_mhz=[],
        bandwidth_hz=500e3,
        snr_floor_db=5.0,
        modulation="MAVLink + LTE backup",
        fhss=True,
        hop_interval_ms=20.0,
        description="Wingcopter VTOL delivery drone — MAVLink, used for medical delivery",
        adapter="mavlink",
    ),
    "sensefly_ebee": DroneSignature(
        key="sensefly_ebee",
        name="senseFly eBee X (fixed-wing)",
        brand="senseFly / AgEagle",
        freqs_24ghz_mhz=list(range(2412, 2472, 5)),
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[902, 906, 915, 918],
        freqs_433mhz_mhz=[],
        bandwidth_hz=2e6,
        snr_floor_db=5.0,
        modulation="FHSS + WiFi",
        fhss=True,
        hop_interval_ms=25.0,
        description="senseFly eBee X — fixed-wing survey drone, RTK GPS",
    ),
    "skydio3": DroneSignature(
        key="skydio3",
        name="Skydio 3",
        brand="Skydio",
        freqs_24ghz_mhz=_WIFI_2G,
        freqs_58ghz_mhz=_WIFI_5G + [5935, 5955, 5975, 5995],  # Wi-Fi 6E 6GHz
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=80e6,
        snr_floor_db=10.0,
        modulation="Wi-Fi 6E (802.11ax, 6 GHz)",
        fhss=False,
        hop_interval_ms=0,
        description="Skydio 3 — Wi-Fi 6E, autonomous AI tracking, 6 GHz band",
        wifi_ip="192.168.77.1",
        wifi_port=50051,
        adapter="generic_udp",
    ),
    "xag_p100": DroneSignature(
        key="xag_p100",
        name="XAG P100 Pro (Agricultural)",
        brand="XAG",
        freqs_24ghz_mhz=[2400.0 + i * 5 for i in range(17)],
        freqs_58ghz_mhz=[5725.0 + i * 5 for i in range(25)],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=10e6,
        snr_floor_db=7.0,
        modulation="FHSS proprietary",
        fhss=True,
        hop_interval_ms=10.0,
        description="XAG P100 Pro — agricultural sprayer, RTK, autonomous field ops",
    ),

    # ── DJI newer O3+ models ──────────────────────────────────────────────────
    "dji_mini3pro": DroneSignature(
        key="dji_mini3pro",
        name="DJI Mini 3 Pro",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=_OCUSYNC_5G,
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=9.0,
        modulation="OcuSync 3 (O3) FHSS OFDM",
        fhss=True,
        hop_interval_ms=4.0,
        description="DJI Mini 3 Pro — 249 g, O3 dual-band, 47 min flight, obstacle avoidance",
        wifi_ssid_prefix="Mavic-",
        adapter="dji_osdk",
    ),
    "dji_mavic3": DroneSignature(
        key="dji_mavic3",
        name="DJI Mavic 3 / 3 Classic",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=_OCUSYNC_5G,
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=9.0,
        modulation="OcuSync 3+ (O3+) FHSS OFDM",
        fhss=True,
        hop_interval_ms=4.0,
        description="DJI Mavic 3 / 3 Classic — 4/3 CMOS Hasselblad, 46 min, O3+ 15 km range",
        wifi_ssid_prefix="Mavic-",
        adapter="dji_osdk",
    ),
    "dji_mavic3e": DroneSignature(
        key="dji_mavic3e",
        name="DJI Mavic 3 Enterprise / 3E",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=_OCUSYNC_5G,
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=9.0,
        modulation="OcuSync 3+ (O3+) FHSS OFDM",
        fhss=True,
        hop_interval_ms=4.0,
        description="DJI Mavic 3 Enterprise — RTK option, thermal/zoom payloads, DJI RC Pro controller",
        adapter="dji_osdk",
    ),
    "dji_inspire3": DroneSignature(
        key="dji_inspire3",
        name="DJI Inspire 3",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=_OCUSYNC_5G,
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=9.0,
        modulation="OcuSync 3+ (O3+) FHSS OFDM",
        fhss=True,
        hop_interval_ms=4.0,
        description="DJI Inspire 3 — cinema platform, Zenmuse X9-8K, dual operator, 28 km range",
        adapter="dji_osdk",
    ),
    "dji_m350": DroneSignature(
        key="dji_m350",
        name="DJI Matrice 350 RTK",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=_OCUSYNC_5G,
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=8.0,
        modulation="OcuSync 3+ Enterprise FHSS OFDM",
        fhss=True,
        hop_interval_ms=4.0,
        description="DJI M350 RTK — enterprise workhorse, 55 min, 20 km, OSDK compatible",
        adapter="dji_osdk",
    ),
    "dji_m30t": DroneSignature(
        key="dji_m30t",
        name="DJI Matrice 30T",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=_OCUSYNC_5G,
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=8.0,
        modulation="OcuSync Enterprise FHSS OFDM",
        fhss=True,
        hop_interval_ms=4.0,
        description="DJI M30T — IP55, thermal+zoom+wide+laser, dock-compatible, OSDK",
        adapter="dji_osdk",
    ),
    "dji_avata_v1": DroneSignature(
        key="dji_avata_v1",
        name="DJI Avata (Gen 1)",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=_OCUSYNC_5G,
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=10.0,
        modulation="OcuSync 3 FHSS OFDM",
        fhss=True,
        hop_interval_ms=4.0,
        description="DJI Avata Gen 1 — cinewhoop FPV, 410 g, Motion Controller / Goggles 2",
    ),
    "dji_o4_video": DroneSignature(
        key="dji_o4_video",
        name="DJI O4 Video Link",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=_OCUSYNC_5G,
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=9.0,
        modulation="OcuSync 4 (O4) FHSS OFDM",
        fhss=True,
        hop_interval_ms=3.0,
        description="DJI O4 video transmission — used in Avata 2, Neo, Mini 4 Pro, 20 km range",
    ),

    # ── Autel expanded ────────────────────────────────────────────────────────
    "autel_evo_lite": DroneSignature(
        key="autel_evo_lite",
        name="Autel EVO Lite / Lite+",
        brand="Autel Robotics",
        freqs_24ghz_mhz=[2402.0 + i * 4 for i in range(20)],
        freqs_58ghz_mhz=[5745.0 + i * 5 for i in range(17)],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=10e6,
        snr_floor_db=9.0,
        modulation="SkyLink 2.0 FHSS",
        fhss=True,
        hop_interval_ms=8.0,
        description="Autel EVO Lite+ — 249 g, 6K CMOS, SkyLink 2.0, 40 min",
        wifi_ssid_prefix="AUTEL-",
        wifi_ip=_IP_AUTEL,
        wifi_port=80,
        adapter="generic_udp",
    ),
    "autel_dragonfish": DroneSignature(
        key="autel_dragonfish",
        name="Autel DragonFish",
        brand="Autel Robotics",
        freqs_24ghz_mhz=[2402.0 + i * 4 for i in range(20)],
        freqs_58ghz_mhz=[5745.0 + i * 5 for i in range(17)],
        freqs_900mhz_mhz=[915.0 + i * 0.5 for i in range(10)],
        freqs_433mhz_mhz=[],
        bandwidth_hz=10e6,
        snr_floor_db=8.0,
        modulation="SkyLink FHSS + MAVLink 915",
        fhss=True,
        hop_interval_ms=8.0,
        description="Autel DragonFish — fixed-wing VTOL, 120 min, enterprise survey/inspection",
        adapter="mavlink",
    ),

    # ── Skydio expanded ───────────────────────────────────────────────────────
    "skydio_x10": DroneSignature(
        key="skydio_x10",
        name="Skydio X10",
        brand="Skydio",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[5180.0 + i * 20 for i in range(25)],
        # WiFi 6E adds 6 GHz — approximate channel list
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=40e6,
        snr_floor_db=8.0,
        modulation="WiFi 6 / 6E (802.11ax) OFDMA",
        fhss=False,
        hop_interval_ms=0.0,
        description="Skydio X10 — AI autonomy, WiFi 6E (2.4/5/6 GHz), dock-compatible, thermal option",
        wifi_ssid_prefix="Skydio-",
        wifi_ip=_IP_SKYDIO,
        wifi_port=50051,
        adapter="generic_udp",
    ),

    # ── Delivery drones ───────────────────────────────────────────────────────
    "zipline_p2": DroneSignature(
        key="zipline_p2",
        name="Zipline P2 Zip",
        brand="Zipline",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[5745.0 + i * 5 for i in range(17)],
        freqs_900mhz_mhz=[902.0 + i * 1 for i in range(26)],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=7.0,
        modulation="Proprietary FHSS + LTE backup",
        fhss=True,
        hop_interval_ms=10.0,
        description="Zipline P2 Zip — medical/package delivery, 40 km range, winged fixed-body",
    ),
    "wing_delivery": DroneSignature(
        key="wing_delivery",
        name="Wing Delivery Drone (Alphabet)",
        brand="Wing",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[5745.0 + i * 5 for i in range(17)],
        freqs_900mhz_mhz=[902.0 + i * 1 for i in range(26)],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=7.0,
        modulation="Proprietary + LTE",
        fhss=True,
        hop_interval_ms=10.0,
        description="Wing (Alphabet) delivery drone — tether winch, 12 kg MTOW, suburban delivery",
    ),
    "amazon_mk30": DroneSignature(
        key="amazon_mk30",
        name="Amazon Prime Air MK30",
        brand="Amazon",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[5745.0 + i * 5 for i in range(17)],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=7.0,
        modulation="Proprietary + LTE",
        fhss=True,
        hop_interval_ms=10.0,
        description="Amazon MK30 — Prime Air delivery drone, hexagonal design, 2.7 kg payload",
    ),
    "matternet_m2": DroneSignature(
        key="matternet_m2",
        name="Matternet M2",
        brand="Matternet",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[902.0 + i * 1 for i in range(26)],
        freqs_433mhz_mhz=[],
        bandwidth_hz=10e6,
        snr_floor_db=7.0,
        modulation="Proprietary FHSS + LTE",
        fhss=True,
        hop_interval_ms=10.0,
        description="Matternet M2 — medical lab sample delivery, 2 kg payload, BVLOS certified",
    ),
    "flytrex": DroneSignature(
        key="flytrex",
        name="Flytrex Delivery Drone",
        brand="Flytrex",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[5745.0 + i * 5 for i in range(17)],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=10e6,
        snr_floor_db=7.0,
        modulation="Proprietary + LTE",
        fhss=False,
        hop_interval_ms=0.0,
        description="Flytrex delivery drone — 6.5 kg payload, last-mile delivery, US suburban",
    ),

    # ── Public safety / security ─────────────────────────────────────────────
    "brinc_lemur2": DroneSignature(
        key="brinc_lemur2",
        name="BRINC Lemur 2",
        brand="BRINC",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[5745.0 + i * 5 for i in range(17)],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=9.0,
        modulation="WiFi 5 (802.11ac) OFDM",
        fhss=False,
        hop_interval_ms=0.0,
        description="BRINC Lemur 2 — tactical indoor/first-responder, armored, two-way audio, LTE",
    ),
    "axon_air": DroneSignature(
        key="axon_air",
        name="Axon Air R7",
        brand="Axon",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[5745.0 + i * 5 for i in range(17)],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=9.0,
        modulation="WiFi 5 / LTE OFDM",
        fhss=False,
        hop_interval_ms=0.0,
        description="Axon Air R7 — drone-as-first-responder (DFR), police/EMS, Taser integration option",
    ),
    "percepto_sparrow": DroneSignature(
        key="percepto_sparrow",
        name="Percepto Sparrow",
        brand="Percepto",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[5745.0 + i * 5 for i in range(17)],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=8.0,
        modulation="WiFi 5 / LTE OFDM",
        fhss=False,
        hop_interval_ms=0.0,
        description="Percepto Sparrow — autonomous security inspection, dock-based, 24/7 ops",
    ),
    "draganflyer": DroneSignature(
        key="draganflyer",
        name="Draganflyer Commander",
        brand="Draganfly",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[5745.0 + i * 5 for i in range(17)],
        freqs_900mhz_mhz=[902.0 + i * 1 for i in range(26)],
        freqs_433mhz_mhz=[],
        bandwidth_hz=10e6,
        snr_floor_db=8.0,
        modulation="MAVLink 900 MHz + WiFi",
        fhss=True,
        hop_interval_ms=20.0,
        description="Draganfly Commander — emergency services, public safety, 900 MHz MAVLink",
        adapter="mavlink",
    ),

    # ── Fixed wing / VTOL survey ──────────────────────────────────────────────
    "wingtra_gen2": DroneSignature(
        key="wingtra_gen2",
        name="WingtraOne Gen II",
        brand="Wingtra",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[902.0 + i * 1 for i in range(26)],
        freqs_433mhz_mhz=[],
        bandwidth_hz=1e6,
        snr_floor_db=8.0,
        modulation="MAVLink 915 MHz SiK + WiFi GCS",
        fhss=True,
        hop_interval_ms=20.0,
        description="WingtraOne Gen II — VTOL fixed-wing, PPK/RTK survey, 59 min, 3 kg payload",
        adapter="mavlink",
    ),
    "quantum_trinity": DroneSignature(
        key="quantum_trinity",
        name="Quantum Systems Trinity F90+",
        brand="Quantum Systems",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[902.0 + i * 1 for i in range(26)],
        freqs_433mhz_mhz=[433.0 + i * 0.5 for i in range(5)],
        bandwidth_hz=1e6,
        snr_floor_db=8.0,
        modulation="MAVLink 915/433 MHz SiK",
        fhss=True,
        hop_interval_ms=20.0,
        description="Trinity F90+ — VTOL fixed-wing, 90 min, 1.2 kg payload, enterprise survey",
        adapter="mavlink",
    ),
    "jouav_cw25": DroneSignature(
        key="jouav_cw25",
        name="JOUAV CW-25E",
        brand="JOUAV",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[5745.0 + i * 5 for i in range(17)],
        freqs_900mhz_mhz=[902.0 + i * 1 for i in range(26)],
        freqs_433mhz_mhz=[],
        bandwidth_hz=10e6,
        snr_floor_db=8.0,
        modulation="MAVLink + proprietary video 5.8 GHz",
        fhss=True,
        hop_interval_ms=15.0,
        description="JOUAV CW-25E — VTOL fixed-wing, 480 min endurance, 2.5 kg payload, China",
        adapter="mavlink",
    ),
    "delair_dx8": DroneSignature(
        key="delair_dx8",
        name="Delair DX8",
        brand="Delair",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[902.0 + i * 1 for i in range(26)],
        freqs_433mhz_mhz=[],
        bandwidth_hz=1e6,
        snr_floor_db=8.0,
        modulation="MAVLink 915 MHz + LTE",
        fhss=True,
        hop_interval_ms=20.0,
        description="Delair DX8 — fixed-wing BVLOS, 2 hr, 4G LTE, PPK mapping",
        adapter="mavlink",
    ),
    "aerov_raven": DroneSignature(
        key="aerov_raven",
        name="AeroVironment RQ-11 Raven",
        brand="AeroVironment",
        freqs_24ghz_mhz=[],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[902.0 + i * 1 for i in range(26)],
        freqs_433mhz_mhz=[433.0 + i * 0.5 for i in range(5)],
        bandwidth_hz=1e6,
        snr_floor_db=7.0,
        modulation="Proprietary 900/433 MHz FHSS",
        fhss=True,
        hop_interval_ms=15.0,
        description="AeroVironment Raven RQ-11 — small military/LE UAS, hand-launched, 60 min",
    ),
    "aerov_puma": DroneSignature(
        key="aerov_puma",
        name="AeroVironment Puma AE",
        brand="AeroVironment",
        freqs_24ghz_mhz=[],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[902.0 + i * 1 for i in range(26)],
        freqs_433mhz_mhz=[433.0 + i * 0.5 for i in range(5)],
        bandwidth_hz=1e6,
        snr_floor_db=7.0,
        modulation="Proprietary 900/433 MHz FHSS",
        fhss=True,
        hop_interval_ms=15.0,
        description="AeroVironment Puma AE — catapult or hand-launched, 3.2 kg, 3+ hr endurance",
    ),

    # ── FPV / RC link ecosystem expansion ─────────────────────────────────────
    "ghost_immersionrc": DroneSignature(
        key="ghost_immersionrc",
        name="ImmersionRC Ghost",
        brand="ImmersionRC",
        freqs_24ghz_mhz=[2400.0 + i * 1 for i in range(83)],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=0.9e6,
        snr_floor_db=8.0,
        modulation="FHSS 2.4 GHz 900 kHz channels",
        fhss=True,
        hop_interval_ms=8.0,
        description="ImmersionRC Ghost — 2.4 GHz RC link, 250 mW, 10 km LOS, FPV racing/freestyle",
    ),
    "frsky_r9": DroneSignature(
        key="frsky_r9",
        name="FrSky R9 / R9M",
        brand="FrSky",
        freqs_24ghz_mhz=[],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[868.0 + i * 1 for i in range(27)],  # 868 EU + 915 US overlap
        freqs_433mhz_mhz=[],
        bandwidth_hz=0.5e6,
        snr_floor_db=8.0,
        modulation="FrSky ACCESS 900 MHz FHSS",
        fhss=True,
        hop_interval_ms=50.0,
        description="FrSky R9/R9M — 900 MHz long-range RC link, 8 ch FHSS, EU 868/US 915 MHz",
    ),
    "frsky_tracer": DroneSignature(
        key="frsky_tracer",
        name="FrSky TRACER",
        brand="FrSky",
        freqs_24ghz_mhz=[2400.0 + i * 1 for i in range(83)],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=0.5e6,
        snr_floor_db=8.0,
        modulation="FrSky ARCHER 2.4 GHz FHSS",
        fhss=True,
        hop_interval_ms=40.0,
        description="FrSky TRACER — 2.4 GHz 250 mW, 32ch ACCST, racing/sport long range",
    ),
    "elrs_868": DroneSignature(
        key="elrs_868",
        name="ExpressLRS 868 MHz (EU)",
        brand="ExpressLRS",
        freqs_24ghz_mhz=[],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[863.0 + i * 0.5 for i in range(30)],
        freqs_433mhz_mhz=[],
        bandwidth_hz=0.5e6,
        snr_floor_db=6.0,
        modulation="LoRa FHSS 868 MHz",
        fhss=True,
        hop_interval_ms=4.0,
        description="ExpressLRS 868 MHz — EU variant, LoRa modulation, 30+ km, low latency",
    ),
    "crsf_nano": DroneSignature(
        key="crsf_nano",
        name="TBS Crossfire Nano / Micro",
        brand="Team BlackSheep",
        freqs_24ghz_mhz=[],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[902.0 + i * 1 for i in range(26)],
        freqs_433mhz_mhz=[],
        bandwidth_hz=0.4e6,
        snr_floor_db=6.0,
        modulation="FHSS 915 MHz 400 kHz BW",
        fhss=True,
        hop_interval_ms=4.0,
        description="TBS Crossfire Nano/Micro — compact RC link, 40 km tested LOS, 868/915 MHz",
    ),

    # ── Agricultural expanded ─────────────────────────────────────────────────
    "dji_t40": DroneSignature(
        key="dji_t40",
        name="DJI Agras T40",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=_OCUSYNC_5G,
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=8.0,
        modulation="OcuSync Enterprise FHSS OFDM",
        fhss=True,
        hop_interval_ms=4.0,
        description="DJI Agras T40 — 40 L tank, 50 kg MTOW, phased-array radar, RTK",
        adapter="dji_osdk",
    ),
    "dji_t60": DroneSignature(
        key="dji_t60",
        name="DJI Agras T60",
        brand="DJI",
        freqs_24ghz_mhz=_OCUSYNC_2G,
        freqs_58ghz_mhz=_OCUSYNC_5G,
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=8.0,
        modulation="OcuSync Enterprise FHSS OFDM",
        fhss=True,
        hop_interval_ms=4.0,
        description="DJI Agras T60 — 60 L tank, 101 kg MTOW, dual spreading system, RTK",
        adapter="dji_osdk",
    ),

    # ── eVTOL / Urban Air Mobility (detect by LTE absence + prop sound ───────
    "ehang_216": DroneSignature(
        key="ehang_216",
        name="EHang 216 / EHang 216S",
        brand="EHang",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=7.0,
        modulation="4G LTE + WiFi backup",
        fhss=False,
        hop_interval_ms=0.0,
        description="EHang 216S — 2-seat AAV, 4G LTE primary, WiFi local backup, FAA G-1 issued",
    ),

    # ── Inspection / Enterprise ───────────────────────────────────────────────
    "freefly_altax": DroneSignature(
        key="freefly_altax",
        name="Freefly Alta X",
        brand="Freefly Systems",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[5745.0 + i * 5 for i in range(17)],
        freqs_900mhz_mhz=[902.0 + i * 1 for i in range(26)],
        freqs_433mhz_mhz=[],
        bandwidth_hz=10e6,
        snr_floor_db=8.0,
        modulation="MAVLink 915 MHz + WiFi",
        fhss=True,
        hop_interval_ms=20.0,
        description="Freefly Alta X — 35 kg payload octocopter, Herelink or SiK MAVLink telemetry",
        adapter="mavlink",
    ),
    "inspired_flight": DroneSignature(
        key="inspired_flight",
        name="Inspired Flight IF1200A",
        brand="Inspired Flight",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[5745.0 + i * 5 for i in range(17)],
        freqs_900mhz_mhz=[902.0 + i * 1 for i in range(26)],
        freqs_433mhz_mhz=[],
        bandwidth_hz=10e6,
        snr_floor_db=8.0,
        modulation="MAVLink 915 MHz + WiFi",
        fhss=True,
        hop_interval_ms=20.0,
        description="Inspired Flight IF1200A — US-made hex, 6 kg payload, MAVLink ArduCopter",
        adapter="mavlink",
    ),
    "parrot_anafi_ai": DroneSignature(
        key="parrot_anafi_ai",
        name="Parrot ANAFI Ai",
        brand="Parrot",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[5180.0 + i * 20 for i in range(25)],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=40e6,
        snr_floor_db=8.0,
        modulation="WiFi 6 (802.11ax) 4G LTE",
        fhss=False,
        hop_interval_ms=0.0,
        description="Parrot ANAFI Ai — 4G LTE, WiFi 6, 32x zoom, Europe-made, NDAA compliant",
        wifi_ssid_prefix="ANAFI-",
        wifi_ip=_IP_PARROT,
        wifi_port=9988,
        adapter="parrot",
    ),
    "lte_bvlos": DroneSignature(
        key="lte_bvlos",
        name="LTE-Connected BVLOS Drone",
        brand="Various",
        freqs_24ghz_mhz=[2412.0 + i * 5 for i in range(14)],
        freqs_58ghz_mhz=[],
        freqs_900mhz_mhz=[],
        freqs_433mhz_mhz=[],
        bandwidth_hz=20e6,
        snr_floor_db=5.0,
        modulation="4G LTE + WiFi local fallback",
        fhss=False,
        hop_interval_ms=0.0,
        description="Generic LTE-connected BVLOS platform — Herelink, Verizon Skyward, AT&T. "
                    "LTE uplink invisible to SDR; local WiFi fallback may be detectable.",
    ),
}

# =============================================================================
# DETECTION / FINGERPRINT DATA STRUCTURES
# =============================================================================

@dataclass
class DroneMatch:
    sig_key: str
    name: str
    brand: str
    confidence: float        # 0.0–1.0
    center_freq_hz: float
    detected_bw_hz: float
    snr_db: float
    peak_db: float
    timestamp_utc: str
    notes: str
    adapter: str = ""
    wifi_ip: str = ""
    wifi_port: int = 0

@dataclass
class ScanResult:
    timestamp_utc: str
    center_freq_hz: float
    peak_db: float
    noise_floor_db: float
    snr_db: float
    occupied_bw_hz: float
    detected: bool
    freq_axis_hz: np.ndarray
    spectrum_db: np.ndarray
    sample_rate: float = 0.0
    frontend_bandwidth_hz: float = 0.0
    recording_path: str = ""
    matches: list = field(default_factory=list)   # list[DroneMatch]

# Hop tracker: remembers which signatures were seen at which frequencies recently
@dataclass
class HopEvent:
    sig_key: str
    freq_hz: float
    timestamp: float        # monotonic

# =============================================================================
# SIGNAL FINGERPRINTING ENGINE
# =============================================================================

def fingerprint_signal(
    center_freq_hz: float,
    detected_bw_hz: float,
    snr_db: float,
    peak_db: float,
) -> list[DroneMatch]:
    """
    Score all drone signatures against a detected signal.
    Returns up to 3 matches sorted by descending confidence.
    """
    freq_mhz = center_freq_hz / 1e6
    matches = []

    for key, sig in DRONE_SIGNATURES.items():
        all_freqs = (
            sig.freqs_24ghz_mhz
            + sig.freqs_58ghz_mhz
            + sig.freqs_900mhz_mhz
            + sig.freqs_433mhz_mhz
        )
        if not all_freqs:
            continue

        # Distance to nearest known channel center (MHz)
        min_dist_mhz = min(abs(freq_mhz - f) for f in all_freqs)
        half_bw_mhz = sig.bandwidth_hz / 2e6

        # Gate: must be within one BW of a known channel
        if min_dist_mhz > half_bw_mhz * 1.5:
            continue

        confidence = 0.0
        notes = []

        # Frequency match score (0–0.50)
        freq_score = max(0.0, 1.0 - min_dist_mhz / max(half_bw_mhz, 0.1))
        confidence += freq_score * 0.50
        notes.append(f"freq_dist={min_dist_mhz:.1f}MHz")

        # Bandwidth match score (0–0.30)
        if detected_bw_hz > 0:
            ratio = detected_bw_hz / max(sig.bandwidth_hz, 1.0)
            bw_score = max(0.0, 1.0 - abs(np.log10(max(ratio, 1e-3))))
            confidence += min(bw_score, 1.0) * 0.30
            notes.append(f"bw_ratio={ratio:.2f}")

        # SNR score (0–0.20)
        if snr_db >= sig.snr_floor_db:
            snr_margin = min((snr_db - sig.snr_floor_db) / 15.0, 1.0)
            confidence += snr_margin * 0.20
            notes.append(f"SNR={snr_db:.1f}dB")

        if confidence < 0.20:
            continue

        matches.append(DroneMatch(
            sig_key=key,
            name=sig.name,
            brand=sig.brand,
            confidence=round(confidence, 3),
            center_freq_hz=center_freq_hz,
            detected_bw_hz=detected_bw_hz,
            snr_db=snr_db,
            peak_db=peak_db,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            notes="; ".join(notes),
            adapter=sig.adapter,
            wifi_ip=sig.wifi_ip,
            wifi_port=sig.wifi_port,
        ))

    matches.sort(key=lambda m: m.confidence, reverse=True)
    return matches[:3]


def score_hop_confidence(hop_events: list[HopEvent], sig_key: str, window_s: float = 5.0) -> float:
    """
    Extra confidence boost when we see the same signature at multiple
    different frequencies in a short time window (consistent with FHSS).
    """
    now = time.monotonic()
    recent = [e for e in hop_events if e.sig_key == sig_key and now - e.timestamp < window_s]
    unique_freqs = len({round(e.freq_hz / 1e6, 0) for e in recent})
    if unique_freqs >= 3:
        return 0.25
    if unique_freqs == 2:
        return 0.10
    return 0.0

# =============================================================================
# OCUSYNC TOOLS
# =============================================================================

# All 36 OcuSync 2.4 GHz channel centers (MHz)
OCUSYNC_CHANNELS_2G = [round(2400.5 + i * 2.3, 1) for i in range(36)]
# OcuSync 5.8 GHz channels
OCUSYNC_CHANNELS_5G = [5730.0, 5750.0, 5770.0, 5790.0, 5810.0, 5830.0]

_BAND_24GHZ = "2.4GHz"
_BAND_58GHZ = "5.8GHz"


def build_ocusync_channel_map(recent_results: list, band: str = _BAND_24GHZ) -> go.Figure:
    """
    Bar chart showing per-channel power across all OcuSync channels.
    Peaks in the right channel grid confirm OcuSync vs generic WiFi.
    """
    channels = OCUSYNC_CHANNELS_2G if band == _BAND_24GHZ else OCUSYNC_CHANNELS_5G
    half_ch_mhz = 1.15 if band == _BAND_24GHZ else 10.0

    power = [-140.0] * len(channels)
    for r in recent_results[-300:]:
        freq_mhz = r.center_freq_hz / 1e6
        for i, ch in enumerate(channels):
            if abs(freq_mhz - ch) < half_ch_mhz:
                power[i] = max(power[i], r.peak_db)

    colors = ["#22c55e" if p > -100 else "#f59e0b" if p > -125 else "#1e293b" for p in power]
    labels = [f"Ch{i+1}\n{ch:.1f}" for i, ch in enumerate(channels)]
    relative = [max(p + 140.0, 0.0) for p in power]

    fig = go.Figure(go.Bar(
        x=labels, y=relative, marker_color=colors,
        hovertext=[f"{ch:.1f} MHz | {p:.1f} dBm" for ch, p in zip(channels, power)],
        hoverinfo="text",
    ))
    active = sum(1 for p in power if p > -100)
    fig.update_layout(
        title=f"OcuSync {band} Channel Occupancy — {active}/{len(channels)} channels active",
        height=280,
        paper_bgcolor="#07111a",
        plot_bgcolor="#0c1722",
        font={"color": "#cbd5e1", "size": 9},
        yaxis={"title": "Rel. power (dB above noise)", "color": "#cbd5e1", "gridcolor": "#1e293b"},
        xaxis={"tickangle": 45, "tickfont": {"size": 7}, "color": "#cbd5e1"},
        margin={"l": 50, "r": 10, "t": 40, "b": 80},
        showlegend=False,
    )
    return fig


# =============================================================================
# BLE REMOTE ID SCANNER
# Detects DJI DroneID and FAA/EASA OpenDroneID broadcasts over Bluetooth LE.
# Requires: pip install bleak
# DJI Mavic 3, Mini 3, Air 3, Mini 4 Pro, and older models with updated
# firmware all broadcast OpenDroneID. Older DJI models use a proprietary
# DroneID format embedded in the BLE advertisement manufacturer data.
# =============================================================================

_ODID_SERVICE_UUID = "0000fffa-0000-1000-8000-00805f9b34fb"
_DJI_COMPANY_IDS = {0x1285, 0x000A}   # DJI Technology Co. Ltd Bluetooth SIG IDs

_UA_TYPES = {
    0: "None", 1: "Aeroplane", 2: "Helicopter/Multirotor", 3: "Gyroplane",
    4: "Hybrid Lift", 5: "Ornithopter", 6: "Glider", 7: "Kite",
    8: "Free Balloon", 9: "Captive Balloon", 10: "Airship",
    11: "Free Fall / Parachute", 12: "Rocket", 13: "Tethered Aircraft",
    14: "Ground Obstacle", 15: "Other",
}


_ODID_ID_TYPES = ["None", "Serial (ANSI/CTA-2063-A)", "CAA Assigned", "UTM Assigned", "Session ID"]

def _odid_basic_id(data: bytes) -> dict:
    id_type = (data[1] >> 4) & 0x0F
    ua_type = data[1] & 0x0F
    return {
        "msg_type": "BasicID",
        "id_type": _ODID_ID_TYPES[id_type] if id_type < len(_ODID_ID_TYPES) else f"Res({id_type})",
        "ua_type": _UA_TYPES.get(ua_type, f"Unknown({ua_type})"),
        "uas_id": data[2:22].decode("ascii", errors="replace").rstrip("\x00"),
    }

def _odid_location(data: bytes) -> dict:
    if len(data) < 25:
        return {"msg_type": "Location", "note": "payload too short"}
    return {
        "msg_type": "Location",
        "direction_deg": round(int.from_bytes(data[2:4], "little") / 100.0, 1),
        "speed_h_ms": round(int.from_bytes(data[4:6], "little") / 100.0, 2),
        "lat": round(int.from_bytes(data[8:12], "little", signed=True) * 1e-7, 6),
        "lon": round(int.from_bytes(data[12:16], "little", signed=True) * 1e-7, 6),
        "alt_pressure_m": round(int.from_bytes(data[16:18], "little") / 2.0 - 1000.0, 1),
        "alt_geodetic_m": round(int.from_bytes(data[18:20], "little") / 2.0 - 1000.0, 1),
    }

def _odid_system(data: bytes) -> dict:
    result = {"msg_type": "System"}
    if len(data) >= 14:
        result["operator_lat"] = round(int.from_bytes(data[4:8], "little", signed=True) * 1e-7, 6)
        result["operator_lon"] = round(int.from_bytes(data[8:12], "little", signed=True) * 1e-7, 6)
    return result

def _odid_operator_id(data: bytes) -> dict:
    return {
        "msg_type": "OperatorID",
        "operator_id": data[2:23].decode("ascii", errors="replace").rstrip("\x00"),
    }

_ODID_DECODERS = {0: _odid_basic_id, 1: _odid_location, 4: _odid_system, 5: _odid_operator_id}

def decode_opendroneid(data: bytes) -> dict:
    """Decode one OpenDroneID message (ASTM F3411-22a / EN 4709-002)."""
    if len(data) < 2:
        return {}
    msg_type = (data[0] >> 4) & 0x0F
    result: dict = {"msg_type_id": msg_type}
    decoder = _ODID_DECODERS.get(msg_type)
    try:
        if decoder:
            result.update(decoder(data))
        else:
            result["msg_type"] = f"MsgType{msg_type}"
    except Exception as exc:
        result["decode_error"] = str(exc)
    return result


async def _ble_scan_async(duration: float) -> tuple[list, str]:
    try:
        from bleak import BleakScanner
    except ImportError:
        return [], "bleak not installed — run: pip install bleak"

    detected: list = []
    seen_addresses: set = set()   # empty set — {} would be a dict literal

    def on_device(device, adv):
        addr = device.address
        if addr in seen_addresses:
            return
        rssi = adv.rssi if hasattr(adv, "rssi") else getattr(device, "rssi", None)
        name = device.name or ""

        # OpenDroneID service UUID
        for uuid, data in (adv.service_data or {}).items():
            if "fffa" in str(uuid).lower():
                seen_addresses.add(addr)
                detected.append({
                    "type": "OpenDroneID",
                    "address": addr,
                    "name": name,
                    "rssi": rssi,
                    "raw_hex": data.hex(),
                    "decoded": decode_opendroneid(data),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                return

        # DJI proprietary manufacturer data
        for company_id, data in (adv.manufacturer_data or {}).items():
            if company_id in _DJI_COMPANY_IDS:
                seen_addresses.add(addr)
                detected.append({
                    "type": "DJI Proprietary",
                    "address": addr,
                    "name": name,
                    "rssi": rssi,
                    "raw_hex": data.hex(),
                    "decoded": {},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                return

        # Heuristic: DJI device name prefix
        name_upper = name.upper()
        dji_prefixes = ("DJI-", "MAVIC", "MINI-", "AIR-", "FPV-", "AVATA", "NEO-", "PHANTOM")
        if any(name_upper.startswith(p) for p in dji_prefixes):
            seen_addresses.add(addr)
            detected.append({
                "type": "DJI (by name)",
                "address": addr,
                "name": name,
                "rssi": rssi,
                "raw_hex": "",
                "decoded": {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    scanner = BleakScanner(detection_callback=on_device)
    await scanner.start()
    import asyncio
    await asyncio.sleep(duration)
    await scanner.stop()
    return detected, "OK"


def run_ble_scan(duration: float = 5.0) -> tuple[list, str]:
    """Thread-safe wrapper around the async BLE scan."""
    import asyncio
    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(_ble_scan_async(duration))
        loop.close()
        return result
    except Exception as exc:
        return [], str(exc)


# =============================================================================
# WIFI DRONE PROBE
# Checks known drone default IPs on the current network.
# Many consumer drones create their own WiFi AP with a fixed gateway IP.
# =============================================================================

WIFI_DRONE_FINGERPRINTS = [
    {"brand": "DJI",        "model": "Tello / Tello EDU",          "ip": _IP_TELLO,   "port": 8889,  "adapter": "tello"},
    {"brand": "DJI",        "model": "Mavic / Mini (RC WiFi mode)", "ip": _IP_TELLO,   "port": 2001,  "adapter": "mavlink"},
    {"brand": "Parrot",     "model": "ANAFI / ANAFI USA",           "ip": _IP_PARROT,  "port": 9988,  "adapter": "parrot"},
    {"brand": "Parrot",     "model": "Bebop 2",                     "ip": _IP_PARROT,  "port": 21,    "adapter": "parrot"},
    {"brand": "Skydio",     "model": "Skydio 2 / X2 / 3",          "ip": _IP_SKYDIO,  "port": 50051, "adapter": "generic_udp"},
    {"brand": "Autel",      "model": "EVO / EVO II",                "ip": _IP_AUTEL,   "port": 80,    "adapter": "generic_udp"},
    {"brand": "FIMI",       "model": "X8 SE / X8 Pro",              "ip": _IP_FIMI,    "port": 80,    "adapter": "generic_udp"},
    {"brand": "Hubsan",     "model": "Zino 2 / Zino Mini Pro",      "ip": _IP_TELLO,   "port": 8888,  "adapter": "generic_udp"},
    {"brand": "Holy Stone", "model": "HS720E / HS360S",             "ip": _IP_GENERIC, "port": 80,    "adapter": "generic_udp"},
    {"brand": "PowerVision","model": "PowerEgg X",                  "ip": _IP_TELLO,   "port": 8080,  "adapter": "generic_udp"},
]


def lookup_wifi_fingerprint(brand: str = "", model: str = "") -> Optional[dict]:
    brand_norm = str(brand or "").strip().lower()
    model_norm = str(model or "").strip().lower()
    if not brand_norm and not model_norm:
        return None

    for entry in WIFI_DRONE_FINGERPRINTS:
        e_brand = entry["brand"].strip().lower()
        e_model = entry["model"].strip().lower()
        if brand_norm and e_brand != brand_norm:
            continue
        if model_norm and (model_norm in e_model or e_model in model_norm):
            return dict(entry)

    for entry in WIFI_DRONE_FINGERPRINTS:
        if brand_norm and entry["brand"].strip().lower() == brand_norm:
            return dict(entry)
    return None


def probe_wifi_drones(timeout: float = 0.4) -> list[dict]:
    """
    Probe known drone gateway IPs on the current network.
    Returns list of reachable drones.
    """
    results = []
    seen_ips: dict[str, bool] = {}

    def check(entry: dict):
        ip = entry["ip"]
        port = entry["port"]
        if ip in seen_ips:
            entry["reachable"] = seen_ips[ip]
            if seen_ips[ip]:
                results.append({**entry})
            return
        try:
            s = socket.create_connection((ip, port), timeout=timeout)
            s.close()
            seen_ips[ip] = True
            results.append({**entry, "reachable": True})
        except Exception:
            seen_ips[ip] = False

    threads = [threading.Thread(target=check, args=(e,), daemon=True)
               for e in WIFI_DRONE_FINGERPRINTS]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout + 0.2)
    return results


# ── Video stream configurations ──────────────────────────────────────────────
DRONE_STREAM_CONFIGS = [
    {"brand": "DJI",     "model": "Tello / Tello EDU",
     "url": "udp://0.0.0.0:11111",         "host": "192.168.10.1",
     "note": "Connect to TELLO-XXXXXX WiFi; stream starts when SDK connect() called"},
    {"brand": "Parrot",  "model": "Bebop 2",
     "url": "rtsp://192.168.42.1/live",    "host": "192.168.42.1", "note": ""},
    {"brand": "Parrot",  "model": "ANAFI / ANAFI Ai",
     "url": "rtsp://192.168.42.1/live",    "host": "192.168.42.1",
     "note": "Requires Olympe SDK or direct RTSP access"},
    {"brand": "Skydio",  "model": "2 / X2 / 3",
     "url": "rtsp://192.168.77.1:8554/live", "host": "192.168.77.1", "note": ""},
    {"brand": "Autel",   "model": "EVO II",
     "url": "rtsp://192.168.2.1:554/live", "host": "192.168.2.1", "note": ""},
    {"brand": "Yuneec",  "model": "Typhoon H / H520",
     "url": "rtsp://192.168.42.1/live",    "host": "192.168.42.1", "note": ""},
    {"brand": "MAVLink", "model": "Companion computer (GStreamer/mediamtx)",
     "url": "rtsp://192.168.1.1:8554/stream", "host": "192.168.1.1",
     "note": "Common GStreamer + mediamtx setup on companion computers"},
    {"brand": "Generic", "model": "IP Camera / FPV VTx",
     "url": "rtsp://192.168.1.1:554/stream", "host": "192.168.1.1",
     "note": "Generic RTSP — adjust IP/port to match your device"},
]

# DJI WiFi credential factory defaults + recovery procedures
_DJI_WIFI_DEFAULTS: dict[str, dict] = {
    "Tello / Tello EDU":    {"password": "(open — no password)",   "reset": "Power cycle the Tello"},
    "Mavic Mini":           {"password": "12345678",                "reset": "Hold Power + RTH 9 s (lights flash)"},
    "Mavic Mini 2":         {"password": "12345678",                "reset": "Hold Power + RTH 9 s"},
    "Mavic Mini 3 / Pro":   {"password": "12345678",                "reset": "Hold Power + RTH 9 s"},
    "Mavic Air / Air 2":    {"password": "12345678",                "reset": "Hold Power + RTH 9 s"},
    "Mavic Air 2S":         {"password": "12345678",                "reset": "Hold Power + RTH 9 s"},
    "Mavic 2 Pro / Zoom":   {"password": "12345678",                "reset": "Hold Power + RTH 9 s"},
    "Mavic 3 (all variants)": {"password": "12345678",             "reset": "Hold Power + RTH 9 s"},
    "Phantom 4 / Pro / RTK": {"password": "12345678",              "reset": "Hold Power + RTH 9 s"},
    "Phantom 3 (all)":      {"password": "12345678",                "reset": "Hold Power + RTH 9 s"},
    "Inspire 1 / 2":        {"password": "12345678",                "reset": "Hold Power 10 s (4 beeps)"},
    "FPV / Avata / Avata 2": {"password": "12345678",              "reset": "Via DJI Fly app → Device → Reset"},
    "Spark":                {"password": "12345678",                "reset": "Hold Power 9 s"},
    "Matrice 200 / 300":    {"password": "12345678",                "reset": "Hold Power + RTH 9 s"},
    "Matrice 30 / 30T":     {"password": "12345678",                "reset": "Hold Power + RTH 9 s"},
    "Agras T-series":       {"password": "12345678",                "reset": "Via DJI Agras app → Settings → Reset"},
}

_ONVIF_NS = {
    "soap": "http://www.w3.org/2003/05/soap-envelope",
    "tds": "http://www.onvif.org/ver10/device/wsdl",
    "d": "http://schemas.xmlsoap.org/ws/2005/04/discovery",
    "wsa": "http://schemas.xmlsoap.org/ws/2004/08/addressing",
    "wsa05": "http://www.w3.org/2005/08/addressing",
}

_MDNS_SERVICE_TYPES = [
    "_http._tcp.local.",
    "_https._tcp.local.",
    "_rtsp._tcp.local.",
    "_airplay._tcp.local.",
    "_hap._tcp.local.",
]

_HTTP_VENDOR_HINTS = [
    ("hikvision", "Hikvision camera / NVR"),
    ("dahua", "Dahua camera / NVR"),
    ("axis", "Axis camera"),
    ("reolink", "Reolink camera"),
    ("amcrest", "Amcrest camera"),
    ("foscam", "Foscam camera"),
    ("ubiquiti", "Ubiquiti / UniFi device"),
    ("unifi", "Ubiquiti / UniFi device"),
    ("tapo", "TP-Link Tapo device"),
    ("tp-link", "TP-Link device"),
    ("wyze", "Wyze camera"),
    ("arlo", "Arlo camera"),
    ("ring", "Ring camera"),
    ("eufy", "Eufy camera"),
    ("goahead", "Embedded GoAhead web server"),
    ("boa", "Embedded Boa web server"),
    ("lighttpd", "Embedded Lighttpd web service"),
    ("nginx", "Embedded HTTP service"),
    ("microhttpd", "Embedded HTTP service"),
]


def scan_video_streams(ip: str, timeout: float = 0.8) -> list[dict]:
    """Probe common drone video streaming ports on the given IP."""
    checks = [
        (554,   "rtsp", "/live"),
        (554,   "rtsp", "/stream"),
        (8554,  "rtsp", "/live"),
        (8554,  "rtsp", "/stream"),
        (7070,  "http", "/video"),
        (8888,  "http", "/videostream.cgi"),
        (11111, "udp",  ""),
        (4747,  "http", "/video"),
        (8080,  "http", "/video"),
    ]
    found = []
    seen_ports: set[int] = set()

    def _probe(port: int, proto: str, path: str):
        if port in seen_ports:
            return
        try:
            s = socket.create_connection((ip, port), timeout=timeout)
            s.close()
            seen_ports.add(port)
            if proto == "udp":
                found.append({"url": f"udp://{ip}:{port}", "proto": "udp", "port": port})
            else:
                found.append({"url": f"{proto}://{ip}:{port}{path}",
                               "proto": proto, "port": port})
        except Exception:
            pass

    ts = [threading.Thread(target=_probe, args=c, daemon=True) for c in checks]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=timeout + 0.3)
    return found


def _xml_first_text(node: ET.Element, *paths: str) -> str:
    """Return the first non-empty text match for a list of namespace-aware paths."""
    for path in paths:
        found = node.find(path, _ONVIF_NS)
        if found is not None and found.text:
            text = found.text.strip()
            if text:
                return text
    return ""


def _decode_onvif_scope(scopes: str) -> dict[str, str]:
    """Extract readable metadata from ONVIF scope URIs."""
    meta: dict[str, str] = {}
    for raw_scope in scopes.split():
        scope = unquote(raw_scope.strip())
        lower = scope.lower()
        for key in ("name", "hardware", "location", "profile", "type"):
            marker = f"/{key}/"
            if marker in lower:
                meta[key] = scope.rsplit("/", 1)[-1].replace("_", " ")
                break
    return meta


def parse_onvif_probe_response(xml_bytes: bytes) -> list[dict]:
    """Parse ONVIF WS-Discovery ProbeMatch XML into normalized records."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []

    matches: list[dict] = []
    for probe_match in root.findall(".//d:ProbeMatch", _ONVIF_NS):
        endpoint = _xml_first_text(
            probe_match,
            "wsa:EndpointReference/wsa:Address",
            "wsa05:EndpointReference/wsa05:Address",
        )
        xaddrs = _xml_first_text(probe_match, "d:XAddrs")
        scopes = _xml_first_text(probe_match, "d:Scopes")
        meta = _decode_onvif_scope(scopes)
        xaddr_list = [x for x in xaddrs.split() if x]
        matches.append({
            "endpoint": endpoint,
            "xaddrs": xaddr_list,
            "xaddr": xaddr_list[0] if xaddr_list else "",
            "types": _xml_first_text(probe_match, "d:Types"),
            "scopes": scopes,
            "metadata_version": _xml_first_text(probe_match, "d:MetadataVersion"),
            "device_name": meta.get("name", ""),
            "hardware": meta.get("hardware", ""),
            "location": meta.get("location", ""),
        })

    deduped: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for match in matches:
        ident = (match.get("endpoint", ""), match.get("xaddr", ""))
        if ident in seen:
            continue
        seen.add(ident)
        deduped.append(match)
    return deduped


def parse_onvif_system_datetime(xml_bytes: bytes) -> Optional[dict]:
    """Parse ONVIF GetSystemDateAndTime response XML."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    def _dt(prefix: str) -> Optional[str]:
        year = _xml_first_text(root, f".//tds:{prefix}/tds:Date/tds:Year")
        month = _xml_first_text(root, f".//tds:{prefix}/tds:Date/tds:Month")
        day = _xml_first_text(root, f".//tds:{prefix}/tds:Date/tds:Day")
        hour = _xml_first_text(root, f".//tds:{prefix}/tds:Time/tds:Hour")
        minute = _xml_first_text(root, f".//tds:{prefix}/tds:Time/tds:Minute")
        second = _xml_first_text(root, f".//tds:{prefix}/tds:Time/tds:Second")
        if not all((year, month, day, hour, minute, second)):
            return None
        return (
            f"{int(year):04d}-{int(month):02d}-{int(day):02d} "
            f"{int(hour):02d}:{int(minute):02d}:{int(second):02d}"
        )

    tz = _xml_first_text(root, ".//tds:TimeZone/tds:TZ")
    return {
        "utc": _dt("UTCDateTime"),
        "local": _dt("LocalDateTime"),
        "timezone": tz,
        "daylight_savings": _xml_first_text(root, ".//tds:DaylightSavings"),
    }


def onvif_ws_discover(timeout: float = 2.0, attempts: int = 2) -> list[dict]:
    """
    Discover ONVIF cameras on the local network using WS-Discovery multicast.
    Returns normalized ProbeMatch records.
    """
    envelope = f"""<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope"
            xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing"
            xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
            xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
  <e:Header>
    <w:MessageID>uuid:{uuid.uuid4()}</w:MessageID>
    <w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>
    <w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>
  </e:Header>
  <e:Body>
    <d:Probe>
      <d:Types>dn:NetworkVideoTransmitter</d:Types>
    </d:Probe>
  </e:Body>
</e:Envelope>""".encode("utf-8")

    devices: list[dict] = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        sock.settimeout(timeout)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        deadline = time.time() + max(timeout, 0.5)
        for _ in range(max(1, attempts)):
            sock.sendto(envelope, ("239.255.255.250", 3702))
        while time.time() < deadline:
            try:
                payload, addr = sock.recvfrom(65535)
            except socket.timeout:
                break
            for match in parse_onvif_probe_response(payload):
                match["source_ip"] = addr[0]
                devices.append(match)
    except Exception:
        return []
    finally:
        sock.close()

    deduped: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for device in devices:
        ident = (
            device.get("endpoint", "") or device.get("source_ip", ""),
            device.get("xaddr", ""),
        )
        if ident in seen:
            continue
        seen.add(ident)
        deduped.append(device)
    return deduped


def candidate_onvif_xaddrs(ip: str, ports: Optional[list[int]] = None) -> list[str]:
    """Build common ONVIF device-service URLs for a specific IP."""
    ports = sorted(set(ports or [80, 443, 8000, 8080, 8443, 8899]))
    candidates: list[str] = []
    for port in ports:
        schemes = ["https"] if port == 443 else ["http"]
        for scheme in schemes:
            suffix = "" if (scheme == "http" and port == 80) or (scheme == "https" and port == 443) else f":{port}"
            candidates.append(f"{scheme}://{ip}{suffix}/onvif/device_service")
    return candidates


def onvif_get_system_datetime(xaddr: str, timeout: float = 3.0) -> tuple[Optional[dict], str]:
    """
    Probe an ONVIF device service with GetSystemDateAndTime.
    Returns (parsed_data, status_message).
    """
    body = """<?xml version="1.0" encoding="UTF-8"?>
<soap:Envelope xmlns:soap="http://www.w3.org/2003/05/soap-envelope">
  <soap:Body>
    <GetSystemDateAndTime xmlns="http://www.onvif.org/ver10/device/wsdl"/>
  </soap:Body>
</soap:Envelope>"""
    headers = {
        "Content-Type": "application/soap+xml; charset=utf-8",
        "User-Agent": "DroneDetect/1.0",
    }
    try:
        resp = requests.post(
            xaddr,
            data=body.encode("utf-8"),
            headers=headers,
            timeout=timeout,
            verify=False,
        )
    except requests.RequestException as exc:
        return None, f"Probe failed: {exc}"

    if resp.status_code in (401, 403):
        return None, f"Auth required ({resp.status_code})"
    if resp.status_code >= 400:
        return None, f"HTTP {resp.status_code}"

    parsed = parse_onvif_system_datetime(resp.content)
    if parsed:
        parsed["status_code"] = resp.status_code
        parsed["server"] = resp.headers.get("Server", "")
        return parsed, f"ok ({resp.status_code})"
    return None, f"Unexpected SOAP response ({resp.status_code})"


def probe_onvif_endpoints(ip: str, ports: Optional[list[int]] = None,
                          timeout: float = 2.5) -> list[dict]:
    """Probe common ONVIF service URLs on a specific IP."""
    findings: list[dict] = []
    for xaddr in candidate_onvif_xaddrs(ip, ports=ports):
        parsed, status = onvif_get_system_datetime(xaddr, timeout=timeout)
        findings.append({
            "ip": ip,
            "xaddr": xaddr,
            "status": status,
            "datetime": parsed,
        })
        if parsed:
            break
    return findings


def _extract_html_title(html_text: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    return title[:160]


def classify_http_fingerprint(server: str = "", title: str = "", realm: str = "",
                              location: str = "") -> str:
    """Turn raw HTTP response hints into a concise vendor/device guess."""
    haystack = " ".join(part for part in (server, title, realm, location) if part).lower()
    for needle, label in _HTTP_VENDOR_HINTS:
        if needle in haystack:
            return label
    if "camera" in haystack or "rtsp" in haystack or "onvif" in haystack:
        return "Network camera / video appliance"
    if "login" in haystack or "admin" in haystack:
        return "Embedded admin UI"
    return "Unknown embedded web service"


def fingerprint_http_device(ip: str, ports: Optional[list[int]] = None,
                            timeout: float = 1.2) -> list[dict]:
    """Probe HTTP(S) endpoints and return lightweight fingerprint data."""
    ports = sorted(set(ports or [80, 443, 8080, 8443, 8888]))
    findings: list[dict] = []
    for port in ports:
        scheme = "https" if port in {443, 8443} else "http"
        suffix = "" if (scheme == "http" and port == 80) or (scheme == "https" and port == 443) else f":{port}"
        url = f"{scheme}://{ip}{suffix}/"
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                verify=False,
                allow_redirects=True,
                headers={"User-Agent": "DroneDetect/1.0"},
                stream=True,
            )
            try:
                chunk = resp.raw.read(4096, decode_content=True) or b""
            finally:
                resp.close()
        except requests.RequestException as exc:
            findings.append({
                "url": url,
                "status": "error",
                "detail": str(exc),
                "device_hint": "",
            })
            continue

        raw_text = chunk.decode(resp.encoding or "utf-8", errors="ignore")
        title = _extract_html_title(raw_text)
        auth_header = resp.headers.get("WWW-Authenticate", "")
        realm_match = re.search(r'realm="?([^",]+)', auth_header, re.IGNORECASE)
        realm = realm_match.group(1).strip() if realm_match else ""
        location = resp.headers.get("Location", "")
        server = resp.headers.get("Server", "")
        findings.append({
            "url": url,
            "status": resp.status_code,
            "server": server,
            "realm": realm,
            "title": title,
            "location": location,
            "device_hint": classify_http_fingerprint(server, title, realm, location),
        })
    return findings


class _MDNSCollector(_ZCServiceListener if HAVE_ZEROCONF else object):
    """Collect mDNS/Bonjour services while Zeroconf browsers run."""
    def __init__(self):
        self.records: list[dict] = []
        self._seen: set[tuple[str, str]] = set()

    def add_service(self, zc, service_type, name):  # pragma: no cover - network callback
        try:
            info = zc.get_service_info(service_type, name, timeout=1000)
        except Exception:
            info = None
        if info is None:
            return
        host = ""
        addresses = []
        for packed in list(getattr(info, "addresses", []) or []):
            try:
                host_ip = socket.inet_ntoa(packed)
            except OSError:
                continue
            addresses.append(host_ip)
        if addresses:
            host = addresses[0]
        ident = (service_type, name)
        if ident in self._seen:
            return
        self._seen.add(ident)
        props = {}
        for key, value in dict(getattr(info, "properties", {}) or {}).items():
            k = key.decode(errors="ignore") if isinstance(key, bytes) else str(key)
            v = value.decode(errors="ignore") if isinstance(value, bytes) else str(value)
            props[k] = v
        self.records.append({
            "service_type": service_type,
            "name": name,
            "ip": host,
            "port": getattr(info, "port", 0),
            "properties": props,
        })

    update_service = add_service

    def remove_service(self, zc, service_type, name):  # pragma: no cover - network callback
        return None


def discover_mdns_services(timeout: float = 2.0,
                           service_types: Optional[list[str]] = None) -> tuple[list[dict], str]:
    """Discover mDNS/Bonjour services for camera/IoT inventory."""
    if not HAVE_ZEROCONF:
        return [], "python-zeroconf not installed"
    zc = _Zeroconf()
    collector = _MDNSCollector()
    browsers = []
    try:
        for service_type in (service_types or _MDNS_SERVICE_TYPES):
            browsers.append(_ZCServiceBrowser(zc, service_type, collector))
        time.sleep(max(timeout, 0.5))
        return sorted(
            collector.records,
            key=lambda row: (row.get("service_type", ""), row.get("name", "")),
        ), ""
    except Exception as exc:
        return [], str(exc)
    finally:
        try:
            zc.close()
        except Exception:
            pass


def module_available(module_name: str) -> bool:
    """Safe import-spec check for optional runtime modules."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def capability_audit() -> list[dict]:
    """Return the runtime capability matrix for protocol discovery and control."""
    items = [
        ("HackRF CLI", have_hackrf(), "RF sweep / raw IQ capture", "`hackrf_transfer`"),
        ("SoapySDR", HAVE_SOAPY, "Multi-radio SDR backends", "SoapySDR"),
        ("requests", module_available("requests"), "HTTP + ONVIF SOAP probing", "pip install requests"),
        ("opencv-python", HAVE_CV2, "RTSP/MJPEG/UDP video frame grabs", "pip install opencv-python"),
        ("djitellopy", HAVE_TELLO_SDK, "Tello direct command + video session", "pip install djitellopy"),
        ("pymavlink", HAVE_MAVLINK, "MAVLink telemetry / action bridge", "pip install pymavlink"),
        ("bleak", module_available("bleak"), "BLE Remote ID scans", "pip install bleak"),
        ("zeroconf", HAVE_ZEROCONF, "mDNS / Bonjour service discovery", "pip install zeroconf"),
        ("scapy", module_available("scapy"), "Deeper packet decode / PCAP workflows", "pip install scapy"),
    ]
    rows = []
    for name, enabled, capability, install_hint in items:
        rows.append({
            "Capability": name,
            "Status": "ready" if enabled else "missing",
            "Unlocks": capability,
            "Install": install_hint,
        })
    return rows


def direction_finding_capability() -> list[dict]:
    """Practical DF/geolocation options for the hardware this app commonly uses."""
    have_hackrf_hw = have_hackrf()
    return [
        {
            "Technique": "Single SDR + directional antenna",
            "Feasible now": "yes" if have_hackrf_hw else "partial",
            "What it gives you": "Manual bearing cuts and strongest-signal hunting",
            "What you still need": "Directional antenna plus multiple bearings or operator movement",
        },
        {
            "Technique": "FDOA",
            "Feasible now": "no",
            "What it gives you": "Doppler-based motion hints only",
            "What you still need": "Receiver motion, tight oscillator discipline, and calibration",
        },
        {
            "Technique": "TDOA",
            "Feasible now": "no",
            "What it gives you": "True multilateration / emitter geolocation",
            "What you still need": "Multiple synchronized receivers with shared timebase or GPSDO/PPS",
        },
        {
            "Technique": "Phase/AOA interferometry",
            "Feasible now": "no",
            "What it gives you": "Instant bearing estimation",
            "What you still need": "Phase-coherent multi-channel SDR such as KrakenSDR/KerberosSDR/USRP MIMO",
        },
    ]


def build_owner_recovery_playbook(ip: str = "", open_ports: Optional[list[int]] = None,
                                  onvif_data: Optional[dict] = None,
                                  http_data: Optional[dict] = None) -> list[str]:
    """Generate owner-safe next steps for a discovered drone/camera/IoT device."""
    open_ports = open_ports or []
    steps: list[str] = []
    if 8889 in open_ports or ip == _IP_TELLO:
        steps.append(
            "Tello-class control path detected: join the drone AP, send `command` to "
            "`192.168.10.1:8889`, then use the Command tab for `takeoff`, `land`, `streamon`."
        )
    if any(port in open_ports for port in (14550, 14551, 5760, 18570)):
        steps.append(
            "MAVLink surface detected: connect a GCS or adapter on `udpin:0.0.0.0:14550` "
            "or `tcp:{ip}:5760`, then issue RTL/Land from the Command tab.".replace("{ip}", ip)
        )
    if any(port in open_ports for port in (554, 8554, 11111, 4747, 7070)):
        steps.append(
            "Video path detected: use Camera tab auto-discovery or try RTSP/UDP URLs before changing device settings."
        )
    if onvif_data:
        steps.append(
            "ONVIF responded without packet guessing. Use the reported device service URL and the vendor's official reset flow if credentials were changed."
        )
    if http_data and http_data.get("realm"):
        steps.append(
            f"HTTP auth realm `{http_data['realm']}` is exposed. Treat this as the owner's admin UI and recover via vendor reset / documented default credentials only."
        )
    if not steps:
        steps.append(
            "No direct control surface was confirmed. Inventory the ports, identify the protocol, and use vendor-supported reset or pairing procedures rather than brute force."
        )
    return steps


def summarize_ports(ports: list[int], limit: int = 4) -> str:
    """Compact port/service summary for tables and cards."""
    if not ports:
        return "—"
    ports = sorted(set(int(p) for p in ports))
    shown = ports[:limit]
    parts = [f"{p}/{_PORT_LABELS.get(p, '')}".rstrip("/") for p in shown]
    if len(ports) > limit:
        parts.append(f"+{len(ports) - limit}")
    return ", ".join(parts)


def summarize_mdns_properties(props: dict, limit: int = 3) -> str:
    """Compact Bonjour property summary."""
    if not props:
        return "—"
    items = [f"{k}={v}" for k, v in sorted(props.items()) if v]
    if not items:
        return "—"
    if len(items) > limit:
        items = items[:limit] + [f"+{len(items) - limit}"]
    return ", ".join(items)


def onvif_device_title(device: dict) -> str:
    """Human-friendly ONVIF device title."""
    return device.get("device_name") or device.get("hardware") or "ONVIF device"


def grab_video_frame(url: str, timeout_ms: int = 3000) -> "Optional[np.ndarray]":
    """Grab one frame from a video stream URL. Returns RGB ndarray or None."""
    if not HAVE_CV2:
        return None
    try:
        cap = _cv2.VideoCapture(url)
        cap.set(_cv2.CAP_PROP_BUFFERSIZE, 1)
        if hasattr(_cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            cap.set(_cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(timeout_ms))
            cap.set(_cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(timeout_ms))
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return None


def generate_dji_password_candidates(ssid: str, serial: str = "") -> list[str]:
    """
    Generate likely WiFi password candidates for a DJI drone that the
    owner has forgotten the password for. Based on documented factory
    defaults and common user patterns.
    """
    candidates: list[str] = ["12345678"]  # factory default on virtually all DJI consumer drones
    # If serial number provided, add serial-based patterns (older firmware)
    if serial:
        s = serial.strip().upper()
        if len(s) >= 8:
            candidates.append(s[-8:])
            candidates.append(s[-6:])
            candidates.append(s[:8])
        candidates.append(s)
    # Extract trailing digits from SSID (some older models used last 4 of S/N in SSID)
    import re
    digits = re.sub(r"[^0-9]", "", ssid)
    if digits:
        candidates.append(digits[-8:].zfill(8))
        candidates.append(digits[-4:])
    # Common user-chosen patterns
    candidates += ["djiowner", "drone1234", "12341234", "00000000", "87654321"]
    # Deduplicate while preserving order
    seen: set[str] = set()
    out = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


# Known drone WiFi SSID prefixes for active network scan
# Comprehensive drone SSID prefix database.
# Format: (brand, model_hint, [SSID_prefix_uppercase, ...])
# Prefixes are matched case-insensitively against the start of the SSID.
DRONE_SSID_PREFIXES = [
    # ── DJI ──────────────────────────────────────────────────────────────────
    ("DJI",         "Tello / Tello EDU",          ["TELLO-", "TELLO_", "RYZERTELLO"]),
    ("DJI",         "Mavic series",               ["MAVIC-", "MAVIC_"]),
    ("DJI",         "Mini series",                ["DJI-MINI", "DJIMINI"]),
    ("DJI",         "Air series",                 ["DJI-AIR", "DJIAIR"]),
    ("DJI",         "FPV / Avata",                ["DJI-FPV", "DJIFPV", "AVATA-", "AVATA_"]),
    ("DJI",         "Phantom",                    ["PHANTOM-", "PHANTOM_"]),
    ("DJI",         "Inspire",                    ["INSPIRE-", "INSPIRE_"]),
    ("DJI",         "Spark",                      ["SPARK-", "SPARK_"]),
    ("DJI",         "Neo / generic",              ["DJI-", "DJI_", "NEO-"]),
    # ── Parrot ────────────────────────────────────────────────────────────────
    ("Parrot",      "ANAFI / ANAFI Ai",           ["ANAFI-", "ANAFI_"]),
    ("Parrot",      "Bebop 1 / 2",                ["BEBOPDRONE", "BEBOP_", "BEBOP-"]),
    ("Parrot",      "Disco (fixed wing)",         ["DISCO-", "DISCO_"]),
    ("Parrot",      "Mambo / Swing toy",          ["MAMBO-", "MAMBO_", "SWING-"]),
    ("Parrot",      "Rolling Spider",             ["ROLLING-", "MINIDRONE"]),
    # ── Skydio ────────────────────────────────────────────────────────────────
    ("Skydio",      "2 / X2 / 3 / X10",          ["SKYDIO-", "SKYDIO_", "X2-", "X10-"]),
    # ── Autel Robotics ────────────────────────────────────────────────────────
    ("Autel",       "EVO / EVO II / Lite",        ["AUTEL-", "AUTEL_", "AUTELEVO", "EVO2-"]),
    ("Autel",       "DragonFish",                 ["DRAGONFISH-", "DRAGONFISH_"]),
    # ── FIMI (Xiaomi) ─────────────────────────────────────────────────────────
    ("FIMI",        "X8 SE / X8 Pro / A3",        ["FIMI-", "FIMI_", "FIMI_X8", "FIMIA3"]),
    # ── Hubsan ────────────────────────────────────────────────────────────────
    ("Hubsan",      "Zino 2 / Mini Pro",          ["HUBSAN", "HUBSAN_", "ZINO-", "ZINO_"]),
    # ── Holy Stone ────────────────────────────────────────────────────────────
    ("Holy Stone",  "HS series",                  ["HS720", "HS360", "HS175", "HS100", "HS400",
                                                   "HOLYSTONE", "HOLY-"]),
    # ── PowerVision ───────────────────────────────────────────────────────────
    ("PowerVision", "PowerEgg / PowerRay",        ["POWEREGG", "POWERRAY", "POWERDOLPHIN", "PVS_"]),
    # ── Yuneec ────────────────────────────────────────────────────────────────
    ("Yuneec",      "Typhoon / H520 / E90",       ["TYPHOON", "YUNEEC", "H520-", "H520E", "E90-"]),
    # ── XAG Agricultural ──────────────────────────────────────────────────────
    ("XAG",         "P100 / V40 / P80",           ["XAG-", "XAG_"]),
    # ── Wingcopter ────────────────────────────────────────────────────────────
    ("Wingcopter",  "198 / 178",                  ["WINGCOPTER", "WC-"]),
    # ── 3DR ───────────────────────────────────────────────────────────────────
    ("3DR",         "Solo / Iris+",               ["SOLOGOPRO", "SOLO-", "3DR-", "IRIS-"]),
    # ── Potensic ──────────────────────────────────────────────────────────────
    ("Potensic",    "Atom / D80 / T25",           ["POTENSIC", "ATOM-"]),
    # ── Ruko ──────────────────────────────────────────────────────────────────
    ("Ruko",        "F11 / U11 series",           ["RUKO-", "RUKO_"]),
    # ── Snaptain ──────────────────────────────────────────────────────────────
    ("Snaptain",    "SP / S5C series",            ["SNAPTAIN", "SP-SNAPTAIN"]),
    # ── Eachine ───────────────────────────────────────────────────────────────
    ("Eachine",     "E520 / EX5 series",          ["EACHINE", "EX-EACHINE"]),
    # ── Walkera ───────────────────────────────────────────────────────────────
    ("Walkera",     "Vitus / Voyager",            ["WALKERA", "VITUS-", "VOYAGER-"]),
    # ── MJX / budget Chinese ──────────────────────────────────────────────────
    ("MJX",         "Bugs series",                ["MJX-", "MJX_", "BUGS-"]),
    # ── SJRC / ZLRC ───────────────────────────────────────────────────────────
    ("SJRC",        "F11 / Z5 series",            ["SJRC-", "SJRC_", "ZLRC-"]),
    # ── Contixo ───────────────────────────────────────────────────────────────
    ("Contixo",     "F24 / F35 series",           ["CONTIXO", "F24-"]),
    # ── Altair Aerial ─────────────────────────────────────────────────────────
    ("Altair",      "Blackhawk / Outlaw",         ["ALTAIR-", "AA-", "BLACKHAWK-"]),
    # ── DBPower / UDI ─────────────────────────────────────────────────────────
    ("DBPower",     "UDI / Discovery",            ["DBPOWER", "UDI-", "DISCOVERY-"]),
    # ── Wingsland ─────────────────────────────────────────────────────────────
    ("Wingsland",   "S6 / Mini",                  ["WINGSLAND", "S6-WINGSLAND"]),
    # ── Force1 ────────────────────────────────────────────────────────────────
    ("Force1",      "F100 / Scour series",        ["FORCE1-", "F100-"]),
    # ── Ryze Tech ─────────────────────────────────────────────────────────────
    ("Ryze Tech",   "Tello EDU",                  ["RYZETECH", "RYZERTELLO"]),
    # ── Teal Drones ───────────────────────────────────────────────────────────
    ("Teal",        "Golden Eagle",               ["TEAL-", "TEAL_"]),
    # ── senseFly / AgEagle ────────────────────────────────────────────────────
    ("AgEagle",     "eBee X / eBee Ag",           ["EBEE-", "EBEE_", "SENSEFLY", "AGEAGLE"]),
    # ── Flyability ────────────────────────────────────────────────────────────
    ("Flyability",  "Elios 3",                    ["ELIOS-", "ELIOS_", "FLYABILITY"]),
    # ── BRINC ─────────────────────────────────────────────────────────────────
    ("BRINC",       "Lemur 2",                    ["BRINC-", "BRINC_", "LEMUR-"]),
    # ── Axon ──────────────────────────────────────────────────────────────────
    ("Axon",        "Air R7",                     ["AXON-", "AXONAIR", "R7-"]),
    # ── Percepto ──────────────────────────────────────────────────────────────
    ("Percepto",    "Sparrow / Sparrow II",       ["PERCEPTO", "SPARROW-"]),
    # ── Draganfly ─────────────────────────────────────────────────────────────
    ("Draganfly",   "Commander / Tango",          ["DRAGANFLY", "DRAGANFLYER"]),
    # ── EHang ─────────────────────────────────────────────────────────────────
    ("EHang",       "216 / 216S",                 ["EHANG-", "EHANG_"]),
    # ── Zipline ───────────────────────────────────────────────────────────────
    ("Zipline",     "P2 Zip",                     ["ZIPLINE-", "ZIPLINE_"]),
    # ── Wing (Alphabet) ───────────────────────────────────────────────────────
    ("Wing",        "Delivery drone",             ["WING-DELIVERY", "ALPHABETWING"]),
    # ── Matternet ─────────────────────────────────────────────────────────────
    ("Matternet",   "M2",                         ["MATTERNET", "M2-MATTERNET"]),
    # ── Freefly Systems ───────────────────────────────────────────────────────
    ("Freefly",     "Alta X / Alta 8",            ["ALTA-", "FREEFLY-"]),
    # ── ModalAI ───────────────────────────────────────────────────────────────
    ("ModalAI",     "Starling / VOXL",            ["VOXL-", "MODALAI-", "STARLING-"]),
    # ── WingtraOne ────────────────────────────────────────────────────────────
    ("Wingtra",     "WingtraOne Gen II",          ["WINGTRA-", "WINGTRAONE"]),
    # ── Quantum Systems ───────────────────────────────────────────────────────
    ("Quantum Sys", "Trinity F90+",              ["TRINITY-", "QUANTUMSYS"]),
    # ── Delair ────────────────────────────────────────────────────────────────
    ("Delair",      "DX8 / UX11",                 ["DELAIR-", "DELAIR_", "UX11-"]),
    # ── Inspired Flight ───────────────────────────────────────────────────────
    ("Inspired",    "IF1200A",                    ["IF1200", "INSPIREDFLIGHT"]),
    # ── Syma (toy) ────────────────────────────────────────────────────────────
    ("Syma",        "X series toy",               ["SYMA-", "SYMA_"]),
    # ── Generic / unknown drone AP ────────────────────────────────────────────
    ("Unknown",     "Generic drone AP",           ["DRONE-", "DRONE_", "UAV-", "UAV_",
                                                   "QUADCOPTER", "FPVDRONE", "FPV-DRONE"]),
]


def scan_wifi_networks() -> tuple[list, list, str]:
    """Scan nearby WiFi using nmcli and match SSIDs against known drone patterns.

    Returns (drone_hits, all_networks, error_str).
    all_networks includes every visible AP — use this for the radar chart.
    """
    try:
        r = subprocess.run(
            ["nmcli", "--terse", "-f", "SSID,SIGNAL,CHAN", "dev", "wifi", "list", "--rescan", "yes"],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode != 0:
            return [], [], r.stderr.strip() or "nmcli returned non-zero"

        all_nets = []
        seen_ssids: set = set()
        for line in r.stdout.splitlines():
            parts = line.rsplit(":", 2)
            if len(parts) < 3:
                continue
            ssid = parts[0].replace("\\:", ":").strip()
            if not ssid:
                continue
            try:
                signal = int(parts[1])
                chan = int(parts[2])
            except ValueError:
                continue

            band = "5 GHz" if chan > 14 else "2.4 GHz"
            entry = {"ssid": ssid, "signal_pct": signal, "channel": chan, "band": band}

            if ssid not in seen_ssids:
                seen_ssids.add(ssid)
                ssid_upper = ssid.upper()
                for brand, model, prefixes in DRONE_SSID_PREFIXES:
                    if any(ssid_upper.startswith(p.upper()) for p in prefixes):
                        entry.update({"brand": brand, "model": model, "is_drone": True})
                        hint = lookup_wifi_fingerprint(brand, model)
                        if hint:
                            entry.update({
                                "ip": hint["ip"],
                                "port": hint["port"],
                                "adapter": hint["adapter"],
                            })
                        break

            all_nets.append(entry)

        drone_hits = [n for n in all_nets if n.get("is_drone")]
        return drone_hits, all_nets, ""
    except FileNotFoundError:
        return [], [], "nmcli not found — install NetworkManager"
    except subprocess.TimeoutExpired:
        return [], [], "WiFi scan timed out after 20 s"
    except Exception as exc:
        return [], [], str(exc)


def build_wifi_radar_fig(all_networks: list, interval_s: float = 20.0,
                          last_scan_time: float = 0.0) -> go.Figure:
    """Horizontal bar chart of all visible WiFi networks — the WiFi-mode heartbeat."""
    if not all_networks:
        fig = go.Figure()
        fig.update_layout(
            title="WiFi Radar — waiting for first scan…",
            paper_bgcolor="#07111a", plot_bgcolor="#0c1722",
            font={"color": "#94a3b8"}, height=160,
        )
        return fig

    nets = sorted(all_networks, key=lambda n: n.get("signal_pct", 0), reverse=True)[:50]
    ssids   = [n.get("ssid", "?")[:28] for n in nets]
    signals = [n.get("signal_pct", 0) for n in nets]
    colors  = [
        "#22c55e" if n.get("is_drone") else
        "#0ea5e9" if n.get("band") == "5 GHz" else
        "#334155"
        for n in nets
    ]
    hovers = [
        (f"{n.get('ssid','?')}<br>Signal: {n.get('signal_pct',0)}%  Ch {n.get('channel','?')}  {n.get('band','?')}"
         + (f"<br>🚁 <b>{n.get('brand','?')}</b> — {n.get('model','?')}" if n.get("is_drone") else ""))
        for n in nets
    ]

    elapsed   = time.time() - last_scan_time if last_scan_time else 0
    remaining = max(0.0, interval_s - elapsed)
    drones    = sum(1 for n in all_networks if n.get("is_drone"))

    fig = go.Figure(go.Bar(
        y=ssids, x=signals,
        orientation="h",
        marker_color=colors,
        hovertext=hovers,
        hoverinfo="text",
    ))
    fig.update_layout(
        title=(f"📶 WiFi Radar — {len(all_networks)} networks · "
               f"{'🚁 ' + str(drones) + ' drone(s) detected!' if drones else 'no drones matched'} · "
               f"next scan in {remaining:.0f}s"),
        xaxis={"title": "Signal %", "range": [0, 105],
               "gridcolor": "#1e293b", "color": "#94a3b8"},
        yaxis={"tickfont": {"size": 9}, "color": "#94a3b8"},
        paper_bgcolor="#07111a",
        plot_bgcolor="#0c1722",
        font={"color": "#cbd5e1"},
        height=max(220, min(640, 18 * len(nets) + 80)),
        margin={"l": 200, "r": 20, "t": 50, "b": 30},
        showlegend=False,
    )
    return fig

# =============================================================================
# SIGNAL ANALYSIS (ported from Scanner.py)
# =============================================================================

MAX_ANALYSIS_SAMPLES = 131072
DETAIL_FFT_BINS = 4096
WATERFALL_ROWS = 60
WATERFALL_BINS = 512

def compute_spectrum(iq: np.ndarray, sample_rate: float):
    analysis_iq = iq[:MAX_ANALYSIS_SAMPLES] if iq.size > MAX_ANALYSIS_SAMPLES else iq
    if analysis_iq.size < 256:
        freqs = np.linspace(-sample_rate / 2, sample_rate / 2, 256, dtype=np.float32)
        return freqs, np.full(256, -140.0, dtype=np.float32)
    nperseg = min(DETAIL_FFT_BINS, analysis_iq.size)
    freqs, psd = scipy_signal.welch(
        analysis_iq,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        return_onesided=False,
        scaling="spectrum",
    )
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    db = 10.0 * np.log10(np.maximum(psd, 1e-15))
    return freqs.astype(np.float32), db.astype(np.float32)


def extract_metrics(freq_axis: np.ndarray, spectrum_db: np.ndarray):
    if spectrum_db.size == 0:
        return -140.0, -140.0, 0.0, 0.0, 0.0
    noise_floor = float(np.percentile(spectrum_db, 60))
    peaks, props = scipy_signal.find_peaks(spectrum_db, prominence=3.0)
    if peaks.size:
        prominences = props.get("prominences", np.zeros(peaks.size))
        best = int(peaks[int(np.argmax(prominences))])
        peak_prominence = float(prominences[int(np.argmax(prominences))])
    else:
        best = int(np.argmax(spectrum_db))
        peak_prominence = float(spectrum_db[best] - noise_floor)
    peak_db = float(spectrum_db[best])
    offset_hz = float(freq_axis[best])
    snr = max(peak_db - noise_floor, peak_prominence, 0.0)
    # Occupied bandwidth: 3dB below gate
    gate = noise_floor + max(3.0, min(6.0, snr * 0.5))
    left, right = best, best
    while left > 0 and spectrum_db[left] >= gate:
        left -= 1
    while right < spectrum_db.size - 1 and spectrum_db[right] >= gate:
        right += 1
    occ_bw = abs(float(freq_axis[right]) - float(freq_axis[left])) if right > left else 0.0
    return peak_db, noise_floor, snr, offset_hz, occ_bw

# =============================================================================
# SDR CAPTURE BACKENDS (same design as Scanner.py)
# =============================================================================

TMP_FILE = "/tmp/dronedetect_iq.raw"
DEFAULT_RECORDINGS_DIR = "recordings"
DEFAULT_RECORD_COOLDOWN_S = 3.0
HACKRF_BW_OPTIONS = np.array([
    1.75e6, 2.5e6, 3.5e6, 5.0e6, 5.5e6, 6.0e6, 7.0e6, 8.0e6,
    9.0e6, 10.0e6, 12.0e6, 14.0e6, 15.0e6, 20.0e6, 24.0e6, 28.0e6,
], dtype=np.float64)
SOAPY_ENUM_TIMEOUT_S = 4.0
SOAPY_AUTO_SKIP_DRIVERS = {"audio", "uhd"}
SOAPY_PREFERRED_DRIVERS = (
    "hackrf",
    "rtlsdr",
    "airspy",
    "lime",
    "sdrplay",
    "plutosdr",
    "bladerf",
    "flex",
)
_SOAPY_ENUM_SCRIPT = """
import json
import sys

payload = {"ok": False, "devices": [], "error": ""}
args = {}
if len(sys.argv) > 1 and sys.argv[1]:
    try:
        args = json.loads(sys.argv[1])
    except Exception as exc:
        payload["error"] = f"bad args: {exc}"
        print(json.dumps(payload))
        raise SystemExit(0)

try:
    import SoapySDR
except Exception as exc:
    payload["error"] = f"SoapySDR import failed: {exc}"
    print(json.dumps(payload))
    raise SystemExit(0)

try:
    raw = SoapySDR.Device.enumerate(args) if args else SoapySDR.Device.enumerate()
    payload["ok"] = True
    payload["devices"] = [{k: d[k] for k in d.keys()} for d in raw]
except Exception as exc:
    payload["error"] = str(exc)

print(json.dumps(payload))
"""


def have_hackrf() -> bool:
    """True only when a HackRF board is physically connected."""
    if shutil.which("hackrf_info") is None:
        return False
    try:
        r = subprocess.run(["hackrf_info"], capture_output=True, text=True, timeout=3)
        return "Found HackRF" in r.stdout or "HackRF One" in r.stdout
    except Exception:
        return False


def parse_soapy_args_text(text: str) -> dict:
    args = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            args[k.strip()] = v.strip()
        elif "driver" not in args:
            args["driver"] = part
    return args


def _normalize_soapy_device(raw: dict) -> dict:
    row = {str(k): str(v) for k, v in dict(raw).items()}
    row["driver"] = row.get("driver", "").strip()
    row["label"] = row.get("label") or row["driver"] or row.get("serial", "") or "unknown"
    return row


def _filter_soapy_devices(devices: list[dict], allow_uhd: bool = True) -> list[dict]:
    blocked = {"audio"}
    if not allow_uhd:
        blocked |= {"uhd"}
    filtered = []
    for raw in devices:
        row = _normalize_soapy_device(raw)
        if row["driver"].lower() in blocked:
            continue
        filtered.append(row)
    return filtered


def choose_soapy_device(devices: list[dict], preferred_driver: str = "") -> Optional[dict]:
    if not devices:
        return None
    preferred_driver = str(preferred_driver or "").strip().lower()
    ranked = []
    for row in devices:
        driver = row.get("driver", "").strip().lower()
        if preferred_driver and driver == preferred_driver:
            rank = (0, row.get("label", ""))
        elif driver in SOAPY_PREFERRED_DRIVERS:
            rank = (1, SOAPY_PREFERRED_DRIVERS.index(driver), row.get("label", ""))
        else:
            rank = (2, driver, row.get("label", ""))
        ranked.append((rank, row))
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def enumerate_soapy_devices_safe(soapy_args_text: str = "", timeout_s: float = SOAPY_ENUM_TIMEOUT_S) -> tuple[list[dict], str]:
    """Enumerate SoapySDR devices out-of-process so driver crashes don't kill Streamlit."""
    if not HAVE_SOAPY:
        return [], "SoapySDR runtime not installed."
    args = parse_soapy_args_text(soapy_args_text)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SOAPY_ENUM_SCRIPT, json.dumps(args)],
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [], "SoapySDR inventory timed out."
    except Exception as exc:
        return [], f"SoapySDR inventory failed: {exc}"

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        if proc.returncode < 0:
            return [], f"SoapySDR inventory crashed with signal {-proc.returncode}."
        detail = stderr.splitlines()[-1] if stderr else stdout.splitlines()[-1] if stdout else f"exit {proc.returncode}"
        return [], f"SoapySDR inventory failed: {detail}"

    if not stdout:
        return [], "SoapySDR inventory returned no data."

    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        return [], "SoapySDR inventory returned invalid data."

    if not payload.get("ok"):
        detail = str(payload.get("error") or "unknown error").strip()
        return [], f"SoapySDR inventory failed: {detail}"

    return [_normalize_soapy_device(row) for row in payload.get("devices", [])], ""


@st.cache_data(ttl=30, show_spinner=False)
def cached_sdr_inventory(soapy_args_text: str = "") -> dict:
    devices, note = enumerate_soapy_devices_safe(soapy_args_text)
    return {"devices": devices, "note": note}


def hackrf_capture(freq_hz: float, sample_rate: float, bandwidth_hz: float,
                   seconds: float, gain: int) -> Optional[np.ndarray]:
    bw = float(HACKRF_BW_OPTIONS[np.argmin(np.abs(HACKRF_BW_OPTIONS - bandwidth_hz))])
    samples = int(sample_rate * seconds)
    cmd = [
        "hackrf_transfer", "-r", TMP_FILE,
        "-f", str(int(freq_hz)),
        "-s", str(int(sample_rate)),
        "-b", str(int(bw)),
        "-g", str(int(gain)),
        "-n", str(samples),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=10)
        if not os.path.exists(TMP_FILE):
            return None
        raw = np.fromfile(TMP_FILE, dtype=np.int8)
        os.remove(TMP_FILE)
        if raw.size < 2:
            return None
        i_vals = raw[0::2].astype(np.float32)
        q_vals = raw[1::2].astype(np.float32)
        return ((i_vals + 1j * q_vals) / 128.0).astype(np.complex64)
    except Exception:
        try:
            os.remove(TMP_FILE)
        except Exception:
            pass
        return None


def sanitize_capture_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)


def save_detection_capture(iq: np.ndarray, result: ScanResult, output_dir: str) -> str:
    out_dir = Path(output_dir or DEFAULT_RECORDINGS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = sanitize_capture_name(
        f"{stamp}_{result.center_freq_hz / 1e6:.3f}MHz_snr{result.snr_db:.1f}"
    )
    npz_path = out_dir / f"{base}.npz"
    cf32_path = out_dir / f"{base}.cf32"
    tmp_npz = npz_path.with_name(f"{npz_path.stem}.tmp.npz")
    tmp_cf32 = cf32_path.with_name(f"{cf32_path.name}.tmp")

    interleaved = np.empty(iq.size * 2, dtype=np.float32)
    interleaved[0::2] = iq.real.astype(np.float32, copy=False)
    interleaved[1::2] = iq.imag.astype(np.float32, copy=False)
    interleaved.tofile(tmp_cf32)
    os.replace(tmp_cf32, cf32_path)

    top_match = result.matches[0] if result.matches else None
    np.savez_compressed(
        tmp_npz,
        iq=iq.astype(np.complex64, copy=False),
        timestamp_utc=np.array(result.timestamp_utc),
        center_freq_hz=np.float64(result.center_freq_hz),
        sample_rate=np.float64(result.sample_rate),
        frontend_bandwidth_hz=np.float64(result.frontend_bandwidth_hz),
        peak_db=np.float32(result.peak_db),
        noise_floor_db=np.float32(result.noise_floor_db),
        snr_db=np.float32(result.snr_db),
        occupied_bw_hz=np.float32(result.occupied_bw_hz),
        top_match_name=np.array(top_match.name if top_match else ""),
        top_match_brand=np.array(top_match.brand if top_match else ""),
        top_match_confidence=np.float32(top_match.confidence if top_match else 0.0),
        top_match_notes=np.array(top_match.notes if top_match else ""),
        cf32_path=np.array(str(cf32_path)),
    )
    os.replace(tmp_npz, npz_path)
    return str(npz_path)


class SoapyCapture:
    def __init__(self, args: dict):
        if not HAVE_SOAPY:
            raise RuntimeError("SoapySDR not available")
        self.dev = SoapySDR.Device(args)
        self.stream = None
        self.sample_rate = None
        self.bandwidth_hz = None
        self.gain = None

    def _setup_stream(self, sample_rate: float, bandwidth_hz: float, gain: int):
        if self.stream is None:
            self.stream = self.dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
            self.dev.activateStream(self.stream)
        if self.sample_rate != sample_rate:
            try:
                self.dev.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
                self.sample_rate = sample_rate
            except Exception:
                pass
        if self.bandwidth_hz != bandwidth_hz:
            try:
                self.dev.setBandwidth(SOAPY_SDR_RX, 0, bandwidth_hz)
                self.bandwidth_hz = bandwidth_hz
            except Exception:
                pass
        if self.gain != gain:
            try:
                self.dev.setGain(SOAPY_SDR_RX, 0, gain)
                self.gain = gain
            except Exception:
                pass

    def capture(self, freq_hz: float, sample_rate: float, bandwidth_hz: float,
                seconds: float, gain: int) -> Optional[np.ndarray]:
        self._setup_stream(sample_rate, bandwidth_hz, gain)
        try:
            self.dev.setFrequency(SOAPY_SDR_RX, 0, freq_hz)
        except Exception:
            return None
        samples = int(sample_rate * seconds)
        buff = np.empty(samples, np.complex64)
        read = 0
        deadline = time.time() + max(1.5, seconds * 2.5)
        while read < samples and time.time() < deadline:
            result = self.dev.readStream(self.stream, [buff[read:]], samples - read, 100000)
            if result.ret > 0:
                read += result.ret
            elif result.ret == SOAPY_SDR_TIMEOUT:
                continue
            else:
                break
        return buff[:read] if read >= 256 else None

    def close(self):
        if self.stream:
            try:
                self.dev.deactivateStream(self.stream)
                self.dev.closeStream(self.stream)
            except Exception:
                pass
            self.stream = None

# =============================================================================
# SCAN THREAD
# =============================================================================

class ScanThread(threading.Thread):
    """
    Background thread that continuously sweeps the configured frequency plan
    and pushes ScanResult objects into a queue for the UI to consume.
    """

    def __init__(
        self,
        freq_plan_hz: np.ndarray,
        sample_rate: float,
        bandwidth_hz: float,
        capture_secs: float,
        gain: int,
        snr_gate_db: float,
        backend: str,
        soapy_args: str,
        result_queue: queue.Queue,
        hop_events: list,
        focus_revisits: int = 1,
        record_detections: bool = True,
        record_dir: str = DEFAULT_RECORDINGS_DIR,
        record_cooldown_s: float = DEFAULT_RECORD_COOLDOWN_S,
    ):
        super().__init__(daemon=True)
        self.freq_plan_hz = freq_plan_hz
        self.sample_rate = sample_rate
        self.bandwidth_hz = bandwidth_hz
        self.capture_secs = capture_secs
        self.gain = gain
        self.snr_gate_db = snr_gate_db
        self.backend = backend
        self.soapy_args = soapy_args
        self.result_queue = result_queue
        self.hop_events = hop_events
        self.running = True
        self.status = "Initializing…"
        self.step = 0
        self.sweep = 0
        self.focus_revisits = max(0, int(focus_revisits))
        self._focus_remaining = 0
        self._focus_freq_hz: Optional[float] = None
        self.record_detections = bool(record_detections)
        self.record_dir = str(record_dir or DEFAULT_RECORDINGS_DIR)
        self.record_cooldown_s = max(0.0, float(record_cooldown_s))
        self._recent_recordings: dict[int, float] = {}

    def _parse_soapy_args(self) -> dict:
        return parse_soapy_args_text(self.soapy_args)

    def _should_record(self, center_freq_hz: float) -> bool:
        if not self.record_detections:
            return False
        now = time.monotonic()
        bucket = int(round(center_freq_hz / max(self.sample_rate, 1.0)))
        last = self._recent_recordings.get(bucket, 0.0)
        if now - last < self.record_cooldown_s:
            return False
        self._recent_recordings[bucket] = now
        return True

    def run(self):
        capture = None
        use_hackrf = False

        # Prefer the native HackRF CLI path in auto mode when a board is present.
        if self.backend == "auto" and have_hackrf():
            use_hackrf = True
            self.status = "HackRF auto-selected"
        elif self.backend == "hackrf":
            if not have_hackrf():
                self.status = "hackrf_transfer not found"
                return
            use_hackrf = True
        elif self.backend in ("auto", "soapy"):
            if HAVE_SOAPY:
                args = self._parse_soapy_args()
                explicit_driver = str(args.get("driver", "")).strip().lower()
                allow_uhd = bool(explicit_driver == "uhd" or self.backend == "soapy")
                all_devs, inv_note = enumerate_soapy_devices_safe(self.soapy_args)
                sdr_devs = _filter_soapy_devices(all_devs, allow_uhd=allow_uhd)
                if sdr_devs:
                    try:
                        chosen = choose_soapy_device(sdr_devs, preferred_driver=explicit_driver) or sdr_devs[0]
                        capture = SoapyCapture(chosen)
                        self.status = f"SoapySDR: {chosen.get('label', chosen.get('driver', 'unknown'))}"
                    except Exception as exc:
                        self.status = f"SoapySDR init failed: {exc}"
                        capture = None
                elif inv_note:
                    self.status = inv_note
            if capture is None:
                if have_hackrf():
                    use_hackrf = True
                else:
                    self.status = "No SDR hardware found — plug in HackRF or SoapySDR device"
                    return

        total = len(self.freq_plan_hz)
        try:
            while self.running:
                focus_step = self._focus_remaining > 0 and self._focus_freq_hz is not None
                if focus_step:
                    freq = float(self._focus_freq_hz)
                    revisit_n = self.focus_revisits - self._focus_remaining + 1
                    self._focus_remaining -= 1
                    self.status = (
                        f"Sweep {self.sweep + 1} | Focus {revisit_n}/{self.focus_revisits} | "
                        f"{freq / 1e6:.3f} MHz"
                    )
                else:
                    freq = float(self.freq_plan_hz[self.step])
                    self.status = (
                        f"Sweep {self.sweep + 1} | Step {self.step + 1}/{total} | "
                        f"{freq / 1e6:.3f} MHz"
                    )
                iq = None
                if use_hackrf:
                    iq = hackrf_capture(freq, self.sample_rate, self.bandwidth_hz,
                                        self.capture_secs, self.gain)
                elif capture:
                    iq = capture.capture(freq, self.sample_rate, self.bandwidth_hz,
                                         self.capture_secs, self.gain)

                if iq is not None and iq.size >= 256:
                    freq_axis, spec_db = compute_spectrum(iq, self.sample_rate)
                    peak_db, noise_floor, snr, offset_hz, occ_bw = extract_metrics(freq_axis, spec_db)
                    detected = snr >= self.snr_gate_db

                    matches = []
                    if detected:
                        matches = fingerprint_signal(freq + offset_hz, occ_bw, snr, peak_db)
                        if self.focus_revisits > 0:
                            self._focus_freq_hz = float(freq + offset_hz)
                            self._focus_remaining = self.focus_revisits
                        # Accumulate hop events for FHSS confidence boosting
                        for m in matches:
                            self.hop_events.append(HopEvent(
                                sig_key=m.sig_key,
                                freq_hz=freq + offset_hz,
                                timestamp=time.monotonic(),
                            ))
                        # Trim old hop events (> 10 seconds)
                        cutoff = time.monotonic() - 10.0
                        while self.hop_events and self.hop_events[0].timestamp < cutoff:
                            self.hop_events.pop(0)
                        # Apply hop confidence boost
                        for m in matches:
                            sig = DRONE_SIGNATURES.get(m.sig_key)
                            if sig and sig.fhss:
                                boost = score_hop_confidence(self.hop_events, m.sig_key)
                                m.confidence = round(min(m.confidence + boost, 0.99), 3)
                                if boost > 0:
                                    m.notes += f"; FHSS hop boost +{boost:.2f}"

                    result = ScanResult(
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        center_freq_hz=freq,
                        peak_db=peak_db,
                        noise_floor_db=noise_floor,
                        snr_db=snr,
                        occupied_bw_hz=occ_bw,
                        detected=detected,
                        freq_axis_hz=freq_axis,
                        spectrum_db=spec_db,
                        sample_rate=self.sample_rate,
                        frontend_bandwidth_hz=self.bandwidth_hz,
                        matches=matches,
                    )
                    if detected and self._should_record(freq + offset_hz):
                        try:
                            result.recording_path = save_detection_capture(iq, result, self.record_dir)
                        except Exception as exc:
                            self.status = f"Recording warning: {exc}"
                    try:
                        self.result_queue.put_nowait(result)
                    except queue.Full:
                        pass

                if not focus_step:
                    self.step += 1
                    if self.step >= total:
                        self.step = 0
                        self.sweep += 1
        finally:
            if capture:
                capture.close()

    def stop(self):
        self.running = False


class WifiScanThread(threading.Thread):
    """Periodically scans nearby WiFi SSIDs for drone patterns via nmcli."""

    def __init__(self, sensor_queue: queue.Queue, interval_s: float = 20.0):
        super().__init__(daemon=True)
        self.sensor_queue    = sensor_queue
        self.interval_s      = interval_s
        self.running         = True
        self.status          = "Starting…"
        self.scan_count      = 0
        self.last_hits: list = []
        self.last_all_networks: list = []
        self.last_scan_time: float   = 0.0   # unix time scan finished
        self.next_scan_time: float   = 0.0

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            self.status = "Scanning WiFi airwaves…"
            try:
                hits, all_nets, err = scan_wifi_networks()
                self.last_hits         = hits
                self.last_all_networks = all_nets
                self.last_scan_time    = time.time()
                self.next_scan_time    = self.last_scan_time + self.interval_s
                self.scan_count       += 1
                stamp_utc = datetime.now(timezone.utc).isoformat()
                for h in hits:
                    try:
                        self.sensor_queue.put_nowait({"source": "wifi", "timestamp_utc": stamp_utc, **h})
                    except queue.Full:
                        pass
                if err:
                    self.status = f"WiFi: {err}"
                elif hits:
                    self.status = f"WiFi scan #{self.scan_count}: 🚁 {len(hits)} drone(s) of {len(all_nets)}"
                else:
                    self.status = f"WiFi scan #{self.scan_count}: {len(all_nets)} networks, no drones"
            except Exception as exc:
                self.status = f"WiFi scan error: {exc}"
            deadline = time.time() + self.interval_s
            while self.running and time.time() < deadline:
                time.sleep(0.5)


class BleScanThread(threading.Thread):
    """Periodically scans BLE for Remote ID / DJI beacons."""

    def __init__(self, sensor_queue: queue.Queue, scan_duration: float = 5.0,
                 interval_s: float = 20.0):
        super().__init__(daemon=True)
        self.sensor_queue = sensor_queue
        self.scan_duration = scan_duration
        self.interval_s = interval_s
        self.running = True
        self.status = "Starting…"
        self.last_devices: list = []

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            self.status = f"BLE scan ({self.scan_duration:.0f} s)…"
            try:
                devices, err = run_ble_scan(duration=self.scan_duration)
                self.last_devices = devices
                stamp_utc = datetime.now(timezone.utc).isoformat()
                for dev in devices:
                    try:
                        self.sensor_queue.put_nowait({"source": "ble", "timestamp_utc": stamp_utc, **dev})
                    except queue.Full:
                        pass
                self.status = f"BLE: {len(devices)} beacon(s)" if devices else "BLE: quiet"
            except Exception as exc:
                self.status = f"BLE error: {exc}"
            deadline = time.time() + self.interval_s
            while self.running and time.time() < deadline:
                time.sleep(0.5)

# =============================================================================
# DRONE COMMAND ADAPTERS
# =============================================================================

@dataclass
class CmdResult:
    success: bool
    message: str
    data: dict = field(default_factory=dict)


class BaseAdapter:
    name: str = "Unknown"
    connected: bool = False
    telemetry: dict = None

    def __init__(self):
        self.telemetry = {}

    def connect(self, **kwargs) -> CmdResult:
        return CmdResult(False, "Not implemented")

    def disconnect(self) -> CmdResult:
        return CmdResult(False, "Not implemented")

    def get_status(self) -> dict:
        return {}

    def is_link_alive(self) -> bool:
        return False

    def return_to_home(self) -> CmdResult:
        return CmdResult(False, "Not implemented")

    def land(self) -> CmdResult:
        return CmdResult(False, "Not implemented")

    def hover(self) -> CmdResult:
        return CmdResult(False, "Not implemented")

    def get_battery(self) -> Optional[int]:
        return None

    def takeoff(self) -> CmdResult:
        return CmdResult(False, "Takeoff not supported by this adapter")

    def arm(self) -> CmdResult:
        return CmdResult(False, "Arm not supported by this adapter")

    def disarm(self) -> CmdResult:
        return CmdResult(False, "Disarm not supported by this adapter")

    def set_mode(self, mode: str) -> CmdResult:
        return CmdResult(False, "Mode selection not supported by this adapter")

    def trigger_camera(self) -> CmdResult:
        return CmdResult(False, "Camera trigger not supported by this adapter")

    def set_gimbal(self, pitch_deg: float = 0.0, roll_deg: float = 0.0,
                   yaw_deg: float = 0.0) -> CmdResult:
        return CmdResult(False, "Gimbal control not supported by this adapter")


class TelloAdapter(BaseAdapter):
    name = "DJI Tello"

    def __init__(self):
        self._drone = None
        self.connected = False
        self.telemetry: dict = {}
        self._poll_running = False

    def connect(self, host: str = "192.168.10.1", **_) -> CmdResult:
        if not HAVE_TELLO_SDK:
            return CmdResult(False, "djitellopy not installed — run: pip install djitellopy")
        try:
            self._drone = _TelloSDK(host=host)
            self._drone.connect()
            self.connected = True
            self._start_poll()
            batt = self._drone.get_battery()
            return CmdResult(True, f"Connected. Battery: {batt}%", {"battery": batt})
        except Exception as exc:
            self.connected = False
            return CmdResult(False, str(exc))

    def _start_poll(self):
        self._poll_running = True
        def _poll():
            while self._poll_running and self.connected and self._drone:
                try:
                    self.telemetry = {
                        "battery_%": self._drone.get_battery(),
                        "height_cm": self._drone.get_height(),
                        "temp_lo_C": self._drone.get_temperature()[0] if isinstance(self._drone.get_temperature(), tuple) else self._drone.get_temperature(),
                        "flight_time_s": self._drone.get_flight_time(),
                        "speed_x": self._drone.get_speed_x(),
                        "speed_y": self._drone.get_speed_y(),
                        "speed_z": self._drone.get_speed_z(),
                        "pitch_deg": self._drone.get_pitch(),
                        "roll_deg": self._drone.get_roll(),
                        "yaw_deg": self._drone.get_yaw(),
                        "is_flying": self._drone.is_flying,
                        "barometer_cm": self._drone.get_barometer(),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                except Exception:
                    pass
                time.sleep(0.5)
        threading.Thread(target=_poll, daemon=True).start()

    def is_link_alive(self) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.5)
            sock.sendto(b"command", ("192.168.10.1", 8889))
            data, _ = sock.recvfrom(128)
            return bool(data)
        except Exception:
            return False
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def get_status(self) -> dict:
        return dict(self.telemetry)

    def return_to_home(self) -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            self._drone.land()
            return CmdResult(True, "Tello landing (RTH not supported — landing in place)")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def land(self) -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            self._drone.land()
            return CmdResult(True, "Land command sent")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def hover(self) -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            self._drone.send_rc_control(0, 0, 0, 0)
            return CmdResult(True, "Hover (RC zero) sent")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def get_battery(self) -> Optional[int]:
        return self.telemetry.get("battery_%")

    def takeoff(self) -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            self._drone.takeoff()
            return CmdResult(True, "Takeoff command sent")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def take_photo(self) -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            self._drone.take_picture()
            return CmdResult(True, "Photo taken")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def flip(self, direction: str = "f") -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            self._drone.flip(direction)
            return CmdResult(True, f"Flip {direction!r}")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def set_speed(self, cm_s: int = 50) -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            self._drone.set_speed(int(cm_s))
            return CmdResult(True, f"Speed → {cm_s} cm/s")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def disconnect(self) -> CmdResult:
        self._poll_running = False
        if self._drone:
            try:
                self._drone.end()
            except Exception:
                pass
        self.connected = False
        self._drone = None
        return CmdResult(True, "Disconnected from Tello")


class MAVLinkAdapter(BaseAdapter):
    """MAVLink drone (ArduPilot/PX4) via pymavlink."""
    name = "MAVLink Drone"

    def __init__(self):
        self._master = None
        self.connected = False
        self.telemetry: dict = {}
        self._last_heartbeat = 0.0
        self._poll_running = False

    def connect(self, connection_string: str = "udpin:0.0.0.0:14550", **_) -> CmdResult:
        if not HAVE_MAVLINK:
            return CmdResult(False, "pymavlink not installed — run: pip install pymavlink")
        try:
            self._master = _mavutil.mavlink_connection(connection_string)
            self._master.wait_heartbeat(timeout=6)
            self._last_heartbeat = time.time()
            self.connected = True
            self._start_poll()
            self._request_streams()
            return CmdResult(True, f"Connected (sysid={self._master.target_system})")
        except Exception as exc:
            self.connected = False
            return CmdResult(False, str(exc))

    def _request_streams(self):
        if not self._master:
            return
        try:
            self._master.mav.request_data_stream_send(
                self._master.target_system,
                self._master.target_component,
                _mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10, 1,
            )
        except Exception:
            pass

    def _start_poll(self):
        self._poll_running = True
        def _poll():
            while self._poll_running and self._master:
                try:
                    msg = self._master.recv_match(blocking=False)
                    if msg:
                        mtype = msg.get_type()
                        if mtype == "HEARTBEAT":
                            self._last_heartbeat = time.time()
                            try:
                                self.telemetry["armed"] = bool(
                                    msg.base_mode & _mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                                )
                                self.telemetry["flight_mode"] = _mavutil.mode_string_v10(msg)
                            except Exception:
                                pass
                        elif mtype == "GLOBAL_POSITION_INT":
                            self.telemetry["lat"] = msg.lat / 1e7
                            self.telemetry["lon"] = msg.lon / 1e7
                            self.telemetry["alt_m"] = msg.alt / 1000.0
                            self.telemetry["rel_alt_m"] = msg.relative_alt / 1000.0
                        elif mtype in ("BATTERY_STATUS", "SYS_STATUS"):
                            if hasattr(msg, "battery_remaining") and msg.battery_remaining >= 0:
                                self.telemetry["battery_%"] = msg.battery_remaining
                            if hasattr(msg, "voltage_battery"):
                                self.telemetry["voltage_mV"] = msg.voltage_battery
                        elif mtype == "VFR_HUD":
                            self.telemetry["airspeed_ms"] = msg.airspeed
                            self.telemetry["groundspeed_ms"] = msg.groundspeed
                            self.telemetry["heading_deg"] = msg.heading
                            self.telemetry["altitude_m"] = msg.alt
                            self.telemetry["climb_ms"] = msg.climb
                        elif mtype == "ATTITUDE":
                            self.telemetry["roll_deg"] = round(float(np.degrees(msg.roll)), 1)
                            self.telemetry["pitch_deg"] = round(float(np.degrees(msg.pitch)), 1)
                            self.telemetry["yaw_deg"] = round(float(np.degrees(msg.yaw)), 1)
                        elif mtype == "GPS_RAW_INT":
                            self.telemetry["gps_fix"] = msg.fix_type
                            self.telemetry["satellites"] = msg.satellites_visible
                        elif mtype == "STATUSTEXT":
                            self.telemetry["status_msg"] = msg.text
                        self.telemetry["link_age_s"] = round(time.time() - self._last_heartbeat, 1)
                        self.telemetry["timestamp"] = datetime.now(timezone.utc).isoformat()
                except Exception:
                    pass
                time.sleep(0.05)
        threading.Thread(target=_poll, daemon=True).start()

    def is_link_alive(self) -> bool:
        return self.connected and (time.time() - self._last_heartbeat) < 4.0

    def get_status(self) -> dict:
        d = dict(self.telemetry)
        d["link_alive"] = self.is_link_alive()
        return d

    def return_to_home(self) -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            self._master.set_mode("RTL")
            return CmdResult(True, "RTL (Return-To-Launch) sent")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def land(self) -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            self._master.set_mode("LAND")
            return CmdResult(True, "LAND mode sent")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def hover(self) -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            self._master.set_mode("LOITER")
            return CmdResult(True, "LOITER (hover) mode sent")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def get_battery(self) -> Optional[int]:
        return self.telemetry.get("battery_%")

    def arm(self) -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            self._master.arducopter_arm()
            self._master.motors_armed_wait()
            return CmdResult(True, "Motors ARMED")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def disarm(self) -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            self._master.arducopter_disarm()
            return CmdResult(True, "Motors DISARMED")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def takeoff(self) -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            self._master.mav.command_long_send(
                self._master.target_system,
                self._master.target_component,
                _mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0, 0, 10.0
            )
            return CmdResult(True, "Takeoff sent (target alt: 10 m)")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def set_mode(self, mode: str) -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            self._master.set_mode(mode)
            return CmdResult(True, f"Mode → {mode}")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def trigger_camera(self) -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            self._master.mav.command_long_send(
                self._master.target_system,
                self._master.target_component,
                _mavutil.mavlink.MAV_CMD_DO_DIGICAM_CONTROL,
                0, 0, 0, 0, 1, 0, 0, 0
            )
            return CmdResult(True, "Camera shutter triggered")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def set_gimbal(self, pitch_deg: float = 0.0, roll_deg: float = 0.0,
                   yaw_deg: float = 0.0) -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            self._master.mav.command_long_send(
                self._master.target_system,
                self._master.target_component,
                _mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL,
                0,
                pitch_deg, roll_deg, yaw_deg,
                0, 0, 0,
                _mavutil.mavlink.MAV_MOUNT_MODE_MAVLINK_TARGETING
            )
            return CmdResult(True, f"Gimbal → pitch {pitch_deg}° roll {roll_deg}° yaw {yaw_deg}°")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def disconnect(self) -> CmdResult:
        self._poll_running = False
        if self._master:
            try:
                self._master.close()
            except Exception:
                pass
        self.connected = False
        self._master = None
        return CmdResult(True, "Disconnected from MAVLink drone")


class DJIOSDKAdapter(BaseAdapter):
    """
    DJI Onboard SDK (OSDK) bridge adapter.
    Works with enterprise drones: Matrice 200/300/600, M30, A3/N3 flight controller.
    Requires DJI OSDK C++ bridge running on the companion computer.
    Consumer Mavic/Mini/Air are NOT supported by OSDK — they require Mobile SDK.

    Bridge protocol: newline-delimited JSON over TCP.
    Implement your own bridge using DJI OSDK C++ library:
    https://github.com/dji-sdk/Onboard-SDK
    """
    name = "DJI OSDK (Enterprise)"

    def __init__(self):
        self._sock = None
        self.connected = False
        self.telemetry: dict = {}

    def connect(self, host: str = "127.0.0.1", port: int = 9988, **_) -> CmdResult:
        try:
            self._sock = socket.create_connection((host, port), timeout=3.0)
            self.connected = True
            return CmdResult(True, f"Connected to DJI OSDK bridge at {host}:{port}")
        except Exception as exc:
            return CmdResult(
                False,
                f"OSDK bridge unavailable at {host}:{port}: {exc}. "
                "Start the OSDK C++ bridge on the companion computer first."
            )

    def _send(self, cmd: dict) -> dict:
        if not self._sock:
            return {"error": "not connected"}
        try:
            self._sock.sendall((json.dumps(cmd) + "\n").encode())
            resp = b""
            while b"\n" not in resp:
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
                resp += chunk
            return json.loads(resp.strip())
        except Exception as exc:
            return {"error": str(exc)}

    def is_link_alive(self) -> bool:
        return self._send({"cmd": "ping"}).get("status") == "ok"

    def get_status(self) -> dict:
        r = self._send({"cmd": "get_telemetry"})
        if r.get("status") == "ok":
            self.telemetry = r.get("data", {})
        return dict(self.telemetry)

    def return_to_home(self) -> CmdResult:
        r = self._send({"cmd": "go_home"})
        return CmdResult(r.get("status") == "ok", r.get("message", str(r)))

    def land(self) -> CmdResult:
        r = self._send({"cmd": "land"})
        return CmdResult(r.get("status") == "ok", r.get("message", str(r)))

    def hover(self) -> CmdResult:
        r = self._send({"cmd": "hover"})
        return CmdResult(r.get("status") == "ok", r.get("message", str(r)))

    def get_battery(self) -> Optional[int]:
        return self.telemetry.get("battery_percent")

    def disconnect(self) -> CmdResult:
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self.connected = False
        self._sock = None
        return CmdResult(True, "Disconnected from DJI OSDK bridge")


class ParrotAdapter(BaseAdapter):
    """
    Parrot drone adapter (ANAFI via Olympe SDK, Bebop 2 via pyparrot).
    Install: pip install parrot-olympe   OR   pip install pyparrot
    """
    name = "Parrot Drone"

    def __init__(self):
        self._drone = None
        self._sdk = None
        self.connected = False
        self.telemetry: dict = {}

    def connect(self, host: str = "192.168.42.1", drone_type: str = "anafi", **_) -> CmdResult:
        # Try Olympe first (ANAFI)
        try:
            import olympe
            self._drone = olympe.Drone(host)
            ok = self._drone.connect()
            if ok:
                self._sdk = "olympe"
                self.connected = True
                return CmdResult(True, f"Connected to Parrot ANAFI at {host} via Olympe")
            return CmdResult(False, "Olympe connect returned False")
        except ImportError:
            pass
        # Fall back to pyparrot (Bebop)
        try:
            from pyparrot.Bebop import Bebop
            self._drone = Bebop()
            ok = self._drone.connect(10)
            if ok:
                self._sdk = "pyparrot"
                self.connected = True
                return CmdResult(True, "Connected to Parrot Bebop via pyparrot")
            return CmdResult(False, "pyparrot connect returned False")
        except ImportError:
            pass
        return CmdResult(
            False,
            "Parrot SDK not installed. Run:  pip install parrot-olympe  "
            "(ANAFI)  OR  pip install pyparrot  (Bebop)"
        )

    def is_link_alive(self) -> bool:
        try:
            s = socket.create_connection(("192.168.42.1", 9988), timeout=0.5)
            s.close()
            return True
        except Exception:
            return False

    def get_status(self) -> dict:
        return dict(self.telemetry)

    def return_to_home(self) -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            if self._sdk == "olympe":
                import olympe.messages.ardrone3.Piloting as P
                self._drone(P.NavigateHome(1)).wait()
            else:
                self._drone.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=0, duration=1)
            return CmdResult(True, "Return-to-home / hover sent")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def land(self) -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            if self._sdk == "olympe":
                import olympe.messages.ardrone3.Piloting as P
                self._drone(P.Landing()).wait()
            else:
                self._drone.safe_land(10)
            return CmdResult(True, "Land command sent")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def hover(self) -> CmdResult:
        if not self.connected:
            return CmdResult(False, "Not connected")
        try:
            if self._sdk == "olympe":
                import olympe.messages.ardrone3.Piloting as P
                self._drone(P.PCMD(0, 0, 0, 0, 0, 0))
            else:
                self._drone.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=0, duration=0.1)
            return CmdResult(True, "Hover sent")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def get_battery(self) -> Optional[int]:
        return self.telemetry.get("battery_%")

    def disconnect(self) -> CmdResult:
        if self._drone:
            try:
                self._drone.disconnect()
            except Exception:
                pass
        self.connected = False
        self._drone = None
        return CmdResult(True, "Disconnected from Parrot drone")


class GenericUDPAdapter(BaseAdapter):
    """
    Generic UDP command socket for drones with custom/open UDP APIs
    (e.g., Skydio, custom firmware, research platforms).
    """
    name = "Generic UDP"

    def __init__(self):
        self._sock = None
        self.connected = False
        self.telemetry: dict = {}
        self._host = ""
        self._port = 0

    def connect(self, host: str = "192.168.77.1", port: int = 50051, **_) -> CmdResult:
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.settimeout(1.0)
            self._host = host
            self._port = port
            self._sock.sendto(b'{"cmd":"ping"}', (host, port))
            resp, _ = self._sock.recvfrom(4096)
            self.connected = True
            return CmdResult(True, f"UDP socket open to {host}:{port} — response: {resp[:64]}")
        except socket.timeout:
            self.connected = True   # no response but port is open
            return CmdResult(True, f"UDP socket open to {host}:{port} (no ping response)")
        except Exception as exc:
            return CmdResult(False, str(exc))

    def _send(self, payload: bytes) -> Optional[bytes]:
        if not self._sock:
            return None
        try:
            self._sock.sendto(payload, (self._host, self._port))
            resp, _ = self._sock.recvfrom(4096)
            return resp
        except Exception:
            return None

    def is_link_alive(self) -> bool:
        return self._send(b'{"cmd":"ping"}') is not None

    def get_status(self) -> dict:
        resp = self._send(b'{"cmd":"status"}')
        if resp:
            try:
                self.telemetry = json.loads(resp)
            except Exception:
                pass
        return dict(self.telemetry)

    def return_to_home(self) -> CmdResult:
        resp = self._send(b'{"cmd":"rth"}')
        return CmdResult(bool(resp), resp.decode(errors="replace") if resp else "No response")

    def land(self) -> CmdResult:
        resp = self._send(b'{"cmd":"land"}')
        return CmdResult(bool(resp), resp.decode(errors="replace") if resp else "No response")

    def hover(self) -> CmdResult:
        resp = self._send(b'{"cmd":"hover"}')
        return CmdResult(bool(resp), resp.decode(errors="replace") if resp else "No response")

    def get_battery(self) -> Optional[int]:
        return self.telemetry.get("battery")

    def disconnect(self) -> CmdResult:
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self.connected = False
        self._sock = None
        return CmdResult(True, "UDP socket closed")


ADAPTER_MAP: dict[str, type] = {
    "tello": TelloAdapter,
    "mavlink": MAVLinkAdapter,
    "dji_osdk": DJIOSDKAdapter,
    "parrot": ParrotAdapter,
    "generic_udp": GenericUDPAdapter,
}

ADAPTER_LABELS = {
    "tello": "DJI Tello (djitellopy)",
    "mavlink": "MAVLink (ArduPilot / PX4)",
    "dji_osdk": "DJI OSDK Enterprise Bridge",
    "parrot": "Parrot (Olympe / pyparrot)",
    "generic_udp": "Generic UDP",
}

# =============================================================================
# HELPERS
# =============================================================================

def human_freq(hz: float) -> str:
    if hz >= 1e9:
        return f"{hz / 1e9:.3f} GHz"
    if hz >= 1e6:
        return f"{hz / 1e6:.3f} MHz"
    if hz >= 1e3:
        return f"{hz / 1e3:.1f} kHz"
    return f"{hz:.0f} Hz"

def human_bw(hz: float) -> str:
    if hz >= 1e6:
        return f"{hz / 1e6:.1f} MHz"
    if hz >= 1e3:
        return f"{hz / 1e3:.0f} kHz"
    return f"{hz:.0f} Hz"


def prefill_command_panel(adapter_type: str, host: str = "", port: int = 0,
                          connection_string: str = ""):
    st.session_state["add_adapter_type"] = adapter_type
    if host:
        st.session_state["add_host"] = host
    if port:
        st.session_state["add_port"] = int(port)
    if connection_string:
        st.session_state["add_conn_str"] = connection_string


def direct_connect_spec_for_match(m: DroneMatch) -> Optional[dict]:
    if not m.wifi_ip:
        return None
    if m.adapter in {"tello", "parrot", "generic_udp"}:
        return {"adapter": m.adapter, "host": m.wifi_ip, "port": m.wifi_port}
    return None


def command_guidance_for_match(m: DroneMatch) -> dict:
    direct = direct_connect_spec_for_match(m)
    if direct:
        label = ADAPTER_LABELS.get(direct["adapter"], direct["adapter"])
        return {
            "title": "Direct command path available",
            "detail": f"Use {label} at {direct['host']}:{direct['port']}.",
            "direct": direct,
            "prefill": direct,
        }
    if m.adapter == "mavlink":
        if m.brand == "DJI":
            return {
                "title": "Passive DJI detection only",
                "detail": (
                    "Consumer DJI OcuSync detections do not expose a direct Python control path. "
                    "Use the RC, DJI mobile app, or an enterprise OSDK bridge if the aircraft supports it."
                ),
                "direct": None,
                "prefill": {"adapter": "mavlink", "connection_string": "udpin:0.0.0.0:14550"},
            }
        return {
            "title": "Use a MAVLink backup link",
            "detail": (
                "This looks like a MAVLink telemetry/RC path. Use the MAVLink adapter and a connection "
                "string such as udpin:0.0.0.0:14550 or your companion-computer endpoint."
            ),
            "direct": None,
            "prefill": {"adapter": "mavlink", "connection_string": "udpin:0.0.0.0:14550"},
        }
    if m.adapter == "dji_osdk":
        return {
            "title": "Enterprise bridge required",
            "detail": (
                "This signature maps to DJI enterprise gear. Control requires a reachable DJI OSDK bridge, "
                "not just passive RF detection."
            ),
            "direct": None,
            "prefill": {"adapter": "dji_osdk", "host": "127.0.0.1", "port": 9988},
        }
    return {
        "title": "Manual follow-up required",
        "detail": (
            "The app identified likely RF activity, but it does not have enough transport details to auto-connect. "
            "Use the Command tab with the right adapter once you are on the drone network or know the control endpoint."
        ),
        "direct": None,
        "prefill": None,
    }


def command_guidance_for_wifi_hit(hit: dict) -> dict:
    hint = lookup_wifi_fingerprint(hit.get("brand", ""), hit.get("model", ""))
    if hint:
        direct = None
        if hint["adapter"] in {"tello", "parrot", "generic_udp"}:
            direct = {"adapter": hint["adapter"], "host": hint["ip"], "port": hint["port"]}
        label = ADAPTER_LABELS.get(hint["adapter"], hint["adapter"])
        return {
            "title": "WiFi control path likely available",
            "detail": (
                f"Join SSID `{hit.get('ssid', '?')}` first, then use {label} at "
                f"{hint['ip']}:{hint['port']}."
            ),
            "direct": direct,
            "prefill": {"adapter": hint["adapter"], "host": hint["ip"], "port": hint["port"]},
        }
    return {
        "title": "Drone WiFi seen, but no default control mapping",
        "detail": (
            f"SSID `{hit.get('ssid', '?')}` matches a drone pattern. Join that network, then use "
            "Net Hunter or the Command tab to probe the control endpoint."
        ),
        "direct": None,
        "prefill": None,
    }


def load_recording_into_inspector(recording_path: str):
    path = Path(recording_path)
    if not path.exists():
        raise FileNotFoundError(f"Recording not found: {path}")
    with np.load(path, allow_pickle=False) as npz:
        st.session_state.inspector_iq = npz["iq"].astype(np.complex64)
        st.session_state.inspector_fs = float(npz.get("sample_rate", 2.4e6))
        st.session_state.inspector_cf_hz = float(npz.get("center_freq_hz", 2.437e9))
        st.session_state.inspector_decode = None

def parse_range(text: str):
    raw = text.strip().lower().replace(" ", "")
    def _hz(s):
        mult = {"g": 1e9, "m": 1e6, "k": 1e3}.get(s[-1], 1.0)
        return float(s[:-1] if s[-1] in "gmk" else s) * mult
    try:
        left, right = raw.split("-", 1)
        return _hz(left), _hz(right)
    except Exception:
        return 2400e6, 2484e6

def build_freq_plan(start: float, end: float, step: float) -> np.ndarray:
    s, e = sorted([start, end])
    step = max(step, 1.0)
    n = int(np.floor((e - s) / step)) + 1
    if n > 50000:
        raise ValueError("Too many steps — increase step size or narrow range")
    return s + np.arange(n, dtype=np.float64) * step


def interleave_freq_plans(plans: list[np.ndarray]) -> np.ndarray:
    """Round-robin multiple per-band plans so each band gets revisited quickly."""
    if not plans:
        return np.array([], dtype=np.float64)
    total = sum(int(plan.size) for plan in plans)
    out = np.empty(total, dtype=np.float64)
    pos = 0
    max_len = max(int(plan.size) for plan in plans)
    for idx in range(max_len):
        for plan in plans:
            if idx < plan.size:
                out[pos] = plan[idx]
                pos += 1
    return out[:pos]


def build_freq_plan_from_text(text: str, step_hz: float, strategy: str = "auto") -> np.ndarray:
    """Build a frequency plan from a range string.

    Supports comma-separated bands for multi-band sweeps, e.g.:
      '433m-435m,902m-928m,2.400g-2.484g,5.725g-5.850g'
    """
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty frequency range")
    plans = [build_freq_plan(*parse_range(p), step_hz) for p in parts]
    if strategy == "auto":
        strategy = "interleaved" if len(plans) > 1 else "sequential"
    if strategy == "interleaved":
        combined = interleave_freq_plans(plans)
    elif strategy == "sequential":
        combined = np.concatenate(plans)
    else:
        raise ValueError(f"Unknown scan strategy: {strategy}")
    if combined.size > 50000:
        raise ValueError(f"Too many steps ({combined.size}) — increase step size or narrow range")
    return combined

def confidence_color(c: float) -> str:
    if c >= 0.70:
        return "🟢"
    if c >= 0.45:
        return "🟡"
    return "🔴"


def build_target_picture(detections: list[ScanResult], sensor_hits: list[dict]) -> list[dict]:
    """Merge recent SDR, WiFi, and BLE observations into a compact operator picture."""
    leads: dict[str, dict] = {}

    def _ensure(label: str, brand: str, source: str) -> dict:
        key = f"{brand}:{label}".lower()
        lead = leads.get(key)
        if lead is None:
            lead = {
                "label": label,
                "brand": brand or "Unknown",
                "sources": set(),
                "best_conf": 0,
                "last_seen": "",
                "freq": "",
                "next_action": "",
            }
            leads[key] = lead
        return lead

    for r in detections[-80:]:
        if not r.matches:
            continue
        m = r.matches[0]
        lead = _ensure(m.name, m.brand, "sdr")
        lead["sources"].add("SDR")
        lead["best_conf"] = max(lead["best_conf"], int(m.confidence * 100))
        lead["last_seen"] = max(lead["last_seen"], r.timestamp_utc)
        lead["freq"] = human_freq(r.center_freq_hz)
        lead["next_action"] = command_guidance_for_match(m)["title"]

    for hit in sensor_hits[-120:]:
        if hit.get("source") == "wifi":
            label = f"{hit.get('brand', '?')} {hit.get('model', '')}".strip()
            lead = _ensure(label, hit.get("brand", "Unknown"), "wifi")
            lead["sources"].add("WiFi")
            lead["last_seen"] = max(lead["last_seen"], str(hit.get("timestamp_utc", "")))
            lead["freq"] = hit.get("band", lead["freq"])
            lead["next_action"] = command_guidance_for_wifi_hit(hit)["title"]
        elif hit.get("source") == "ble":
            label = hit.get("name") or hit.get("type") or "Remote ID beacon"
            brand = hit.get("brand") or hit.get("type") or "BLE"
            lead = _ensure(label, brand, "ble")
            lead["sources"].add("BLE")
            lead["last_seen"] = max(lead["last_seen"], str(hit.get("timestamp_utc", "")))
            rssi = hit.get("rssi")
            lead["next_action"] = "Correlate with WiFi/SDR target" if rssi is None else f"BLE proximity hit ({rssi} dBm)"

    rows = []
    for lead in leads.values():
        last_seen = lead["last_seen"]
        if "T" in last_seen:
            last_seen = last_seen[11:19] + "Z"
        rows.append({
            "Lead": lead["label"],
            "Brand": lead["brand"],
            "Sensors": " + ".join(sorted(lead["sources"])) or "—",
            "Best conf %": lead["best_conf"] or "—",
            "Last seen": last_seen or "live",
            "Freq/Band": lead["freq"] or "—",
            "Next action": lead["next_action"] or "Review manually",
            "_score": (len(lead["sources"]) * 1000) + int(lead["best_conf"]),
        })
    rows.sort(key=lambda row: (row["_score"], row["Last seen"]), reverse=True)
    for row in rows:
        row.pop("_score", None)
    return rows

# =============================================================================
# STREAMLIT UI
# =============================================================================

QUICK_BANDS = {
    "DJI 2.4 GHz (OcuSync/Tello)":               ("2.400g-2.484g", 10e6, 20e6),
    "DJI 5.8 GHz (OcuSync)":                     ("5.725g-5.850g", 20e6, 20e6),
    "MAVLink 915 MHz":                            ("902m-928m",      0.5e6, 1e6),
    "MAVLink 433 MHz":                            ("433m-435m",      0.5e6, 0.5e6),
    "Full Sub-GHz":                               ("400m-960m",      2e6,   5e6),
    "Full 2.4 GHz + 5.8 GHz":                    ("2.400g-5.850g",  20e6, 100e6),
    # ── Kitchen Sink presets (HackRF required) ────────────────────────────────
    # Fast: only the 4 drone-relevant bands — quick sweep, high hit-rate
    "🍳 Kitchen Sink – Fast (433/915/2.4/5.8)":  (
        "433m-435m,902m-928m,2.400g-2.484g,5.725g-5.850g", 2e6, 10e6
    ),
    # Full: HackRF's entire usable range 1 MHz → 6 GHz at 20 MHz steps (~300 steps/sweep)
    "🍳 Kitchen Sink – Full (HackRF 1M–6G)":     ("1m-6g",          20e6, 20e6),
    "Custom…":                                    ("",               2.4e6, None),
}

def init_session():
    defaults = {
        "scan_thread": None,
        "scan_queue": queue.Queue(maxsize=500),
        "hop_events": [],
        "scanning": False,
        "results": deque(maxlen=500),
        "detections": [],
        "waterfall": None,
        "wf_freqs": None,
        "latest_recording_path": "",
        "adapters": {},
        "cmd_log": [],
        "last_refresh": 0.0,
        "sweep_count": 0,
        "wifi_thread": None,
        "ble_thread": None,
        "sensor_queue": queue.Queue(maxsize=200),
        "sensor_hits": [],
        # Net Hunter
        "nh_subnet_results": [],
        "nh_arp_results": [],
        "nh_mavlink_results": [],
        "nh_port_scan_results": {},
        "nh_http_results": {},
        "nh_onvif_results": {},
        "nh_mdns_results": [],
        "nh_wifi_all": [],
        "nh_wifi_drones": [],
        # Camera / IoT inventory
        "camera_onvif_results": [],
        "camera_mdns_results": [],
        "camera_stream_candidates": [],
        # Packet Sniffer
        "sniffer_thread": None,
        "sniffer_queue": queue.Queue(maxsize=500),
        "sniffer_packets": [],
        # Signal Inspector
        "inspector_iq": None,
        "inspector_fs": 2.4e6,
        "inspector_cf_hz": 2.437e9,
        "inspector_decode": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def drain_queue():
    """Pull pending results from SDR, WiFi, and BLE queues into session state."""
    # Drain WiFi / BLE sensor queue
    sq: queue.Queue = st.session_state.sensor_queue
    while True:
        try:
            hit = sq.get_nowait()
            st.session_state.sensor_hits.append(hit)
        except queue.Empty:
            break
    # Keep only the 200 most recent sensor hits
    if len(st.session_state.sensor_hits) > 200:
        st.session_state.sensor_hits = st.session_state.sensor_hits[-200:]

    # Drain SDR scan queue
    q: queue.Queue = st.session_state.scan_queue
    new_results = []
    while True:
        try:
            r: ScanResult = q.get_nowait()
            new_results.append(r)
            st.session_state.results.append(r)
            if r.detected and r.matches:
                st.session_state.detections.append(r)
            if r.recording_path:
                st.session_state.latest_recording_path = r.recording_path
        except queue.Empty:
            break

    if not new_results:
        return

    # Update waterfall: each sweep step adds a row for its frequency bin
    # We accumulate a 2D power matrix [time × freq_bin]
    if st.session_state.waterfall is None or st.session_state.wf_freqs is None:
        # First result: initialize waterfall dimensions
        r0 = new_results[0]
        st.session_state.wf_freqs = r0.center_freq_hz + r0.freq_axis_hz
        st.session_state.waterfall = np.full(
            (WATERFALL_ROWS, len(r0.freq_axis_hz)), -140.0, dtype=np.float32
        )

    for r in new_results:
        wf = st.session_state.waterfall
        wf = np.roll(wf, -1, axis=0)
        # Snap this result's spectrum into the waterfall
        wf[-1, :] = np.interp(
            st.session_state.wf_freqs,
            r.center_freq_hz + r.freq_axis_hz,
            r.spectrum_db,
            left=-140.0,
            right=-140.0,
        )
        st.session_state.waterfall = wf


def build_spectrum_fig(results: list) -> go.Figure:
    fig = sp.make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        shared_xaxes=True,
        vertical_spacing=0.04,
    )
    if not results:
        fig.update_layout(height=420, paper_bgcolor="#07111a", plot_bgcolor="#0c1722")
        return fig

    last: ScanResult = results[-1]
    freqs_mhz = (last.center_freq_hz + last.freq_axis_hz) / 1e6

    # Spectrum trace
    fig.add_trace(go.Scatter(
        x=freqs_mhz, y=last.spectrum_db,
        mode="lines", line=dict(color="#38bdf8", width=1.2),
        name="PSD",
    ), row=1, col=1)

    # Mark peak
    peak_idx = int(np.argmax(last.spectrum_db))
    fig.add_trace(go.Scatter(
        x=[freqs_mhz[peak_idx]], y=[last.spectrum_db[peak_idx]],
        mode="markers",
        marker=dict(color="#f59e0b", size=8, symbol="triangle-up"),
        name=f"Peak {last.peak_db:.1f} dB",
    ), row=1, col=1)

    # Noise floor reference line
    fig.add_hline(
        y=last.noise_floor_db, line_dash="dot",
        line_color="#64748b", annotation_text="Noise floor",
        row=1, col=1,
    )

    # SNR gate
    if last.snr_db > 0:
        gate_db = last.noise_floor_db + (last.snr_db * 0)   # just visual marker
        # Mark SNR gate threshold
        fig.add_hline(
            y=last.noise_floor_db + st.session_state.get("snr_gate", 12.0),
            line_dash="dash", line_color="#ef4444",
            annotation_text="SNR gate",
            row=1, col=1,
        )

    # Waterfall
    wf = st.session_state.waterfall
    wf_freqs = st.session_state.wf_freqs
    if wf is not None and wf_freqs is not None:
        vmin = float(np.percentile(wf, 5))
        vmax = float(np.percentile(wf, 97))
        fig.add_trace(go.Heatmap(
            z=wf,
            x=wf_freqs / 1e6,
            colorscale="Magma",
            zmin=vmin, zmax=vmax,
            showscale=False,
            name="Waterfall",
        ), row=2, col=1)

    fig.update_layout(
        height=480,
        paper_bgcolor="#07111a",
        plot_bgcolor="#0c1722",
        font=dict(color="#cbd5e1"),
        margin=dict(l=60, r=20, t=30, b=40),
        legend=dict(bgcolor="#07111a", font=dict(size=10)),
        xaxis2=dict(title="Frequency (MHz)", color="#cbd5e1", gridcolor="#1e293b"),
        yaxis=dict(title="Power (dB)", color="#cbd5e1", gridcolor="#1e293b"),
        yaxis2=dict(title="Time →", color="#cbd5e1", showticklabels=False),
    )
    return fig


def render_detection_card(r: ScanResult, idx: int):
    """Render a single detection result as a styled card."""
    top_match: DroneMatch = r.matches[0]
    icon = confidence_color(top_match.confidence)
    plan = command_guidance_for_match(top_match)
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            st.markdown(f"**{icon} {top_match.name}** ({top_match.brand})")
            st.caption(f"{r.timestamp_utc[:19]}Z")
        with col2:
            st.metric("Frequency", human_freq(r.center_freq_hz))
            st.metric("SNR", f"{r.snr_db:.1f} dB")
        with col3:
            st.metric("Confidence", f"{top_match.confidence * 100:.0f}%")
            st.metric("BW", human_bw(r.occupied_bw_hz))
        st.caption(plan["detail"])
        action_cols = st.columns(3)
        with action_cols[0]:
            direct = plan.get("direct")
            if direct:
                if st.button("Connect now", key=f"connect_now_{idx}"):
                    _connect_from_match(top_match)
            else:
                prefill = plan.get("prefill")
                if prefill and st.button("Prefill command panel", key=f"prefill_cmd_{idx}"):
                    prefill_command_panel(
                        prefill["adapter"],
                        host=prefill.get("host", ""),
                        port=prefill.get("port", 0),
                        connection_string=prefill.get("connection_string", ""),
                    )
                    st.success("Command panel prefilled from this detection.")
        with action_cols[1]:
            if r.recording_path and st.button("Load clip in inspector", key=f"load_clip_{idx}"):
                try:
                    load_recording_into_inspector(r.recording_path)
                    st.success(f"Loaded {Path(r.recording_path).name} into Signal Inspector.")
                except Exception as exc:
                    st.error(f"Recording load failed: {exc}")
        with action_cols[2]:
            if r.recording_path and Path(r.recording_path).exists():
                st.download_button(
                    "Download .npz",
                    data=Path(r.recording_path).read_bytes(),
                    file_name=Path(r.recording_path).name,
                    mime="application/octet-stream",
                    key=f"dl_npz_{idx}",
                )
        with st.expander("All matches / notes"):
            for m in r.matches:
                st.markdown(
                    f"- **{m.name}** — conf={m.confidence:.2f} — {m.notes}"
                )
            if r.recording_path:
                st.markdown(f"Saved IQ clip: `{Path(r.recording_path).name}`")


def connect_direct_target(adapter_key: str, host: str, port: int, label: str):
    adapter_cls = ADAPTER_MAP.get(adapter_key)
    if not adapter_cls:
        st.error(f"No adapter for '{adapter_key}'")
        return
    key = f"{adapter_key}_{host}_{port}"
    if key not in st.session_state.adapters:
        st.session_state.adapters[key] = adapter_cls()
    adapter = st.session_state.adapters[key]
    result = adapter.connect(host=host, port=port)
    if result.success:
        st.success(result.message)
    else:
        st.error(result.message)
    st.session_state.cmd_log.append(
        f"{datetime.now(timezone.utc).isoformat()[:19]}Z  CONNECT  {label}: {result.message}"
    )


def _connect_from_match(m: DroneMatch):
    direct = direct_connect_spec_for_match(m)
    if not direct:
        plan = command_guidance_for_match(m)
        st.error(plan["detail"])
        prefill = plan.get("prefill")
        if prefill:
            prefill_command_panel(
                prefill["adapter"],
                host=prefill.get("host", ""),
                port=prefill.get("port", 0),
                connection_string=prefill.get("connection_string", ""),
            )
        return
    connect_direct_target(direct["adapter"], direct["host"], direct["port"], m.name)


def render_adapter_panel(label: str, adapter: BaseAdapter):
    """Render telemetry and command panel for a connected adapter."""
    status = adapter.get_status()
    alive = adapter.is_link_alive()
    is_mav = isinstance(adapter, MAVLinkAdapter)
    is_tel = isinstance(adapter, TelloAdapter)

    link_col = "#16a34a" if alive else "#dc2626"
    link_txt = "🟢 LINK ALIVE" if alive else "🔴 LINK LOST"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">'
        f'<span style="font-size:1.1rem;font-weight:700;color:#e0f2fe">{adapter.name}</span>'
        f'<span style="padding:2px 10px;border-radius:999px;font-size:0.72rem;font-weight:700;'
        f'background:{link_col}22;color:{link_col};border:1px solid {link_col}">{link_txt}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Telemetry ──────────────────────────────────────────────────────────────
    if status:
        priority = ["battery_%", "flight_mode", "armed", "rel_alt_m", "alt_m",
                    "heading_deg", "groundspeed_ms", "airspeed_ms",
                    "lat", "lon", "roll_deg", "pitch_deg", "yaw_deg",
                    "satellites", "gps_fix", "height_cm", "is_flying",
                    "voltage_mV", "climb_ms", "status_msg"]
        ordered = [k for k in priority if k in status]
        rest = [k for k in status if k not in priority and k not in ("link_alive", "timestamp")]
        display_keys = (ordered + rest)[:16]

        cols = st.columns(4)
        for i, k in enumerate(display_keys):
            v = status[k]
            label_nice = k.replace("_", " ").title()
            if k == "battery_%":
                label_nice = "Battery"
                v = f"{v}%"
            elif k in ("lat", "lon"):
                v = f"{v:.6f}"
            elif k in ("alt_m", "rel_alt_m"):
                v = f"{v:.1f} m"
            elif k in ("groundspeed_ms", "airspeed_ms", "climb_ms"):
                v = f"{v:.1f} m/s"
            elif k in ("roll_deg", "pitch_deg", "yaw_deg", "heading_deg"):
                v = f"{v:.1f}°"
            elif k == "voltage_mV":
                label_nice = "Voltage"
                v = f"{v / 1000:.2f} V"
            with cols[i % 4]:
                st.metric(label_nice, str(v))

        # GPS map link
        lat = status.get("lat")
        lon = status.get("lon")
        if lat and lon and abs(lat) > 0.001 and abs(lon) > 0.001:
            st.markdown(
                f"📍 [Open in Google Maps](https://maps.google.com/?q={lat:.6f},{lon:.6f}) · "
                f"[OpenStreetMap](https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}&zoom=16) · "
                f"Coords: `{lat:.6f}, {lon:.6f}`"
            )

        if status.get("status_msg"):
            st.info(f"FC status: {status['status_msg']}")

    st.divider()
    st.markdown("**Safety commands**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("Return to Home", key=f"rth_{label}"):
            r = adapter.return_to_home()
            st.session_state.cmd_log.append(
                f"{datetime.now(timezone.utc).isoformat()[:19]}Z  RTH  {r.message}"
            )
            st.info(r.message)
    with c2:
        if st.button("Land", key=f"land_{label}"):
            r = adapter.land()
            st.session_state.cmd_log.append(
                f"{datetime.now(timezone.utc).isoformat()[:19]}Z  LAND  {r.message}"
            )
            st.info(r.message)
    with c3:
        if st.button("Hover / Loiter", key=f"hover_{label}"):
            r = adapter.hover()
            st.session_state.cmd_log.append(
                f"{datetime.now(timezone.utc).isoformat()[:19]}Z  HOVER  {r.message}"
            )
            st.info(r.message)
    with c4:
        if st.button("Disconnect", key=f"disc_{label}"):
            r = adapter.disconnect()
            st.session_state.cmd_log.append(
                f"{datetime.now(timezone.utc).isoformat()[:19]}Z  DISCONNECT  {r.message}"
            )
            del st.session_state.adapters[label]
            st.rerun()

    # ── Extended commands ──────────────────────────────────────────────────────
    with st.expander("Extended control"):
        st.markdown("**Takeoff & motors**")
        ea, eb, ec = st.columns(3)
        with ea:
            if st.button("Takeoff", key=f"tkof_{label}"):
                r = adapter.takeoff()
                st.session_state.cmd_log.append(
                    f"{datetime.now(timezone.utc).isoformat()[:19]}Z  TAKEOFF  {r.message}"
                )
                (st.success if r.success else st.error)(r.message)
        with eb:
            if st.button("ARM", key=f"arm_{label}", type="primary"):
                r = adapter.arm()
                st.session_state.cmd_log.append(
                    f"{datetime.now(timezone.utc).isoformat()[:19]}Z  ARM  {r.message}"
                )
                (st.success if r.success else st.warning)(r.message)
        with ec:
            if st.button("DISARM", key=f"darm_{label}"):
                r = adapter.disarm()
                st.session_state.cmd_log.append(
                    f"{datetime.now(timezone.utc).isoformat()[:19]}Z  DISARM  {r.message}"
                )
                (st.success if r.success else st.warning)(r.message)

        if is_mav:
            st.markdown("**Flight mode selector**")
            mav_modes = ["STABILIZE", "ALT_HOLD", "LOITER", "AUTO", "GUIDED",
                         "RTL", "LAND", "POSHOLD", "BRAKE", "THROW"]
            fd1, fd2 = st.columns([3, 1])
            chosen_mode = fd1.selectbox("Mode", mav_modes, key=f"mode_sel_{label}",
                                         label_visibility="collapsed")
            if fd2.button("Set Mode", key=f"set_mode_{label}"):
                r = adapter.set_mode(chosen_mode)
                st.session_state.cmd_log.append(
                    f"{datetime.now(timezone.utc).isoformat()[:19]}Z  MODE  {r.message}"
                )
                (st.success if r.success else st.error)(r.message)

            st.markdown("**Gimbal control** (MAVLink mount)")
            gp, gr, gy = st.columns(3)
            g_pitch = gp.slider("Pitch °", -90, 45, 0, key=f"g_pitch_{label}")
            g_roll  = gr.slider("Roll °",  -45, 45, 0, key=f"g_roll_{label}")
            g_yaw   = gy.slider("Yaw °",  -180, 180, 0, key=f"g_yaw_{label}")
            if st.button("Move Gimbal", key=f"gimbal_{label}"):
                r = adapter.set_gimbal(float(g_pitch), float(g_roll), float(g_yaw))
                st.session_state.cmd_log.append(
                    f"{datetime.now(timezone.utc).isoformat()[:19]}Z  GIMBAL  {r.message}"
                )
                (st.success if r.success else st.error)(r.message)

            st.markdown("**Camera**")
            if st.button("Trigger camera shutter", key=f"cam_trig_{label}"):
                r = adapter.trigger_camera()
                st.session_state.cmd_log.append(
                    f"{datetime.now(timezone.utc).isoformat()[:19]}Z  CAM  {r.message}"
                )
                (st.success if r.success else st.error)(r.message)

        if is_tel:
            st.markdown("**Tello extras**")
            tt1, tt2, tt3 = st.columns(3)
            with tt1:
                if st.button("Take photo", key=f"tpho_{label}"):
                    r = adapter.take_photo()
                    (st.success if r.success else st.error)(r.message)
            with tt2:
                flip_dir = st.selectbox("Flip", ["f", "b", "l", "r"],
                                         key=f"flip_dir_{label}",
                                         label_visibility="collapsed")
                if st.button("Flip!", key=f"flip_{label}"):
                    r = adapter.flip(flip_dir)
                    (st.success if r.success else st.error)(r.message)
            with tt3:
                spd = st.slider("Speed cm/s", 10, 100, 50, key=f"tspd_{label}")
                if st.button("Set speed", key=f"setspd_{label}"):
                    r = adapter.set_speed(spd)
                    (st.success if r.success else st.error)(r.message)


# =============================================================================
# NETWORK DISCOVERY  (Net Hunter)
# =============================================================================

_DRONE_SUBNETS = [
    "192.168.10",   # DJI Tello / Mavic RC-WiFi / Hubsan
    "192.168.1",    # Holy Stone / generic
    "192.168.2",    # Autel EVO series
    "192.168.11",   # FIMI X8
    "192.168.42",   # Parrot (all models)
    "192.168.77",   # Skydio 2/X2/3
    "192.168.0",    # generic home router AP
    "192.168.4",    # DJI RC hotspot mode
    "192.168.100",  # DJI RC Pro / Smart Controller
    "192.168.99",   # Yuneec
    "10.0.0",       # companion computer / Raspberry Pi APs
    "10.1.1",       # some enterprise companion builds
]

_DRONE_PROBE_LAST_OCTETS = [1, 2, 10, 100, 254]

_DRONE_PORTS_ALL = [
    80, 443, 554, 4747, 5000, 7070, 8080, 8554, 8888, 8889,
    9988, 11111, 14550, 14551, 50051,
]

_DRONE_SUBNET_BRANDS: dict[str, tuple[str, str]] = {
    "192.168.10": ("DJI",    "Tello / Mavic WiFi-mode"),
    "192.168.42": ("Parrot", "ANAFI / Bebop"),
    "192.168.77": ("Skydio", "2 / X2 / 3"),
    "192.168.2":  ("Autel",  "EVO series"),
    "192.168.11": ("FIMI",   "X8 series"),
    "192.168.100":("DJI",    "RC Pro / Smart Controller"),
    "192.168.99": ("Yuneec", "Typhoon / H520"),
}

_PORT_LABELS: dict[int, str] = {
    80: "HTTP", 443: "HTTPS", 554: "RTSP", 4747: "DroidCam/MJPEG",
    5000: "Flask/video", 7070: "RTSP alt", 8000: "HTTP alt",
    8080: "HTTP alt", 8443: "HTTPS alt", 8554: "RTSP alt",
    8888: "webcam", 8889: "Tello SDK", 8899: "HTTP alt",
    9988: "DJI OSDK bridge", 11111: "Tello video UDP",
    14550: "MAVLink GCS", 14551: "MAVLink 2", 50051: "gRPC/generic",
}

_MAV_MSG_NAMES: dict[int, str] = {
    0: "HEARTBEAT", 1: "SYS_STATUS", 2: "SYSTEM_TIME",
    24: "GPS_RAW_INT", 30: "ATTITUDE", 32: "LOCAL_POSITION_NED",
    33: "GLOBAL_POSITION_INT", 36: "RC_CHANNELS_RAW",
    42: "MISSION_CURRENT", 74: "VFR_HUD",
    76: "COMMAND_ACK", 77: "COMMAND_LONG",
    105: "HIGHRES_IMU", 147: "BATTERY_STATUS", 253: "STATUSTEXT",
}


def format_hex_dump(data: bytes, width: int = 16) -> str:
    """Classic Wireshark-style hex dump: offset | hex | ASCII"""
    lines = []
    for i in range(0, len(data), width):
        chunk = data[i : i + width]
        offset_s = f"{i:04X}"
        hex_s = " ".join(f"{b:02X}" for b in chunk).ljust(width * 3 - 1)
        asc_s = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{offset_s}  {hex_s}  {asc_s}")
    return "\n".join(lines) or "(empty)"


def parse_mavlink_packet(data: bytes) -> Optional[dict]:
    """Try to parse a MAVLink v1 or v2 packet header. Returns None if not MAVLink."""
    if not data:
        return None
    if data[0] == 0xFE and len(data) >= 8:   # v1
        payload_len = data[1]
        return {
            "version": 1, "seq": data[2], "sysid": data[3],
            "compid": data[4], "msgid": data[5], "payload_len": payload_len,
            "msg_name": _MAV_MSG_NAMES.get(data[5], f"#{data[5]}"),
        }
    if data[0] == 0xFD and len(data) >= 10:  # v2
        msgid = data[7] | (data[8] << 8) | (data[9] << 16)
        return {
            "version": 2, "seq": data[4], "sysid": data[5],
            "compid": data[6], "msgid": msgid, "payload_len": data[1],
            "msg_name": _MAV_MSG_NAMES.get(msgid, f"#{msgid}"),
        }
    return None


def parse_tello_packet(data: bytes) -> Optional[dict]:
    try:
        txt = data.decode("ascii", errors="replace").strip()
        if txt.startswith(("ok", "error", "mid:", "x:", "pitch:")):
            return {"type": "sdk", "text": txt[:200]}
    except Exception:
        pass
    return None


def get_network_interfaces() -> list[dict]:
    """List local network interfaces with IP/prefix via `ip -j addr`."""
    ifaces: list[dict] = []
    try:
        out = subprocess.check_output(["ip", "-j", "addr"], timeout=3, stderr=subprocess.DEVNULL)
        import json as _json
        for iface in _json.loads(out):
            name = iface.get("ifname", "")
            for ai in iface.get("addr_info", []):
                if ai.get("family") == "inet":
                    ifaces.append({"name": name, "ip": ai.get("local", ""),
                                   "prefix": ai.get("prefixlen", 24)})
    except Exception:
        try:
            ips = subprocess.check_output(["hostname", "-I"], timeout=3,
                                           stderr=subprocess.DEVNULL).decode().split()
            ifaces = [{"name": "?", "ip": ip, "prefix": 24} for ip in ips]
        except Exception:
            pass
    return ifaces


def arp_neighbors() -> list[dict]:
    """Parse `ip neigh show` for current-subnet devices."""
    devices: list[dict] = []
    try:
        out = subprocess.check_output(["ip", "neigh", "show"], timeout=3,
                                       stderr=subprocess.DEVNULL).decode()
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            ip = parts[0]
            mac = next((parts[i+1] for i, p in enumerate(parts) if p == "lladdr" and i+1 < len(parts)), "")
            state = parts[-1]
            devices.append({"ip": ip, "mac": mac, "state": state})
    except Exception:
        pass
    return devices


def probe_drone_subnets(timeout: float = 0.7) -> list[dict]:
    """
    Probe every known drone gateway IP across 12+ subnets in parallel.
    Returns list of {ip, open_ports, brand, model, subnet}.
    """
    targets: list[tuple[str, int]] = []
    for subnet in _DRONE_SUBNETS:
        for last in _DRONE_PROBE_LAST_OCTETS:
            ip = f"{subnet}.{last}"
            for port in [80, 8889, 14550, 554, 8080, 9988]:
                targets.append((ip, port))

    found: dict[str, dict] = {}
    lock = threading.Lock()

    def _chk(ip: str, port: int):
        try:
            s = socket.create_connection((ip, port), timeout=timeout)
            s.close()
            with lock:
                if ip not in found:
                    subnet = ".".join(ip.split(".")[:3])
                    brand, model = _DRONE_SUBNET_BRANDS.get(subnet, ("Unknown", ""))
                    found[ip] = {"ip": ip, "open_ports": [], "brand": brand,
                                 "model": model, "subnet": subnet}
                found[ip]["open_ports"].append(port)
        except Exception:
            pass

    threads = [threading.Thread(target=_chk, args=t, daemon=True) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout + 0.6)

    return list(found.values())


def port_scan(ip: str, ports: Optional[list[int]] = None,
              timeout: float = 0.6) -> list[int]:
    """TCP port scan of a single IP. Returns sorted list of open ports."""
    open_ports: list[int] = []
    lock = threading.Lock()

    def _chk(port: int):
        try:
            s = socket.create_connection((ip, port), timeout=timeout)
            s.close()
            with lock:
                open_ports.append(port)
        except Exception:
            pass

    tgts = ports or _DRONE_PORTS_ALL
    ts = [threading.Thread(target=_chk, args=(p,), daemon=True) for p in tgts]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=timeout + 0.4)
    return sorted(open_ports)


def listen_mavlink_udp(timeout: float = 4.0,
                       ports: Optional[list[int]] = None) -> list[dict]:
    """Bind UDP sockets and listen for MAVLink heartbeats."""
    ports = ports or [14550, 14551, 14552, 5760, 18570]
    heard: list[dict] = []
    stop = threading.Event()

    def _listen(port: int):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
            sock.settimeout(0.5)
            deadline = time.time() + timeout
            while not stop.is_set() and time.time() < deadline:
                try:
                    data, addr = sock.recvfrom(512)
                    pkt = parse_mavlink_packet(data)
                    if pkt:
                        heard.append({"src": f"{addr[0]}:{addr[1]}",
                                      "bind_port": port, **pkt})
                except socket.timeout:
                    pass
        except Exception:
            pass
        finally:
            try:
                sock.close()
            except Exception:
                pass

    threads = [threading.Thread(target=_listen, args=(p,), daemon=True) for p in ports]
    for t in threads:
        t.start()
    time.sleep(timeout)
    stop.set()
    for t in threads:
        t.join(timeout=1.0)
    return heard


class PacketSnifferThread(threading.Thread):
    """
    Background daemon that binds UDP sockets on common drone ports and
    captures all arriving packets into `out_queue` as formatted dicts.
    """
    def __init__(self, out_queue: queue.Queue,
                 ports: Optional[list[int]] = None,
                 max_packets: int = 500):
        super().__init__(daemon=True)
        self.ports = ports or [8889, 14550, 14551, 11111, 4747, 5000, 8080, 9988]
        self.out_queue = out_queue
        self.max_packets = max_packets
        self.running = False
        self.packet_count = 0
        self.status = "idle"

    def stop(self):
        self.running = False

    def run(self):
        self.running = True
        socks: list[tuple[int, socket.socket]] = []
        for port in self.ports:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", port))
                s.settimeout(0.15)
                socks.append((port, s))
            except Exception:
                pass

        if not socks:
            self.status = "Could not bind any ports (may need root for <1024)"
            return

        bound = [p for p, _ in socks]
        self.status = f"Listening on ports {bound}"
        try:
            while self.running and self.packet_count < self.max_packets:
                for port, sock in socks:
                    try:
                        data, addr = sock.recvfrom(4096)
                    except socket.timeout:
                        continue
                    ts = time.time()
                    pkt: dict = {
                        "ts": ts,
                        "ts_str": time.strftime("%H:%M:%S", time.localtime(ts)),
                        "src_ip": addr[0],
                        "src_port": addr[1],
                        "dst_port": port,
                        "length": len(data),
                        "proto": "UDP",
                        "proto_detail": "",
                        "hex_short": data[:16].hex(" ").upper(),
                        "ascii_short": "".join(chr(b) if 32 <= b < 127 else "."
                                               for b in data[:32]),
                        "hex_dump": format_hex_dump(data[:256]),
                        "raw": data,
                    }
                    # Protocol identification
                    mav = parse_mavlink_packet(data)
                    if mav:
                        pkt["proto"] = "MAVLink"
                        pkt["proto_detail"] = (
                            f"v{mav['version']} sysid={mav['sysid']} "
                            f"compid={mav['compid']} msg={mav['msg_name']}"
                        )
                        pkt["parsed"] = mav
                    elif port == 8889:
                        tel = parse_tello_packet(data)
                        if tel:
                            pkt["proto"] = "Tello"
                            pkt["proto_detail"] = tel.get("text", "")[:80]
                            pkt["parsed"] = tel
                    elif len(data) >= 2 and data[0] == 0xCC:
                        pkt["proto"] = "DJI"
                        pkt["proto_detail"] = f"DJI frame 0x{data[0]:02X}{data[1]:02X}"
                    else:
                        # Try to show printable content
                        try:
                            txt = data[:80].decode("ascii", errors="strict")
                            pkt["proto_detail"] = txt[:60]
                        except Exception:
                            pass

                    try:
                        self.out_queue.put_nowait(pkt)
                    except queue.Full:
                        pass
                    self.packet_count += 1
                    self.status = f"{self.packet_count} pkts captured"
        finally:
            for _, s in socks:
                try:
                    s.close()
                except Exception:
                    pass
            self.status = f"Stopped — {self.packet_count} pkts captured"


# =============================================================================
# IQ SIGNAL ANALYSIS  (Signal Inspector — Scanner.py features ported)
# =============================================================================

def build_constellation_fig(iq: np.ndarray, center_freq_hz: float) -> go.Figure:
    """IQ constellation scatter (phase plane view)."""
    sub = max(1, len(iq) // 3000)
    iq_s = iq[::sub]
    scale = float(np.percentile(np.abs(iq_s), 97)) or 1.0
    ir = iq_s.real / scale
    ii = iq_s.imag / scale
    theta = np.linspace(0, 2 * np.pi, 300)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode="lines", line=dict(color="#1e3a52", width=1), name="unit circle",
        showlegend=False,
    ))
    fig.add_trace(go.Scattergl(
        x=ir, y=ii, mode="markers",
        marker=dict(size=3, color=np.angle(iq_s), colorscale="HSV",
                    showscale=False, opacity=0.55),
        name="IQ",
    ))
    fig.update_layout(
        title=dict(text=f"Constellation — {human_freq(center_freq_hz)}",
                   font=dict(color="#cbd5e1", size=13), x=0.5),
        paper_bgcolor="#07111a", plot_bgcolor="#0c1722",
        font=dict(color="#cbd5e1"), height=340,
        xaxis=dict(title="I", range=[-1.6, 1.6], gridcolor="#1e293b",
                   zeroline=True, zerolinecolor="#334155"),
        yaxis=dict(title="Q", range=[-1.6, 1.6], scaleanchor="x",
                   gridcolor="#1e293b", zeroline=True, zerolinecolor="#334155"),
        margin=dict(l=50, r=20, t=45, b=40),
    )
    return fig


def build_waveform_fig(iq: np.ndarray, sample_rate: float,
                        center_freq_hz: float) -> go.Figure:
    """Time-domain I, Q, and envelope waveform."""
    sub = max(1, len(iq) // 4000)
    iq_s = iq[::sub]
    t_ms = (np.arange(len(iq_s)) * sub / sample_rate) * 1e3
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_ms, y=iq_s.real,
                              mode="lines", line=dict(color="#0ea5e9", width=1), name="I"))
    fig.add_trace(go.Scatter(x=t_ms, y=iq_s.imag,
                              mode="lines", line=dict(color="#f59e0b", width=1), name="Q"))
    fig.add_trace(go.Scatter(x=t_ms, y=np.abs(iq_s),
                              mode="lines", line=dict(color="#22c55e", width=1.5,
                              dash="dot"), name="|IQ|"))
    fig.update_layout(
        title=dict(text=f"Waveform — {human_freq(center_freq_hz)}",
                   font=dict(color="#cbd5e1", size=13), x=0.5),
        paper_bgcolor="#07111a", plot_bgcolor="#0c1722",
        font=dict(color="#cbd5e1"), height=300,
        xaxis=dict(title="Time (ms)", gridcolor="#1e293b"),
        yaxis=dict(title="Amplitude", gridcolor="#1e293b"),
        legend=dict(bgcolor="#07111a"),
        margin=dict(l=60, r=20, t=45, b=40),
    )
    return fig


def build_detail_spectrogram_fig(iq: np.ndarray, sample_rate: float,
                                   center_freq_hz: float) -> go.Figure:
    """Spectrogram (time vs freq) of a single IQ capture."""
    from scipy import signal as _ss2
    nperseg = min(1024, max(256, len(iq) // 24))
    noverlap = int(nperseg * 0.75)
    freqs, times, spec = _ss2.spectrogram(
        iq, fs=sample_rate, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        return_onesided=False, mode="magnitude",
    )
    freqs = np.fft.fftshift(freqs)
    spec = np.fft.fftshift(spec, axes=0)
    spec_db = 20.0 * np.log10(np.maximum(spec, 1e-12))
    freqs_mhz = (freqs + center_freq_hz) / 1e6
    fig = go.Figure(go.Heatmap(
        z=spec_db.T, x=freqs_mhz, y=times * 1e3,
        colorscale="Plasma",
        zmin=float(np.percentile(spec_db, 5)),
        zmax=float(np.percentile(spec_db, 98)),
        showscale=True,
        colorbar=dict(title="dB", tickfont=dict(color="#cbd5e1")),
    ))
    fig.update_layout(
        title=dict(text=f"Spectrogram — {human_freq(center_freq_hz)}",
                   font=dict(color="#cbd5e1", size=13), x=0.5),
        paper_bgcolor="#07111a", plot_bgcolor="#0c1722",
        font=dict(color="#cbd5e1"), height=300,
        xaxis=dict(title="Frequency (MHz)", gridcolor="#1e293b"),
        yaxis=dict(title="Time (ms)", gridcolor="#1e293b"),
        margin=dict(l=60, r=20, t=45, b=40),
    )
    return fig


def _to_binary_series(series: np.ndarray, smooth_w: int = 1) -> np.ndarray:
    if smooth_w > 1 and series.size >= smooth_w:
        k = np.ones(smooth_w, dtype=np.float32) / smooth_w
        s = np.convolve(series.astype(np.float32), k, mode="same")
    else:
        s = series.astype(np.float32)
    thr = (s.max() + s.min()) / 2.0
    return (s > thr).astype(np.uint8)


def _est_sym_samples(binary: np.ndarray) -> Optional[int]:
    transitions = np.where(np.diff(binary))[0]
    if len(transitions) < 4:
        return None
    gaps = np.diff(transitions)
    gaps = gaps[gaps > 0]
    if not len(gaps):
        return None
    med = float(np.median(gaps))
    return None if med < 2 else max(2, int(round(med)))


def decode_iq_quick(iq: np.ndarray, sample_rate: float,
                     mode: str = "auto") -> dict:
    """
    Attempt ASK/FSK/Manchester decode of IQ data.
    Returns a dict with: mode, symbol_rate_hz, hex_preview, ascii_preview,
                         bit_preview, confidence, notes, byte_values (ndarray)
    """
    envelope = np.abs(iq).astype(np.float32)
    if iq.size > 1:
        disc = np.angle(iq[1:] * np.conj(iq[:-1])).astype(np.float32)
    else:
        disc = np.zeros(0, dtype=np.float32)

    series_pool = []
    if mode in ("auto", "ask"):
        series_pool.append((envelope, "ASK/OOK"))
    if mode in ("auto", "fsk"):
        series_pool.append((disc, "2-FSK"))
    if not series_pool:
        series_pool = [(envelope, "ASK/OOK"), (disc, "2-FSK")]

    best: Optional[dict] = None
    for series, label in series_pool:
        if series.size < 512:
            continue
        for sw in (1, 3, 7, 15, 31, 63):
            binary = _to_binary_series(series, sw)
            sym_s = _est_sym_samples(binary)
            if not sym_s:
                continue
            bits = binary[sym_s // 2 :: sym_s]
            if bits.size < 16:
                continue
            n8 = (bits.size // 8) * 8
            if n8 < 8:
                continue
            byte_vals = np.packbits(bits[:n8].reshape(-1, 8), axis=1,
                                     bitorder="big").flatten()
            idle = float(np.mean((byte_vals == 0x00) | (byte_vals == 0xFF)))
            uniq = len(np.unique(byte_vals[:24])) / max(min(len(byte_vals), 24), 1)
            if idle > 0.93 and uniq < 0.12:
                continue
            printable = float(np.mean((byte_vals >= 32) & (byte_vals < 127)))
            conf = min(0.99, 0.30 * min(bits.size / 128, 1.0)
                       + 0.25 * (1 - idle) + 0.25 * uniq + 0.20 * printable)
            if best and conf <= best["confidence"]:
                continue
            best = {
                "mode": label,
                "symbol_rate_hz": sample_rate / sym_s,
                "hex_preview": " ".join(f"{b:02X}" for b in byte_vals[:32]),
                "ascii_preview": "".join(chr(b) if 32 <= b < 127 else "."
                                          for b in byte_vals[:48]),
                "bit_preview": " ".join(
                    "".join(str(b) for b in bits[i:i+8])
                    for i in range(0, min(bits.size, 64), 8)
                ),
                "confidence": conf,
                "notes": (f"{label} smooth={sw} sym={sym_s}samp "
                           f"{bits.size}bits {byte_vals.size}bytes "
                           f"idle={idle:.2f} uniq={uniq:.2f}"),
                "byte_values": byte_vals,
            }

    if best is None:
        return {
            "mode": mode, "symbol_rate_hz": 0.0,
            "hex_preview": "— no decode —", "ascii_preview": "—",
            "bit_preview": "—", "confidence": 0.0,
            "notes": "No usable bitstream found in this capture",
            "byte_values": np.zeros(0, dtype=np.uint8),
        }
    return best


def main():
    st.set_page_config(
        page_title="DroneDetect",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
    /* ── Base ── */
    .stApp { background-color: #07111a; color: #e2e8f0; }
    [data-testid="stSidebar"] { background-color: #0c1a27; border-right: 1px solid #1e3a52; }
    [data-testid="stSidebar"] .stMarkdown p { color: #94a3b8; font-size: 0.82rem; }

    /* ── Header bar ── */
    .dd-header {
        display: flex; align-items: center; gap: 14px;
        padding: 10px 18px; margin-bottom: 12px;
        background: linear-gradient(90deg, #0c2a3f 0%, #071828 100%);
        border-bottom: 2px solid #0ea5e9; border-radius: 6px;
    }
    .dd-header h1 { margin: 0; font-size: 1.4rem; color: #e0f2fe; letter-spacing: 0.04em; }
    .dd-header .dd-live  { background:#16a34a; color:#fff; padding:2px 10px;
                           border-radius:999px; font-size:0.72rem; font-weight:700;
                           animation: pulse 1.4s infinite; }
    .dd-header .dd-idle  { background:#374151; color:#9ca3af; padding:2px 10px;
                           border-radius:999px; font-size:0.72rem; font-weight:700; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.55} }

    /* ── Detection cards ── */
    .det-card {
        border: 1px solid #1e3a52; border-radius: 8px;
        padding: 12px 16px; margin-bottom: 8px;
        background: #0c1e2e;
    }
    .det-card.high  { border-left: 4px solid #16a34a; }
    .det-card.med   { border-left: 4px solid #d97706; }
    .det-card.low   { border-left: 4px solid #dc2626; }
    .det-card .brand { font-size: 0.72rem; color: #64748b; text-transform: uppercase;
                       letter-spacing: 0.08em; margin-bottom: 2px; }
    .det-card .name  { font-size: 1.0rem; font-weight: 700; color: #e0f2fe; }
    .det-card .meta  { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; }
    .det-card .conf-bar {
        height: 4px; border-radius: 2px; margin-top: 8px;
        background: linear-gradient(90deg, #16a34a, #d97706, #dc2626);
    }

    /* ── Sensor status pills ── */
    .sensor-pill {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 3px 10px; border-radius: 999px; font-size: 0.75rem;
        font-weight: 600; margin-right: 6px; margin-bottom: 4px;
    }
    .pill-on  { background: #064e3b; color: #6ee7b7; border: 1px solid #10b981; }
    .pill-off { background: #1f2937; color: #6b7280; border: 1px solid #374151; }

    /* ── Metric tweaks ── */
    .stMetric label { color: #64748b !important; font-size: 0.75rem !important; }
    .stMetric [data-testid="metric-container"] { background: #0c1e2e;
        border: 1px solid #1e3a52; border-radius: 6px; padding: 8px 12px; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { background: #0c1722; gap: 2px; }
    .stTabs [data-baseweb="tab"] { border-radius: 4px 4px 0 0;
        background: #0c1722; color: #64748b; padding: 6px 16px; }
    .stTabs [aria-selected="true"] { background: #0c1e2e !important;
        color: #0ea5e9 !important; border-bottom: 2px solid #0ea5e9; }

    /* ── Sidebar section headers ── */
    .sb-section { font-size: 0.68rem; font-weight: 700; color: #475569;
                  text-transform: uppercase; letter-spacing: 0.1em;
                  margin: 12px 0 4px; padding-top: 8px;
                  border-top: 1px solid #1e3a52; }
    </style>
    """, unsafe_allow_html=True)

    init_session()

    # ── Page header ───────────────────────────────────────────────────────────
    scanning_now = st.session_state.get("scanning", False)
    n_sigs   = len(DRONE_SIGNATURES)
    pill_cls = "dd-live" if scanning_now else "dd-idle"
    pill_txt = "● LIVE" if scanning_now else "● IDLE"
    sdr_dets_hdr = st.session_state.get("detections", [])
    sensor_hits_hdr = st.session_state.get("sensor_hits", [])
    total_hits = len(sdr_dets_hdr) + len(sensor_hits_hdr)
    st.markdown(f"""
    <div class="dd-header">
      <div><h1>📡 DroneDetect</h1></div>
      <span class="{pill_cls}">{pill_txt}</span>
      <span style="margin-left:auto;font-size:0.76rem;color:#475569;">
        {n_sigs} signatures &nbsp;·&nbsp; {total_hits} hits this session
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📡 DroneDetect")

        # Hardware status pills
        h_hackrf = have_hackrf()
        soapy_inventory = cached_sdr_inventory("")
        soapy_devices = _filter_soapy_devices(soapy_inventory.get("devices", []), allow_uhd=True)
        soapy_note = soapy_inventory.get("note", "")
        sdr_device_label = ""
        if soapy_devices:
            d0 = choose_soapy_device(soapy_devices) or soapy_devices[0]
            sdr_device_label = d0.get("label") or d0.get("driver") or ""

        hackrf_pill = "pill-on" if h_hackrf else "pill-off"
        soapy_pill  = "pill-on" if sdr_device_label else "pill-off"
        wifi_pill   = "pill-on"
        tello_pill  = "pill-on" if HAVE_TELLO_SDK else "pill-off"
        mav_pill    = "pill-on" if HAVE_MAVLINK else "pill-off"
        st.markdown(f"""
        <div style="margin:6px 0 10px">
          <span class="sensor-pill {hackrf_pill}">HackRF</span>
          <span class="sensor-pill {soapy_pill}">{'SoapySDR: ' + sdr_device_label[:12] if sdr_device_label else 'SoapySDR'}</span>
          <span class="sensor-pill {wifi_pill}">WiFi</span>
          <span class="sensor-pill {tello_pill}">Tello SDK</span>
          <span class="sensor-pill {mav_pill}">MAVLink</span>
        </div>
        """, unsafe_allow_html=True)
        if soapy_note:
            st.caption(soapy_note)
        elif HAVE_SOAPY and not sdr_device_label and not h_hackrf:
            st.caption("No RF SDR found. FlexRadio: set args `driver=flex`")

        with st.expander("Capability audit"):
            st.caption(
                "What this runtime can actually do right now across RF, WiFi, BLE, camera, and control surfaces."
            )
            st.dataframe(capability_audit(), width="stretch", hide_index=True)

        if soapy_devices:
            with st.expander("Detected SDR inventory"):
                st.dataframe(
                    [
                        {
                            "Driver": row.get("driver", "—"),
                            "Label": row.get("label", "—"),
                            "Serial": row.get("serial", "—"),
                        }
                        for row in soapy_devices[:12]
                    ],
                    width="stretch",
                    hide_index=True,
                )

        st.markdown('<div class="sb-section">Band Preset</div>', unsafe_allow_html=True)
        band_choice = st.selectbox("Preset", list(QUICK_BANDS.keys()), index=0,
                                    label_visibility="collapsed")
        band_range, band_step, band_sr = QUICK_BANDS[band_choice]

        range_text = st.text_input(
            "Frequency range",
            value=band_range if band_range else "2.400g-2.484g",
            help="Single: 2.400g-2.484g  |  Multi-band: 433m-435m,902m-928m,2.4g-2.484g",
        )
        if "," in range_text:
            st.caption("Interleaved sweep mode active.")

        # ── Sensor selection ──────────────────────────────────────────────────
        st.markdown('<div class="sb-section">Active Sensors</div>', unsafe_allow_html=True)
        use_sdr  = st.checkbox("📻 SDR (HackRF / SoapySDR)",  value=True,  key="use_sdr")
        use_wifi = st.checkbox("📶 WiFi chip (SSID scan)",     value=True,  key="use_wifi")
        use_ble  = st.checkbox("🔵 BLE (Remote ID / DJI)",    value=False, key="use_ble")

        sr_options = [0.5e6, 1e6, 2e6, 2.4e6, 5e6, 10e6, 20e6]
        default_sr_idx = sr_options.index(band_sr) if band_sr in sr_options else 3
        with st.expander("Advanced scan options"):
            backend = st.selectbox("Backend", ["auto", "hackrf", "soapy"], index=0)
            soapy_args = st.text_input(
                "SoapySDR args",
                value="",
                placeholder="driver=rtlsdr  /  driver=hackrf  /  driver=flex  /  driver=uhd",
                help="Blank = auto. Auto prefers HackRF when present. Use explicit Soapy mode for USRP/UHD or other specific radios.",
            )
            st.caption("`auto` prefers HackRF and avoids probing unstable UHD paths unless you explicitly select Soapy / `driver=uhd`.")
            sample_rate = st.selectbox(
                "Sample rate",
                sr_options,
                index=default_sr_idx,
                format_func=lambda x: human_bw(x) + "ps",
            )
            step_mhz = st.number_input(
                "Step (MHz)",
                min_value=0.01,
                max_value=100.0,
                value=float(band_step / 1e6),
                step=0.1,
                format="%.2f",
            )
            gc1, gc2 = st.columns(2)
            gain = gc1.slider("Gain dB", 0, 76, 20)
            cap_ms = gc2.slider("Cap ms", 50, 2000, 200)
            snr_gate = st.slider("SNR gate dB", 3, 40, 12)
            record_hits = st.checkbox("💾 Auto-record SDR detections", value=True, key="record_hits")
            record_dir = st.text_input("Recording directory", value=DEFAULT_RECORDINGS_DIR, key="record_dir")
            record_cooldown = st.slider(
                "Record cooldown (s)",
                0,
                30,
                int(DEFAULT_RECORD_COOLDOWN_S),
                key="record_cooldown",
            )
            if use_ble:
                ble_interval = st.slider("BLE interval (s)", 10, 120, 20, key="ble_interval")
            else:
                ble_interval = 20
        st.session_state["snr_gate"] = float(snr_gate)
        st.caption(
            f"{human_bw(sample_rate)}ps · step {step_mhz:.2f} MHz · {cap_ms} ms capture · SNR {snr_gate} dB"
        )

        # ── Scan controls ─────────────────────────────────────────────────────
        col_s, col_p = st.columns(2)
        with col_s:
            start_btn = st.button("▶ Start", width="stretch", type="primary",
                                   disabled=st.session_state.scanning)
        with col_p:
            stop_btn = st.button("⏹ Stop", width="stretch",
                                  disabled=not st.session_state.scanning)

        if start_btn and not st.session_state.scanning:
            st.session_state.sensor_queue  = queue.Queue(maxsize=200)
            st.session_state.sensor_hits   = []

            if use_sdr:
                try:
                    freq_plan = build_freq_plan_from_text(range_text, step_mhz * 1e6)
                    band_count = len([p for p in range_text.split(",") if p.strip()])
                    st.session_state.scan_queue = queue.Queue(maxsize=500)
                    st.session_state.results    = deque(maxlen=500)
                    st.session_state.detections = []
                    st.session_state.waterfall  = None
                    st.session_state.wf_freqs   = None
                    st.session_state.hop_events = []
                    st.session_state.latest_recording_path = ""
                    sdr_t = ScanThread(
                        freq_plan_hz=freq_plan,
                        sample_rate=float(sample_rate),
                        bandwidth_hz=float(sample_rate),
                        capture_secs=float(cap_ms / 1000.0),
                        gain=int(gain),
                        snr_gate_db=float(snr_gate),
                        backend=backend,
                        soapy_args=soapy_args,
                        result_queue=st.session_state.scan_queue,
                        hop_events=st.session_state.hop_events,
                        focus_revisits=2 if band_count > 1 else 1,
                        record_detections=record_hits,
                        record_dir=record_dir,
                        record_cooldown_s=float(record_cooldown),
                    )
                    sdr_t.start()
                    st.session_state.scan_thread = sdr_t
                except ValueError as exc:
                    st.error(str(exc))

            if use_wifi:
                wt = WifiScanThread(st.session_state.sensor_queue, interval_s=20.0)
                wt.start()
                st.session_state.wifi_thread = wt

            if use_ble:
                bt = BleScanThread(st.session_state.sensor_queue,
                                   scan_duration=5.0, interval_s=float(ble_interval))
                bt.start()
                st.session_state.ble_thread = bt

            st.session_state.scanning = True
            st.rerun()

        if stop_btn and st.session_state.scanning:
            for key in ("scan_thread", "wifi_thread", "ble_thread"):
                t = st.session_state.get(key)
                if t:
                    t.stop()
                    st.session_state[key] = None
            st.session_state.scanning = False
            st.rerun()

        # Live status for all active sensors
        if st.session_state.scanning:
            sdr_t  = st.session_state.scan_thread
            wifi_t = st.session_state.wifi_thread
            ble_t  = st.session_state.ble_thread
            status_parts = []
            if sdr_t:
                status_parts.append(f"SDR: {sdr_t.status}")
            if wifi_t:
                status_parts.append(f"WiFi: {wifi_t.status}")
            if ble_t:
                status_parts.append(f"BLE: {ble_t.status}")
            if status_parts:
                st.caption(" | ".join(status_parts))

        st.divider()
        if st.button("Clear detections"):
            st.session_state.detections = []
            st.session_state.results = deque(maxlen=500)
            st.session_state.waterfall = None
            st.session_state.wf_freqs = None
            st.session_state.latest_recording_path = ""
            st.rerun()

    # ── Main content ──────────────────────────────────────────────────────────
    drain_queue()

    (tab_workflow, tab_spectrum, tab_detections, tab_command, tab_db,
     tab_ocusync, tab_ble, tab_wifi, tab_camera,
     tab_nethunter, tab_sniffer, tab_inspector) = st.tabs([
        "🧭 Workflow", "📊 Spectrum", "🎯 Detections", "🕹️ Command", "📋 Drone DB",
        "📡 OcuSync Map", "📶 Remote ID", "🔍 WiFi Probe", "📹 Camera",
        "🌐 Net Hunter", "📡 Pkt Sniffer", "🔬 Signal Inspector",
    ])

    with tab_workflow:
        detections = list(st.session_state.detections)
        sensor_hits = list(st.session_state.sensor_hits)
        wifi_hits = [h for h in reversed(sensor_hits) if h.get("source") == "wifi"][:12]
        ble_hits = [h for h in reversed(sensor_hits) if h.get("source") == "ble"][:12]

        st.subheader("Find -> Command -> Record")
        st.caption(
            "This is the replacement flow for Scanner.py: use all available sensors to find a likely drone, "
            "take the right command path for that target, and keep a reusable IQ clip when SDR hits trigger."
        )

        f1, f2, f3 = st.columns(3)
        f1.metric("SDR hits", len(detections))
        f2.metric("WiFi drone SSIDs", len(wifi_hits))
        f3.metric("BLE / Remote ID hits", len(ble_hits))
        target_rows = build_target_picture(detections, sensor_hits)
        if target_rows:
            st.markdown("##### Target picture")
            st.dataframe(target_rows[:12], width="stretch", hide_index=True)

        with st.expander("Direction finding / geolocation reality check"):
            st.caption(
                "With one laptop and one SDR you can do detection and manual bearing work, "
                "but not true TDOA or phase-based direction finding."
            )
            st.dataframe(direction_finding_capability(), width="stretch", hide_index=True)
            st.markdown(
                "Owner-safe next step: use a directional antenna, log peak power by heading, "
                "and take 2-3 bearings from different positions to narrow the source area."
            )

        target_map: dict[str, dict] = {}
        target_keys: list[str] = []

        for display_idx, r in enumerate(reversed(detections[-12:]), start=1):
            if not r.matches:
                continue
            m = r.matches[0]
            key = f"sdr:{display_idx}"
            target_map[key] = {"kind": "sdr", "result": r}
            target_keys.append(key)
        for display_idx, hit in enumerate(wifi_hits, start=1):
            key = f"wifi:{display_idx}"
            target_map[key] = {"kind": "wifi", "hit": hit}
            target_keys.append(key)
        for display_idx, hit in enumerate(ble_hits, start=1):
            key = f"ble:{display_idx}"
            target_map[key] = {"kind": "ble", "hit": hit}
            target_keys.append(key)

        def _target_label(target_key: str) -> str:
            item = target_map[target_key]
            if item["kind"] == "sdr":
                r = item["result"]
                m = r.matches[0]
                return (
                    f"SDR  {m.name}  {int(m.confidence * 100)}%  "
                    f"{human_freq(r.center_freq_hz)}  {r.timestamp_utc[11:19]}Z"
                )
            if item["kind"] == "wifi":
                hit = item["hit"]
                return (
                    f"WiFi  {hit.get('brand', '?')} {hit.get('model', '')}  "
                    f"{hit.get('ssid', '?')}  ch{hit.get('channel', '?')}"
                )
            hit = item["hit"]
            return (
                f"BLE  {hit.get('type', '?')}  {hit.get('name', 'Unknown')}  "
                f"{hit.get('address', '?')}"
            )

        if not target_keys:
            st.info("No targets yet. Start scanning with SDR, WiFi, or BLE enabled.")
        else:
            selected_target = st.selectbox(
                "Working target",
                target_keys,
                format_func=_target_label,
                key="workflow_target",
            )
            target = target_map[selected_target]

            col_find, col_cmd, col_rec = st.columns(3)
            with col_find:
                st.markdown("#### 1. Find")
                if target["kind"] == "sdr":
                    r = target["result"]
                    m = r.matches[0]
                    st.markdown(f"**{m.name}** ({m.brand})")
                    st.markdown(
                        f"Freq: `{human_freq(r.center_freq_hz)}`  \n"
                        f"SNR: `{r.snr_db:.1f} dB`  \n"
                        f"Confidence: `{int(m.confidence * 100)}%`  \n"
                        f"Observed: `{r.timestamp_utc[:19]}Z`"
                    )
                    st.caption(m.notes)
                elif target["kind"] == "wifi":
                    hit = target["hit"]
                    st.markdown(f"**{hit.get('brand', '?')} {hit.get('model', '')}**")
                    st.markdown(
                        f"SSID: `{hit.get('ssid', '?')}`  \n"
                        f"Signal: `{hit.get('signal_pct', '?')}%`  \n"
                        f"Band: `{hit.get('band', '?')}`  \n"
                        f"Channel: `{hit.get('channel', '?')}`"
                    )
                    st.caption("WiFi SSID hit from the host adapter.")
                else:
                    hit = target["hit"]
                    decoded = hit.get("decoded", {})
                    st.markdown(f"**{hit.get('type', '?')}**")
                    st.markdown(
                        f"Name: `{hit.get('name', 'Unknown')}`  \n"
                        f"Address: `{hit.get('address', '?')}`  \n"
                        f"RSSI: `{hit.get('rssi', '?')} dBm`"
                    )
                    if decoded:
                        st.caption(json.dumps(decoded, sort_keys=True)[:220])
                    else:
                        st.caption("BLE / Remote ID beacon observed.")

            with col_cmd:
                st.markdown("#### 2. Command")
                if target["kind"] == "sdr":
                    r = target["result"]
                    m = r.matches[0]
                    plan = command_guidance_for_match(m)
                elif target["kind"] == "wifi":
                    hit = target["hit"]
                    plan = command_guidance_for_wifi_hit(hit)
                else:
                    plan = {
                        "title": "Broadcast only",
                        "detail": (
                            "BLE / Remote ID gives you identity and proximity, not a control channel. "
                            "Use WiFi or SDR detections to move to a commandable path."
                        ),
                        "direct": None,
                        "prefill": None,
                    }
                st.markdown(f"**{plan['title']}**")
                st.write(plan["detail"])

                direct = plan.get("direct")
                if direct:
                    if st.button("Connect from workflow", key=f"wf_connect_{selected_target}"):
                        connect_direct_target(
                            direct["adapter"], direct["host"], direct["port"], _target_label(selected_target)
                        )
                prefill = plan.get("prefill")
                if prefill:
                    if st.button("Prefill command tab", key=f"wf_prefill_{selected_target}"):
                        prefill_command_panel(
                            prefill["adapter"],
                            host=prefill.get("host", ""),
                            port=prefill.get("port", 0),
                            connection_string=prefill.get("connection_string", ""),
                        )
                        st.success("Command tab fields updated for this target.")

            with col_rec:
                st.markdown("#### 3. Record")
                recording_path = ""
                if target["kind"] == "sdr":
                    recording_path = target["result"].recording_path
                elif st.session_state.latest_recording_path:
                    recording_path = st.session_state.latest_recording_path

                if recording_path and Path(recording_path).exists():
                    rec_path = Path(recording_path)
                    cf32_path = rec_path.with_suffix(".cf32")
                    st.markdown(f"Saved clip: `{rec_path.name}`")
                    st.caption(f"Dir: `{rec_path.parent}`")
                    if st.button("Load into inspector", key=f"wf_load_rec_{selected_target}"):
                        try:
                            load_recording_into_inspector(str(rec_path))
                            st.success(f"Loaded {rec_path.name} into Signal Inspector.")
                        except Exception as exc:
                            st.error(f"Recording load failed: {exc}")
                    st.download_button(
                        "Download NPZ clip",
                        data=rec_path.read_bytes(),
                        file_name=rec_path.name,
                        mime="application/octet-stream",
                        key=f"wf_dl_npz_{selected_target}",
                    )
                    if cf32_path.exists():
                        st.download_button(
                            "Download CF32 clip",
                            data=cf32_path.read_bytes(),
                            file_name=cf32_path.name,
                            mime="application/octet-stream",
                            key=f"wf_dl_cf32_{selected_target}",
                        )
                else:
                    st.info(
                        "No IQ clip is attached to this target yet. Enable `Auto-record SDR detections` and run the SDR sensor to save clips automatically."
                    )

    # Tab 1: Spectrum / Live View
    with tab_spectrum:
        sdr_t  = st.session_state.scan_thread
        wifi_t = st.session_state.wifi_thread
        ble_t  = st.session_state.ble_thread
        sdr_dets     = st.session_state.detections
        sensor_hits  = st.session_state.sensor_hits
        results_list = list(st.session_state.results)

        # ── Detection alert banner ─────────────────────────────────────────────
        recent_wifi_ble = sensor_hits[-6:]
        banner_items = []
        if sdr_dets and sdr_dets[-1].matches:
            m = sdr_dets[-1].matches[0]
            banner_items.append(("success", f"📻 **{m.name}** ({m.brand})  conf {m.confidence*100:.0f}%"))
        for h in recent_wifi_ble:
            icon = "📶" if h.get("source") == "wifi" else "🔵"
            banner_items.append(("warning",
                f"{icon} **{h.get('brand','?')} {h.get('model') or h.get('name','')}** "
                f"— `{h.get('ssid') or h.get('address','')}`"))
        if banner_items:
            bcols = st.columns(len(banner_items))
            for bi, (btype, btxt) in enumerate(banner_items):
                with bcols[bi]:
                    if btype == "success":
                        st.success(btxt)
                    else:
                        st.warning(btxt)

        # ── Sweep progress (SDR mode) ──────────────────────────────────────────
        if sdr_t and sdr_t.is_alive():
            total_steps = max(1, len(sdr_t.freq_plan_hz))
            cur_step    = min(sdr_t.step, total_steps - 1)
            cur_freq    = sdr_t.freq_plan_hz[cur_step]
            prog        = (cur_step + 1) / total_steps

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Sweep #", sdr_t.sweep + 1)
            m2.metric("Step", f"{cur_step + 1} / {total_steps}")
            m3.metric("Freq", human_freq(cur_freq))
            m4.metric("SDR detections", len(sdr_dets))
            m5.metric("All hits", len(sdr_dets) + len(sensor_hits))
            st.progress(prog, text=f"↗ Sweeping: {sdr_t.status}")

        elif wifi_t and wifi_t.is_alive():
            # WiFi-only: show scan count and countdown
            scan_n     = wifi_t.scan_count
            last_t     = wifi_t.last_scan_time
            next_t     = wifi_t.next_scan_time
            all_nets   = wifi_t.last_all_networks
            drone_hits = wifi_t.last_hits
            elapsed    = time.time() - last_t if last_t else 0
            remaining  = max(0.0, wifi_t.interval_s - elapsed)
            progress_v = min(1.0, elapsed / wifi_t.interval_s) if wifi_t.interval_s else 0

            wc1, wc2, wc3 = st.columns(3)
            wc1.metric("WiFi scans run", scan_n)
            wc2.metric("Networks seen", len(all_nets))
            wc3.metric("Drone SSIDs", len(drone_hits))
            if last_t:
                st.progress(progress_v,
                    text=f"📶 Next scan in {remaining:.0f}s  ·  "
                         f"last: {time.strftime('%H:%M:%S', time.localtime(last_t))}")
            else:
                st.progress(0.0, text="📶 First WiFi scan in progress…")

        elif ble_t and ble_t.is_alive():
            bc1, bc2 = st.columns(2)
            bc1.metric("BLE scans", ble_t.scan_count if hasattr(ble_t, "scan_count") else "—")
            bc2.metric("BLE devices found", len(ble_t.last_devices))
            st.info(f"🔵 {ble_t.status}")

        elif not st.session_state.scanning:
            st.info("Select sensors in the sidebar and click **▶ Start**.")

        # ── Main plots ─────────────────────────────────────────────────────────
        wifi_active_no_sdr = (
            wifi_t and wifi_t.is_alive()
            and (sdr_t is None or not sdr_t.is_alive())
        )

        if wifi_active_no_sdr:
            # WiFi radar replaces the spectrum chart
            all_nets = getattr(wifi_t, "last_all_networks", [])
            fig_radar = build_wifi_radar_fig(
                all_nets,
                interval_s=wifi_t.interval_s,
                last_scan_time=wifi_t.last_scan_time,
            )
            st.plotly_chart(fig_radar, width="stretch")

            # BLE side-by-side if also active
            if ble_t and ble_t.last_devices:
                st.markdown("**BLE Remote ID devices detected:**")
                for dev in ble_t.last_devices:
                    st.markdown(
                        f"- `{dev.get('address','?')}` — {dev.get('name','Unknown')} "
                        f"({dev.get('rssi','?')} dBm)  {dev.get('type','')}"
                    )
        else:
            # SDR spectrum + waterfall (or empty state)
            if results_list:
                last: ScanResult = results_list[-1]
                if not (sdr_t and sdr_t.is_alive()):
                    # Show metrics row only when not already shown above
                    rm1, rm2, rm3, rm4 = st.columns(4)
                    rm1.metric("Center Freq", human_freq(last.center_freq_hz))
                    rm2.metric("Peak",        f"{last.peak_db:.1f} dB")
                    rm3.metric("SNR",         f"{last.snr_db:.1f} dB")
                    rm4.metric("Occ. BW",     human_bw(last.occupied_bw_hz))
            fig = build_spectrum_fig(results_list)
            st.plotly_chart(fig, width="stretch")

        # ── Session tally ──────────────────────────────────────────────────────
        sdr_n  = len(sdr_dets)
        wfi_n  = sum(1 for h in sensor_hits if h.get("source") == "wifi")
        ble_n  = sum(1 for h in sensor_hits if h.get("source") == "ble")
        if sdr_n + wfi_n + ble_n:
            st.caption(f"Session detections — SDR: {sdr_n} · WiFi: {wfi_n} · BLE: {ble_n}")

    # Tab 2: Detections
    with tab_detections:
        dets: list[ScanResult] = st.session_state.detections
        if not dets:
            st.info("No detections yet. Start a scan targeting drone frequency bands.")
            st.markdown("**Recommended start**: Select *DJI 2.4 GHz (OcuSync/Tello)* from the Quick Band menu.")
        else:
            st.subheader(f"Detections ({len(dets)})")
            all_brands = sorted({r.matches[0].brand for r in dets if r.matches})
            fc1, fc2 = st.columns([2, 1])
            sel_brands = fc1.multiselect("Brand filter", all_brands, default=all_brands, key="det_brand_filter")
            min_conf = fc2.slider("Min confidence %", 0, 100, 45, key="det_min_conf")
            filtered_dets = [
                r for r in dets
                if r.matches
                and (not sel_brands or r.matches[0].brand in sel_brands)
                and int(r.matches[0].confidence * 100) >= min_conf
            ]
            d1, d2, d3 = st.columns(3)
            d1.metric("Filtered hits", len(filtered_dets))
            d2.metric("High confidence", sum(1 for r in filtered_dets if r.matches and r.matches[0].confidence >= 0.70))
            d3.metric("Saved clips", sum(1 for r in filtered_dets if r.recording_path))
            if not filtered_dets:
                st.info("No detections match the current filters.")
            # Summary table
            rows = []
            for r in reversed(filtered_dets[-50:]):
                if r.matches:
                    m = r.matches[0]
                    rows.append({
                        "Time": r.timestamp_utc[11:19] + "Z",
                        "Drone": m.name,
                        "Brand": m.brand,
                        "Conf %": int(m.confidence * 100),
                        "Freq": human_freq(r.center_freq_hz),
                        "SNR dB": round(r.snr_db, 1),
                        "BW": human_bw(r.occupied_bw_hz),
                    })
            if rows:
                st.dataframe(rows, width="stretch")

            st.divider()
            st.subheader("Recent detections")
            for idx, r in enumerate(reversed(filtered_dets[-10:])):
                render_detection_card(r, idx)

        # Export
        if dets:
            export_rows = []
            for r in dets:
                for m in r.matches:
                    export_rows.append({
                        "timestamp_utc": r.timestamp_utc,
                        "center_freq_hz": r.center_freq_hz,
                        "peak_db": r.peak_db,
                        "snr_db": r.snr_db,
                        "occupied_bw_hz": r.occupied_bw_hz,
                        "recording_path": r.recording_path,
                        "match_name": m.name,
                        "match_brand": m.brand,
                        "match_confidence": m.confidence,
                        "match_notes": m.notes,
                    })
            if export_rows:
                import io
                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=export_rows[0].keys())
                writer.writeheader()
                writer.writerows(export_rows)
                st.download_button(
                    "Export detections CSV",
                    data=buf.getvalue(),
                    file_name=f"dronedetect_{datetime.now().strftime('%Y%m%dT%H%M%S')}.csv",
                    mime="text/csv",
                )

    # Tab 3: Command
    with tab_command:
        st.subheader("Drone Command Interface")
        st.caption(
            "Connect to your drone via WiFi, MAVLink, or DJI OSDK. "
            "The SDR scan shows you whether the RF link is active — "
            "this panel lets you send commands."
        )

        # Add adapter manually
        with st.expander("Add drone connection"):
            adapter_type = st.selectbox(
                "Adapter type",
                list(ADAPTER_LABELS.keys()),
                format_func=lambda k: ADAPTER_LABELS[k],
                key="add_adapter_type",
            )
            conn_host = st.text_input("Host / IP", value="192.168.10.1", key="add_host")
            conn_port = st.number_input("Port", min_value=1, max_value=65535, value=8889, key="add_port")
            conn_extra = st.text_input(
                "Connection string (MAVLink only)",
                value="udpin:0.0.0.0:14550",
                key="add_conn_str",
            )
            if st.button("Connect", key="add_connect_btn"):
                adapter_cls = ADAPTER_MAP.get(adapter_type)
                if adapter_cls:
                    adapter_key = f"{adapter_type}_{conn_host}_{conn_port}"
                    adapter = adapter_cls()
                    if adapter_type == "mavlink":
                        result = adapter.connect(connection_string=conn_extra)
                    else:
                        result = adapter.connect(host=conn_host, port=int(conn_port))
                    if result.success:
                        st.session_state.adapters[adapter_key] = adapter
                        st.session_state.cmd_log.append(
                            f"{datetime.now(timezone.utc).isoformat()[:19]}Z  CONNECT  "
                            f"{adapter.name}: {result.message}"
                        )
                        st.success(result.message)
                        st.rerun()
                    else:
                        st.error(result.message)

        # Show active adapters
        if not st.session_state.adapters:
            st.info("No drone connections active. Add one above or click 'Connect to this drone' in the Detections tab.")
        else:
            for key, adapter in list(st.session_state.adapters.items()):
                with st.container(border=True):
                    render_adapter_panel(key, adapter)

        # Command log
        if st.session_state.cmd_log:
            st.divider()
            with st.expander("Command log"):
                for line in reversed(st.session_state.cmd_log[-50:]):
                    st.text(line)

        # Link lost recovery guide
        with st.expander("Lost link recovery guide"):
            st.markdown("""
            **If you've lost control of your DJI drone:**

            1. **Check RF presence** — use the Spectrum tab scanning 2.4 GHz / 5.8 GHz.
               If you see OcuSync signals, the drone is still transmitting and is alive.

            2. **DJI Tello** — connect to the drone's WiFi (`TELLO-XXXXXX`) from any device,
               then use the *Connect* panel above. Send `Land` immediately.

            3. **DJI Mavic/Mini/Air (OcuSync)** — move your RC controller to line-of-sight.
               These drones have an automatic Return-to-Home if RC link is lost > 3 s.
               You **cannot** command OcuSync drones in Python without DJI Mobile SDK
               on an iOS/Android device, or DJI OSDK on enterprise models.

            4. **MAVLink drones (ArduPilot/PX4)** — connect with a backup GCS
               (e.g., `udpin:0.0.0.0:14550`) using the MAVLink adapter. Send RTL/Land.

            5. **DJI Enterprise (Matrice/Agras)** — use the DJI OSDK adapter with
               the companion computer bridge. Refer to:
               https://github.com/dji-sdk/Onboard-SDK

            6. **Parrot** — connect to drone WiFi (`ANAFI-XXXXXX`), use Parrot adapter.
               Send `Land` command via Olympe SDK.
            """)

        with st.expander("DJI WiFi credential recovery (own drone — forgot password)"):
            st.markdown("""
            Use documented defaults and vendor reset paths only. This panel stays on the
            owner-recovery side of the line: no brute force, no speculative cracking,
            no protocol bypass.
            """)
            dji_model_keys = list(_DJI_WIFI_DEFAULTS.keys())
            sel_model = st.selectbox("Select your DJI model", dji_model_keys,
                                      key="dji_pw_model")
            entry = _DJI_WIFI_DEFAULTS[sel_model]
            st.markdown(
                f"**Factory default WiFi password:** `{entry['password']}`\n\n"
                f"**Factory reset procedure:** {entry['reset']}\n\n"
                "_A factory reset restores the WiFi password to the default and resets "
                "all flight parameters. Your drone is NOT reformatted — photos on the SD card "
                "are safe. After reset, reconnect via DJI Fly / DJI GO 4 / DJI Pilot._"
            )
            st.divider()
            st.markdown("**Recovery flow**")
            st.markdown("""
            1. Try the documented factory default above.
            2. If it fails, use the model-specific reset procedure above.
            3. Re-join the drone AP and verify the expected gateway IP in the WiFi Probe tab.
            4. For enterprise hardware, prefer DJI Assistant 2 / DJI Pilot / controller pairing over ad-hoc recovery.
            """)
            st.divider()
            st.markdown("""
            **Alternative: USB recovery via DJI Assistant 2**
            1. Connect drone to PC via USB-C while drone is powered on.
            2. Open DJI Assistant 2 (free from dl.dji.com).
            3. Click your drone → Settings → Reset WiFi Password.
            4. Works on all Mavic, Mini, Air, Phantom, Inspire, Matrice models.
            """)

    # Tab 4: Drone database
    with tab_db:
        st.subheader("Known Drone Signatures")
        st.caption("Signature database used for RF fingerprinting. All entries are passive detection only.")
        for key, sig in DRONE_SIGNATURES.items():
            with st.expander(f"{sig.name} ({sig.brand})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Protocol:** {sig.modulation}")
                    st.markdown(f"**FHSS:** {'Yes' if sig.fhss else 'No'}")
                    if sig.hop_interval_ms:
                        st.markdown(f"**Hop interval:** {sig.hop_interval_ms} ms")
                    st.markdown(f"**Typical BW:** {human_bw(sig.bandwidth_hz)}")
                    st.markdown(f"**Min SNR gate:** {sig.snr_floor_db} dB")
                with col2:
                    if sig.freqs_24ghz_mhz:
                        st.markdown(f"**2.4 GHz channels:** {len(sig.freqs_24ghz_mhz)} "
                                    f"({sig.freqs_24ghz_mhz[0]:.1f}–{sig.freqs_24ghz_mhz[-1]:.1f} MHz)")
                    if sig.freqs_58ghz_mhz:
                        st.markdown(f"**5.8 GHz channels:** {len(sig.freqs_58ghz_mhz)} "
                                    f"({sig.freqs_58ghz_mhz[0]:.0f}–{sig.freqs_58ghz_mhz[-1]:.0f} MHz)")
                    if sig.freqs_900mhz_mhz:
                        st.markdown(f"**900 MHz band:** {len(sig.freqs_900mhz_mhz)} channels")
                    if sig.freqs_433mhz_mhz:
                        st.markdown(f"**433 MHz band:** {len(sig.freqs_433mhz_mhz)} channels")
                    if sig.wifi_ip:
                        st.markdown(f"**WiFi IP:** {sig.wifi_ip}:{sig.wifi_port}")
                    if sig.adapter:
                        st.markdown(f"**Command adapter:** {ADAPTER_LABELS.get(sig.adapter, sig.adapter)}")
                st.caption(sig.description)

    # Tab 5: OcuSync channel map
    with tab_ocusync:
        st.subheader("OcuSync Channel Occupancy")
        st.caption(
            "Maps scan results onto OcuSync channel grid. "
            "Green bars = active channel (strong signal), yellow = weak, dark = empty. "
            "FHSS hopping appears as many green bars spread across the grid."
        )
        band_choice = st.radio(
            "Band", [_BAND_24GHZ, _BAND_58GHZ], horizontal=True, key="ocu_band"
        )
        results_for_map = list(st.session_state.results)
        fig_ocu = build_ocusync_channel_map(results_for_map, band=band_choice)
        st.plotly_chart(fig_ocu, width="stretch")
        st.markdown("""
        **How to use:**
        1. Select *DJI 2.4 GHz* quick band preset in the sidebar and start scanning.
        2. Watch this chart — OcuSync FHSS will light up 3–6 channels spread evenly
           across the 2400–2483 MHz range.
        3. Switch to *5.8 GHz* if you want to check the O3/O4 upper band.
        """)

    # Tab 6: BLE Remote ID scanner
    with tab_ble:
        st.subheader("BLE Remote ID Scanner")
        st.caption(
            "Scans nearby Bluetooth advertisements for FAA Remote ID broadcasts "
            "(OpenDroneID / ASTM F3411-22a, UUID 0xFFFA) and DJI proprietary BLE beacons."
        )
        ble_duration = st.slider("Scan duration (s)", 2, 15, 5, key="ble_dur")
        if st.button("Scan BLE", key="ble_scan_btn"):
            with st.spinner(f"Scanning BLE for {ble_duration} s…"):
                ble_devices, ble_err = run_ble_scan(duration=float(ble_duration))
            if ble_err:
                st.warning(f"BLE scan note: {ble_err}")
            if ble_devices:
                st.success(f"Found {len(ble_devices)} drone-related BLE device(s).")
                for dev in ble_devices:
                    with st.container(border=True):
                        c1, c2 = st.columns([2, 3])
                        with c1:
                            st.markdown(f"**{dev.get('name', 'Unknown')}**")
                            st.markdown(f"Address: `{dev.get('address', '—')}`")
                            st.markdown(f"RSSI: {dev.get('rssi', '—')} dBm")
                        with c2:
                            odid = dev.get("odid")
                            if odid:
                                st.markdown("**OpenDroneID payload:**")
                                for k, v in odid.items():
                                    st.markdown(f"- **{k}:** {v}")
                            else:
                                st.markdown(f"Type: `{dev.get('type', 'DJI proprietary')}`")
                                raw = dev.get("raw_hex", "")
                                if raw:
                                    st.code(raw, language=None)
            else:
                st.info("No drone BLE beacons detected. Make sure Bluetooth is enabled and a drone is nearby.")
        st.markdown("""
        **Requirements:** `pip install bleak`
        **What it detects:**
        - FAA Remote ID (OpenDroneID) broadcast via BLE 4/5 — mandatory on drones ≥ 250 g from Mar 2024
        - DJI proprietary BLE manufacturer beacons (Mavic, Mini, Air series)
        """)

    # Tab 7: WiFi drone probe
    with tab_wifi:
        st.subheader("WiFi Drone Probe")

        # ── Section 1: SSID scan using the host WiFi chip ─────────────────────
        st.markdown("#### Scan airwaves for drone SSIDs")
        st.caption(
            "Uses your computer's WiFi adapter (via nmcli) to scan all nearby networks "
            "and flag SSIDs matching known drone patterns — no need to be connected to the drone."
        )
        if st.button("Scan WiFi Airwaves", key="wifi_ssid_btn"):
            with st.spinner("Scanning WiFi (forces a fresh rescan, ~5 s)…"):
                ssid_hits, _all_nets, ssid_err = scan_wifi_networks()
            if ssid_err:
                st.warning(f"WiFi scan issue: {ssid_err}")
            if ssid_hits:
                st.success(f"Detected {len(ssid_hits)} drone network(s) in range!")
                for net in ssid_hits:
                    with st.container(border=True):
                        ca, cb = st.columns([3, 2])
                        with ca:
                            st.markdown(f"**{net['brand']} — {net['model']}**")
                            st.markdown(f"SSID: `{net['ssid']}`")
                        with cb:
                            st.markdown(f"Signal: {net['signal_pct']}%")
                            st.markdown(f"Ch {net['channel']} · {net['band']}")
                st.info(
                    "Drone SSID detected! Connect your WiFi to it, then use "
                    "**Probe Network** below to reach the control API."
                )
            else:
                st.info("No drone SSIDs found nearby. Power on your drone and try again.")

        st.divider()
        # ── Section 2: IP probe on current connected network ──────────────────
        st.markdown("#### Probe current network for drone APIs")
        st.caption(
            "Probes factory-default gateway IPs on your current network. "
            "Works when you're already connected to a drone's built-in WiFi AP."
        )
        probe_timeout = st.slider("Probe timeout (s)", 0.2, 2.0, 0.5, step=0.1, key="wifi_timeout")
        if st.button("Probe Network", key="wifi_probe_btn"):
            with st.spinner("Probing drone gateway IPs…"):
                wifi_hits = probe_wifi_drones(timeout=float(probe_timeout))
            if wifi_hits:
                st.success(f"Found {len(wifi_hits)} reachable drone(s) on this network!")
                for hit in wifi_hits:
                    with st.container(border=True):
                        col_a, col_b = st.columns([3, 2])
                        with col_a:
                            st.markdown(f"**{hit['brand']} — {hit['model']}**")
                            st.markdown(f"IP: `{hit['ip']}` | Port: `{hit['port']}`")
                        with col_b:
                            adapter_label = ADAPTER_LABELS.get(hit["adapter"], hit["adapter"])
                            if st.button(f"Connect ({adapter_label})", key=f"wifi_conn_{hit['ip']}_{hit['port']}"):
                                adapter_cls = ADAPTER_MAP.get(hit["adapter"])
                                if adapter_cls:
                                    adapter = adapter_cls()
                                    r = adapter.connect(host=hit["ip"], port=hit["port"])
                                    ak = f"{hit['adapter']}_{hit['ip']}_{hit['port']}"
                                    if r.success:
                                        st.session_state.adapters[ak] = adapter
                                        st.session_state.cmd_log.append(
                                            f"{datetime.now(timezone.utc).isoformat()[:19]}Z  "
                                            f"WIFI-CONNECT  {adapter.name}: {r.message}"
                                        )
                                        st.success(r.message)
                                        st.rerun()
                                    else:
                                        st.error(r.message)
            else:
                st.info(
                    "No drone APs detected on this network. "
                    "Connect your PC/laptop to the drone's WiFi hotspot first, then probe."
                )
        st.markdown("""
        **Supported models (factory defaults):**
        | Brand | Model | Default IP |
        |-------|-------|-----------|
        | DJI | Tello / Tello EDU | 192.168.10.1 |
        | DJI | Mavic / Mini (RC WiFi mode) | 192.168.10.1 |
        | Parrot | ANAFI / Bebop 2 | 192.168.42.1 |
        | Skydio | 2 / X2 / 3 | 192.168.77.1 |
        | Autel | EVO / EVO II | 192.168.2.1 |
        | FIMI | X8 SE / X8 Pro | 192.168.11.1 |
        | Hubsan | Zino 2 / Zino Mini Pro | 192.168.10.1 |
        | Holy Stone | HS720E / HS360S | 192.168.1.1 |
        """)

    # Tab 8: Camera Feed
    with tab_camera:
        st.subheader("Camera & IoT")
        st.caption("Live video, ONVIF inventory, and Bonjour discovery in one place.")

        if not HAVE_CV2:
            st.warning(
                "opencv-python is not installed. "
                "Run `pip install opencv-python` to enable video feed."
            )

        cam_ip = st.text_input("Device IP", value="192.168.10.1", key="cam_ip")
        cam_timeout = st.slider("Probe timeout (s)", 0.5, 3.0, 1.0, key="cam_timeout")
        cam_video_tab, cam_inventory_tab, cam_ref_tab = st.tabs(["Video", "Inventory", "Reference"])

        with cam_video_tab:
            cc_left, cc_right = st.columns([1, 2])
            with cc_left:
                if st.button("Auto-discover streams", key="cam_discover", type="primary"):
                    with st.spinner(f"Scanning {cam_ip} for video ports…"):
                        st.session_state.camera_stream_candidates = scan_video_streams(
                            cam_ip, timeout=float(cam_timeout)
                        )
                stream_candidates = st.session_state.camera_stream_candidates
                if stream_candidates:
                    selected_stream_idx = st.selectbox(
                        "Discovered stream",
                        list(range(len(stream_candidates))),
                        format_func=lambda i: stream_candidates[i]["url"],
                        key="cam_stream_pick",
                    )
                    if st.button("Use selected stream", key="cam_use_stream"):
                        st.session_state.cam_url = stream_candidates[selected_stream_idx]["url"]
                        st.rerun()
                    st.caption(f"{len(stream_candidates)} candidate stream(s) found on {cam_ip}.")
                else:
                    st.info("No cached stream candidates yet.")

            with cc_right:
                stream_url = st.text_input(
                    "Stream URL",
                    placeholder="rtsp://192.168.42.1/live  |  udp://0.0.0.0:11111",
                    key="cam_url",
                    help="RTSP, MJPEG, or UDP H.264. Tello: udp://0.0.0.0:11111 after SDK connect.",
                )
                cam_live = st.checkbox("Live mode (auto-refresh with scan)", key="cam_live")
                grab_col, _ = st.columns([1, 3])
                with grab_col:
                    grab_btn = st.button(
                        "Grab frame",
                        key="cam_grab",
                        disabled=not HAVE_CV2 or not stream_url,
                    )

                frame_placeholder = st.empty()
                if (grab_btn or (cam_live and st.session_state.scanning)) and stream_url:
                    if not HAVE_CV2:
                        frame_placeholder.error("opencv-python not installed")
                    else:
                        with st.spinner("Connecting to stream…"):
                            frame = grab_video_frame(stream_url, timeout_ms=int(cam_timeout * 1000))
                        if frame is not None:
                            frame_placeholder.image(
                                frame,
                                caption=f"{stream_url}  ·  {frame.shape[1]}×{frame.shape[0]}",
                                width="stretch",
                            )
                        else:
                            frame_placeholder.error(
                                "No frame returned. Verify you are on the device AP, the stream is enabled, and the URL is correct."
                            )
                elif not stream_url:
                    frame_placeholder.info("Paste a stream URL or pick one from auto-discovery.")

        with cam_inventory_tab:
            act1, act2, act3 = st.columns(3)
            with act1:
                if st.button("Discover ONVIF on LAN", key="cam_onvif_lan", type="primary"):
                    with st.spinner("Sending WS-Discovery probes…"):
                        discovered = onvif_ws_discover(timeout=float(cam_timeout) + 0.8, attempts=2)
                    enriched = []
                    for device in discovered:
                        xaddr = device.get("xaddr", "")
                        probe_data = None
                        probe_status = "No device service URL"
                        if xaddr:
                            probe_data, probe_status = onvif_get_system_datetime(
                                xaddr, timeout=float(cam_timeout) + 1.0
                            )
                        enriched.append({**device, "probe_status": probe_status, "probe_datetime": probe_data})
                    st.session_state.camera_onvif_results = enriched
            with act2:
                if st.button("Probe ONVIF on current IP", key="cam_onvif_ip"):
                    with st.spinner(f"Probing common ONVIF URLs on {cam_ip}…"):
                        manual = probe_onvif_endpoints(
                            cam_ip,
                            ports=[80, 443, 8000, 8080, 8443, 8899],
                            timeout=float(cam_timeout) + 1.0,
                        )
                    st.session_state.camera_onvif_results = [
                        {
                            "endpoint": cam_ip,
                            "source_ip": cam_ip,
                            "xaddrs": [row["xaddr"]],
                            "xaddr": row["xaddr"],
                            "types": "manual onvif probe",
                            "scopes": "",
                            "device_name": "",
                            "hardware": "",
                            "location": "",
                            "probe_status": row["status"],
                            "probe_datetime": row["datetime"],
                        }
                        for row in manual
                    ]
            with act3:
                if st.button("Browse mDNS / Bonjour", key="cam_mdns"):
                    with st.spinner("Listening for mDNS services…"):
                        mdns_rows, mdns_err = discover_mdns_services(timeout=2.5)
                    if mdns_err:
                        st.warning(mdns_err)
                    st.session_state.camera_mdns_results = mdns_rows

            onvif_rows = st.session_state.camera_onvif_results
            mdns_rows = st.session_state.camera_mdns_results
            m1, m2 = st.columns(2)
            m1.metric("ONVIF targets", len(onvif_rows))
            m2.metric("mDNS services", len(mdns_rows))

            if onvif_rows:
                summary_rows = []
                for idx, device in enumerate(onvif_rows):
                    probe = device.get("probe_datetime") or {}
                    summary_rows.append({
                        "Target": idx + 1,
                        "Name": onvif_device_title(device),
                        "IP": device.get("source_ip", "") or device.get("endpoint", ""),
                        "Status": device.get("probe_status", "—"),
                        "UTC": probe.get("utc") or "—",
                        "XAddr": device.get("xaddr", ""),
                    })
                st.dataframe(summary_rows, width="stretch", hide_index=True)

                selected_onvif_idx = st.selectbox(
                    "Inspect ONVIF target",
                    list(range(len(onvif_rows))),
                    format_func=lambda i: (
                        f"{onvif_device_title(onvif_rows[i])} · "
                        f"{onvif_rows[i].get('source_ip', '') or onvif_rows[i].get('endpoint', '')}"
                    ),
                    key="cam_onvif_focus",
                )
                device = onvif_rows[selected_onvif_idx]
                xaddr = device.get("xaddr", "")
                probe = device.get("probe_datetime") or {}
                parsed = urlparse(xaddr) if xaddr else None
                onvif_ports = []
                if parsed and parsed.port:
                    onvif_ports.append(parsed.port)
                elif parsed and parsed.scheme == "https":
                    onvif_ports.append(443)
                elif parsed and parsed.scheme == "http":
                    onvif_ports.append(80)
                playbook = build_owner_recovery_playbook(
                    ip=device.get("source_ip", "") or device.get("endpoint", ""),
                    open_ports=onvif_ports,
                    onvif_data=probe or None,
                )
                with st.container(border=True):
                    st.markdown(f"**{onvif_device_title(device)}**")
                    st.code(xaddr or "(no device service URL)")
                    st.caption(device.get("types", ""))
                    if probe.get("utc") or probe.get("local"):
                        st.markdown(
                            f"UTC: `{probe.get('utc') or '—'}`  \n"
                            f"Local: `{probe.get('local') or '—'}`  \n"
                            f"TZ: `{probe.get('timezone') or '—'}`"
                        )
                    if device.get("scopes"):
                        with st.expander("Scopes"):
                            st.code(device["scopes"])
                    with st.expander("Owner-safe next steps"):
                        for step in playbook:
                            st.markdown(f"- {step}")
            else:
                st.info("No ONVIF inventory loaded yet.")

            if mdns_rows:
                with st.expander("mDNS / Bonjour services"):
                    st.dataframe(
                        [
                            {
                                "Service": row["service_type"],
                                "Name": row["name"],
                                "IP": row["ip"],
                                "Port": row["port"],
                                "Properties": summarize_mdns_properties(row.get("properties", {})),
                            }
                            for row in mdns_rows
                        ],
                        width="stretch",
                        hide_index=True,
                    )

        with cam_ref_tab:
            st.dataframe(
                [
                    {
                        "Brand": cfg["brand"],
                        "Model": cfg["model"],
                        "URL": cfg["url"],
                        "Note": cfg["note"],
                    }
                    for cfg in DRONE_STREAM_CONFIGS
                ],
                width="stretch",
                hide_index=True,
            )
            st.caption("Common stream patterns only. Actual URLs vary by app state, firmware, and transport.")

    # ── Tab 9: Net Hunter ─────────────────────────────────────────────────────
    with tab_nethunter:
        st.subheader("Net Hunter — Find Drones on Any Network")
        st.caption("Subnet inventory, WiFi discovery, and protocol-specific follow-up without the stacked report spam.")
        nh_overview_tab, nh_probe_tab, nh_tools_tab = st.tabs(["Overview", "Subnet Probe", "Deep Tools"])

        with nh_overview_tab:
            ov_left, ov_right = st.columns([1, 2])

            with ov_left:
                with st.expander("Local network context", expanded=True):
                    ifaces = get_network_interfaces()
                    if ifaces:
                        st.dataframe(
                            [{"Interface": i["name"], "IP/Mask": f"{i['ip']}/{i['prefix']}"} for i in ifaces],
                            width="stretch",
                            hide_index=True,
                        )
                    else:
                        st.info("Could not enumerate interfaces.")

                    if st.button("Refresh ARP table", key="nh_arp"):
                        st.session_state.nh_arp_results = arp_neighbors()
                    arp_res = st.session_state.nh_arp_results
                    if arp_res:
                        st.dataframe(
                            [{"IP": d["ip"], "MAC": d["mac"], "State": d["state"]} for d in arp_res],
                            width="stretch",
                            hide_index=True,
                        )

            with ov_right:
                if st.button("Scan all WiFi SSIDs", key="nh_wifi", type="primary"):
                    with st.spinner("Scanning WiFi (forced rescan ~5 s)…"):
                        drone_hits, all_nets, err = scan_wifi_networks()
                    if err:
                        st.warning(err)
                    st.session_state.nh_wifi_all = all_nets
                    st.session_state.nh_wifi_drones = drone_hits

                all_nets = st.session_state.nh_wifi_all
                drone_hits = st.session_state.nh_wifi_drones
                w1, w2 = st.columns(2)
                w1.metric("All SSIDs", len(all_nets))
                w2.metric("Drone SSIDs", len(drone_hits))
                if drone_hits:
                    st.dataframe(
                        [
                            {
                                "Brand": hit.get("brand", ""),
                                "Model": hit.get("model", ""),
                                "SSID": hit.get("ssid", ""),
                                "Signal": hit.get("signal_pct", ""),
                                "Band": hit.get("band", ""),
                                "Channel": hit.get("channel", ""),
                            }
                            for hit in drone_hits
                        ],
                        width="stretch",
                        hide_index=True,
                    )
                    with st.expander("Connection instructions"):
                        for d in drone_hits:
                            st.markdown(
                                f"- `{d['ssid']}` → connect with `nmcli dev wifi connect '{d['ssid']}'`, then use WiFi Probe or Subnet Probe."
                            )
                elif all_nets:
                    st.info("No known drone SSIDs in the latest WiFi scan.")

                if all_nets:
                    with st.expander("All nearby networks"):
                        drone_ssids = {d["ssid"] for d in drone_hits}
                        st.dataframe(
                            [
                                {
                                    "SSID": ("🚁 " if net.get("ssid", "") in drone_ssids else "") + net.get("ssid", ""),
                                    "Signal": net.get("signal_pct", ""),
                                    "Band": net.get("band", ""),
                                    "Channel": net.get("channel", ""),
                                    "Brand": net.get("brand", "") if net.get("ssid", "") in drone_ssids else "",
                                }
                                for net in all_nets
                            ],
                            width="stretch",
                            hide_index=True,
                        )

        with nh_probe_tab:
            nh_timeout = st.slider("Probe timeout (s)", 0.3, 2.0, 0.7, key="nh_sub_timeout")
            if st.button("Probe all drone subnets", key="nh_probe", type="primary"):
                with st.spinner(
                    f"Probing {len(_DRONE_SUBNETS)} subnets × {len(_DRONE_PROBE_LAST_OCTETS)} gateway IPs …"
                ):
                    st.session_state.nh_subnet_results = probe_drone_subnets(float(nh_timeout))

            subnet_res = st.session_state.nh_subnet_results
            if subnet_res:
                st.dataframe(
                    [
                        {
                            "IP": device["ip"],
                            "Brand": device.get("brand", "?"),
                            "Model": device.get("model", ""),
                            "Ports": summarize_ports(device["open_ports"]),
                            "Subnet": device.get("subnet", ""),
                        }
                        for device in subnet_res
                    ],
                    width="stretch",
                    hide_index=True,
                )
                selected_device_ip = st.selectbox(
                    "Inspect subnet hit",
                    [d["ip"] for d in subnet_res],
                    format_func=lambda ip: next(
                        f"{d.get('brand', '?')} {d.get('model', '')} · {ip}"
                        for d in subnet_res if d["ip"] == ip
                    ),
                    key="nh_device_focus",
                )
                device = next(d for d in subnet_res if d["ip"] == selected_device_ip)
                st.markdown(
                    f"**{device.get('brand', '?')} {device.get('model', '')}** · `{device['ip']}` · {summarize_ports(device['open_ports'])}"
                )

                action1, action2, action3 = st.columns(3)
                with action1:
                    if st.button("Full port scan", key=f"ps_{device['ip']}"):
                        with st.spinner(f"Port scanning {device['ip']} …"):
                            st.session_state.nh_port_scan_results[device["ip"]] = port_scan(device["ip"])
                with action2:
                    if st.button("HTTP fingerprint", key=f"httpfp_{device['ip']}"):
                        http_ports = [p for p in device["open_ports"] if p in {80, 443, 8000, 8080, 8443, 8888, 8899}]
                        with st.spinner(f"Fingerprinting HTTP services on {device['ip']} …"):
                            st.session_state.nh_http_results[device["ip"]] = fingerprint_http_device(
                                device["ip"],
                                ports=http_ports or None,
                                timeout=1.2,
                            )
                with action3:
                    if st.button("Probe ONVIF", key=f"onvif_{device['ip']}"):
                        http_ports = [p for p in device["open_ports"] if p in {80, 443, 8000, 8080, 8443, 8899}]
                        with st.spinner(f"Probing ONVIF endpoints on {device['ip']} …"):
                            st.session_state.nh_onvif_results[device["ip"]] = probe_onvif_endpoints(
                                device["ip"],
                                ports=http_ports or None,
                                timeout=2.0,
                            )

                ps_res = st.session_state.nh_port_scan_results.get(device["ip"])
                http_res = st.session_state.nh_http_results.get(device["ip"])
                onvif_res = st.session_state.nh_onvif_results.get(device["ip"])

                if ps_res is not None:
                    st.caption(f"Expanded ports: {summarize_ports(ps_res, limit=8)}")

                if http_res:
                    with st.expander("HTTP fingerprints"):
                        st.dataframe(
                            [
                                {
                                    "URL": row["url"],
                                    "Status": row.get("status", ""),
                                    "Hint": row.get("device_hint", ""),
                                    "Server": row.get("server", ""),
                                    "Realm": row.get("realm", ""),
                                    "Title": row.get("title", ""),
                                    "Detail": row.get("detail", ""),
                                }
                                for row in http_res
                            ],
                            width="stretch",
                            hide_index=True,
                        )

                if onvif_res:
                    with st.expander("ONVIF probes"):
                        st.dataframe(
                            [
                                {
                                    "XAddr": row["xaddr"],
                                    "Status": row["status"],
                                    "UTC": (row.get("datetime") or {}).get("utc", "—"),
                                    "Local": (row.get("datetime") or {}).get("local", "—"),
                                    "TZ": (row.get("datetime") or {}).get("timezone", "—"),
                                }
                                for row in onvif_res
                            ],
                            width="stretch",
                            hide_index=True,
                        )

                http_ok = next((row for row in (http_res or []) if row.get("status") != "error"), None)
                onvif_ok = next((row for row in (onvif_res or []) if row.get("datetime")), None)
                playbook = build_owner_recovery_playbook(
                    ip=device["ip"],
                    open_ports=ps_res or device["open_ports"],
                    onvif_data=onvif_ok.get("datetime") if onvif_ok else None,
                    http_data=http_ok,
                )
                with st.expander("Owner-safe next steps", expanded=True):
                    for step in playbook:
                        st.markdown(f"- {step}")
            elif "nh_subnet_results" in st.session_state:
                st.info("No devices found on the known drone subnets in the last probe.")

        with nh_tools_tab:
            col_mav, col_ps = st.columns(2)
            with col_mav:
                st.markdown("##### MAVLink heartbeat listener")
                mav_dur = st.slider("Listen duration (s)", 2, 15, 5, key="nh_mav_dur")
                if st.button("Listen for MAVLink", key="nh_mavlisten"):
                    with st.spinner(f"Listening on MAVLink UDP ports for {mav_dur} s …"):
                        st.session_state.nh_mavlink_results = listen_mavlink_udp(float(mav_dur))
                if st.session_state.nh_mavlink_results:
                    st.dataframe(
                        [
                            {
                                "Source": pkt["src"],
                                "Bind port": pkt["bind_port"],
                                "Version": pkt["version"],
                                "SysID": pkt["sysid"],
                                "Message": pkt["msg_name"],
                            }
                            for pkt in st.session_state.nh_mavlink_results
                        ],
                        width="stretch",
                        hide_index=True,
                    )

            with col_ps:
                st.markdown("##### Manual port scanner")
                ps_ip = st.text_input("IP to scan", value="192.168.10.1", key="nh_ps_ip")
                ps_timeout = st.slider("Timeout (s)", 0.3, 2.0, 0.6, key="nh_ps_timeout")
                ps_btn1, ps_btn2, ps_btn3 = st.columns(3)
                with ps_btn1:
                    if st.button("Run port scan", key="nh_ps_run"):
                        with st.spinner(f"Scanning {ps_ip} …"):
                            st.session_state.nh_port_scan_results[ps_ip] = port_scan(
                                ps_ip, timeout=float(ps_timeout)
                            )
                with ps_btn2:
                    if st.button("HTTP fingerprint", key="nh_ps_http"):
                        with st.spinner(f"Fingerprinting {ps_ip} …"):
                            st.session_state.nh_http_results[ps_ip] = fingerprint_http_device(
                                ps_ip, timeout=max(0.8, float(ps_timeout))
                            )
                with ps_btn3:
                    if st.button("Probe ONVIF", key="nh_ps_onvif"):
                        with st.spinner(f"Probing ONVIF on {ps_ip} …"):
                            st.session_state.nh_onvif_results[ps_ip] = probe_onvif_endpoints(
                                ps_ip,
                                timeout=max(1.2, float(ps_timeout) + 0.6),
                            )
                ps_r = st.session_state.nh_port_scan_results.get(ps_ip)
                if ps_r is not None:
                    st.caption(f"Ports: {summarize_ports(ps_r, limit=8)}")
                http_r = st.session_state.nh_http_results.get(ps_ip)
                if http_r:
                    with st.expander("HTTP fingerprints"):
                        st.dataframe(
                            [
                                {
                                    "URL": row["url"],
                                    "Status": row.get("status", ""),
                                    "Hint": row.get("device_hint", ""),
                                    "Server": row.get("server", ""),
                                    "Realm": row.get("realm", ""),
                                    "Title": row.get("title", ""),
                                    "Detail": row.get("detail", ""),
                                }
                                for row in http_r
                            ],
                            width="stretch",
                            hide_index=True,
                        )
                onvif_r = st.session_state.nh_onvif_results.get(ps_ip)
                if onvif_r:
                    with st.expander("ONVIF endpoint probes"):
                        st.dataframe(
                            [
                                {
                                    "XAddr": row["xaddr"],
                                    "Status": row["status"],
                                    "UTC": (row.get("datetime") or {}).get("utc", "—"),
                                    "Local": (row.get("datetime") or {}).get("local", "—"),
                                    "TZ": (row.get("datetime") or {}).get("timezone", "—"),
                                }
                                for row in onvif_r
                            ],
                            width="stretch",
                            hide_index=True,
                        )

    # ── Tab 10: Packet Sniffer ─────────────────────────────────────────────────
    with tab_sniffer:
        st.subheader("Packet Sniffer — Live UDP Network Inspector")
        st.caption(
            "Binds UDP sockets on common drone ports and captures all arriving packets. "
            "Decodes MAVLink, Tello SDK, DJI proprietary frames with hex/ASCII view. "
            "Connect your laptop to the drone's WiFi first."
        )

        # Drain the sniffer queue into session state
        sn_q: queue.Queue = st.session_state.sniffer_queue
        while True:
            try:
                pkt = sn_q.get_nowait()
                st.session_state.sniffer_packets.append(pkt)
            except queue.Empty:
                break
        if len(st.session_state.sniffer_packets) > 500:
            st.session_state.sniffer_packets = st.session_state.sniffer_packets[-500:]

        sn_t = st.session_state.sniffer_thread
        sn_alive = sn_t is not None and sn_t.is_alive()
        sn_pkt_count = len(st.session_state.sniffer_packets)

        # Status + controls
        pill_cls = "pill-on" if sn_alive else "pill-off"
        pill_txt = "● LIVE" if sn_alive else "● IDLE"
        st.markdown(
            f'<span class="sensor-pill {pill_cls}">{pill_txt}</span> '
            f'Packets captured: **{sn_pkt_count}**  '
            + (f'Status: _{sn_t.status}_' if sn_t else ''),
            unsafe_allow_html=True,
        )

        sn_ports_input = st.text_input(
            "UDP ports to listen on",
            value="8889, 14550, 14551, 11111, 4747, 5000, 8080, 9988",
            key="sn_ports_input",
        )

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            if st.button("▶ Start sniffer", key="sn_start",
                          disabled=sn_alive, type="primary"):
                try:
                    ports = [int(p.strip()) for p in sn_ports_input.split(",") if p.strip()]
                except ValueError:
                    ports = [14550, 8889, 11111]
                st.session_state.sniffer_queue  = queue.Queue(maxsize=500)
                st.session_state.sniffer_packets = []
                new_sn = PacketSnifferThread(st.session_state.sniffer_queue,
                                              ports=ports)
                new_sn.start()
                st.session_state.sniffer_thread = new_sn
                st.rerun()
        with sc2:
            if st.button("⏹ Stop sniffer", key="sn_stop", disabled=not sn_alive):
                if sn_t:
                    sn_t.stop()
                st.session_state.sniffer_thread = None
                st.rerun()
        with sc3:
            if st.button("Clear packets", key="sn_clear"):
                st.session_state.sniffer_packets = []
                st.rerun()

        st.divider()

        # Filters
        filt_proto = st.multiselect(
            "Filter by protocol",
            ["MAVLink", "Tello", "DJI", "UDP"],
            default=[],
            key="sn_proto_filter",
        )
        show_hex = st.checkbox("Show full hex dump", value=False, key="sn_show_hex")

        # Packet table
        pkts = st.session_state.sniffer_packets
        if filt_proto:
            pkts = [p for p in pkts if p["proto"] in filt_proto]

        if pkts:
            rows = []
            for p in reversed(pkts[-100:]):
                rows.append({
                    "Time": p["ts_str"],
                    "From": f"{p['src_ip']}:{p['src_port']}",
                    "DstPort": p["dst_port"],
                    "Proto": p["proto"],
                    "Len": p["length"],
                    "Detail": p.get("proto_detail", ""),
                    "Hex": p["hex_short"],
                    "ASCII": p["ascii_short"],
                })
            st.dataframe(rows, width="stretch", height=350)

            # Detail inspector
            st.markdown("##### Packet detail inspector")
            detail_idx = st.number_input(
                "Packet # (from end, 0 = latest)",
                min_value=0, max_value=max(0, len(st.session_state.sniffer_packets) - 1),
                value=0, key="sn_detail_idx",
            )
            if st.session_state.sniffer_packets:
                sel_pkt = st.session_state.sniffer_packets[-(detail_idx + 1)]
                pi1, pi2 = st.columns(2)
                with pi1:
                    st.markdown(
                        f"**Time:** {sel_pkt['ts_str']}  \n"
                        f"**Source:** `{sel_pkt['src_ip']}:{sel_pkt['src_port']}`  \n"
                        f"**Dest port:** `{sel_pkt['dst_port']}` "
                        f"({_PORT_LABELS.get(sel_pkt['dst_port'], '')})  \n"
                        f"**Protocol:** `{sel_pkt['proto']}`  \n"
                        f"**Length:** {sel_pkt['length']} bytes"
                    )
                    if "parsed" in sel_pkt:
                        st.markdown("**Decoded fields:**")
                        for k, v in sel_pkt["parsed"].items():
                            st.markdown(f"- **{k}:** `{v}`")
                with pi2:
                    st.markdown("**Hex dump (first 256 bytes):**")
                    st.code(sel_pkt["hex_dump"], language=None)
        else:
            st.info(
                "No packets captured yet. "
                "Start the sniffer and make sure you're connected to the drone's WiFi "
                "with the drone powered on."
            )

    # ── Tab 11: Signal Inspector ───────────────────────────────────────────────
    with tab_inspector:
        st.subheader("Signal Inspector — IQ Analysis & Decoder")
        st.caption(
            "Load a raw IQ capture (.cf32 or .npz) or use SDR results from this session. "
            "View constellation, waveform, spectrogram, and attempt bit decoding "
            "(ASK/FSK/Manchester — as in Scanner.py)."
        )

        # Load IQ from file upload or from last SDR capture
        ins_left, ins_right = st.columns([1, 2])
        with ins_left:
            st.markdown("##### Load IQ data")
            load_method = st.radio("Source", ["Upload .cf32/.npz file",
                                               "Last SDR capture in session"],
                                   key="ins_load_method")
            if load_method == "Upload .cf32/.npz file":
                uploaded = st.file_uploader(
                    "Drop a .cf32 or .npz file here",
                    type=["cf32", "npz"], key="ins_upload",
                )
                ins_fs = st.number_input("Sample rate (Hz)", value=2_400_000.0,
                                          step=100_000.0, key="ins_fs_upload")
                ins_cf = st.number_input("Center freq (Hz)", value=2_437_000_000.0,
                                          step=1_000_000.0, key="ins_cf_upload")
                if uploaded and st.button("Load file", key="ins_load_btn"):
                    try:
                        raw = uploaded.read()
                        if uploaded.name.endswith(".cf32"):
                            arr = np.frombuffer(raw, dtype=np.float32)
                            if arr.size % 2 == 0:
                                iq = arr[0::2] + 1j * arr[1::2]
                            else:
                                iq = arr[:-1:2] + 1j * arr[1:-1:2]
                            iq = iq.astype(np.complex64)
                        else:
                            npz = np.load(uploaded)
                            iq = npz["iq"].astype(np.complex64)
                            ins_fs = float(npz.get("sample_rate", ins_fs))
                            ins_cf = float(npz.get("center_freq_hz", ins_cf))
                        st.session_state.inspector_iq = iq
                        st.session_state.inspector_fs = ins_fs
                        st.session_state.inspector_cf_hz = ins_cf
                        st.session_state.inspector_decode = None
                        st.success(f"Loaded {iq.size} samples — {human_freq(ins_cf)} @ {human_bw(ins_fs)}ps")
                    except Exception as exc:
                        st.error(f"Load failed: {exc}")
            else:
                results_list = list(st.session_state.results)
                if results_list:
                    last_r = results_list[-1]
                    st.info(
                        f"Last SDR result: {human_freq(last_r.center_freq_hz)} "
                        f"SNR={last_r.snr_db:.1f} dB"
                    )
                    if st.button("Use last SDR result", key="ins_use_sdr"):
                        # Re-capture IQ for the last detected frequency
                        try:
                            iq = hackrf_capture(
                                last_r.center_freq_hz, 2.4e6, 2.4e6, 0.25, 20
                            )
                            if iq is not None and iq.size > 0:
                                st.session_state.inspector_iq = iq
                                st.session_state.inspector_fs = 2.4e6
                                st.session_state.inspector_cf_hz = last_r.center_freq_hz
                                st.session_state.inspector_decode = None
                                st.success(f"Captured {iq.size} samples")
                        except Exception as exc:
                            st.error(f"Re-capture failed: {exc}")
                else:
                    st.info("No SDR results in this session. Run a scan first.")

            st.divider()
            iq = st.session_state.inspector_iq
            if iq is not None and iq.size > 0:
                ins_fs = st.session_state.inspector_fs
                ins_cf = st.session_state.inspector_cf_hz
                st.markdown("##### Decode settings")
                dec_mode = st.selectbox("Mode", ["auto", "ask", "fsk", "manchester"],
                                         key="ins_dec_mode")
                if st.button("Decode signal", key="ins_decode_btn", type="primary"):
                    with st.spinner("Decoding…"):
                        result = decode_iq_quick(iq, ins_fs, dec_mode)
                    st.session_state.inspector_decode = result

        with ins_right:
            iq = st.session_state.inspector_iq
            if iq is not None and iq.size > 0:
                ins_fs = st.session_state.inspector_fs
                ins_cf = st.session_state.inspector_cf_hz

                st.markdown(
                    f"Loaded: **{iq.size}** samples · "
                    f"{human_freq(ins_cf)} · {human_bw(ins_fs)}ps"
                )

                view_tab_spec, view_tab_wave, view_tab_const, view_tab_spec2 = st.tabs([
                    "📈 Spectrum", "〰️ Waveform", "⊙ Constellation", "🌊 Spectrogram",
                ])
                with view_tab_spec:
                    freq_axis, spec_db = compute_spectrum(iq, ins_fs)
                    freq_mhz = (freq_axis + ins_cf) / 1e6
                    fig_sp = go.Figure()
                    fig_sp.add_trace(go.Scatter(
                        x=freq_mhz, y=spec_db,
                        mode="lines", line=dict(color="#0ea5e9", width=1),
                        name="Spectrum",
                    ))
                    peak_db, noise_db, snr_db, _, occ_bw = extract_metrics(freq_axis, spec_db)
                    fig_sp.add_hline(y=noise_db, line_dash="dot", line_color="#64748b",
                                     annotation_text=f"Noise {noise_db:.1f} dB")
                    fig_sp.update_layout(
                        paper_bgcolor="#07111a", plot_bgcolor="#0c1722",
                        font=dict(color="#cbd5e1"), height=300,
                        xaxis=dict(title="Frequency (MHz)", gridcolor="#1e293b"),
                        yaxis=dict(title="Power (dB)", gridcolor="#1e293b"),
                        margin=dict(l=60, r=20, t=20, b=40),
                    )
                    st.plotly_chart(fig_sp, width="stretch")
                    sm1, sm2, sm3 = st.columns(3)
                    sm1.metric("Peak", f"{peak_db:.1f} dB")
                    sm2.metric("SNR", f"{snr_db:.1f} dB")
                    sm3.metric("Occ. BW", human_bw(occ_bw))

                with view_tab_wave:
                    st.plotly_chart(
                        build_waveform_fig(iq, ins_fs, ins_cf),
                        width="stretch",
                    )

                with view_tab_const:
                    st.plotly_chart(
                        build_constellation_fig(iq, ins_cf),
                        width="stretch",
                    )

                with view_tab_spec2:
                    st.plotly_chart(
                        build_detail_spectrogram_fig(iq, ins_fs, ins_cf),
                        width="stretch",
                    )

                # Decode results
                dec = st.session_state.inspector_decode
                if dec:
                    st.divider()
                    st.markdown("##### Decode result")
                    conf_pct = int(dec["confidence"] * 100)
                    conf_color = "#16a34a" if conf_pct >= 60 else "#d97706" if conf_pct >= 30 else "#dc2626"
                    st.markdown(
                        f'<span style="font-weight:700;color:{conf_color};">'
                        f'[{dec["mode"]}]</span> '
                        f'confidence **{conf_pct}%** · '
                        f'symbol rate **{human_bw(dec["symbol_rate_hz"])}** baud',
                        unsafe_allow_html=True,
                    )
                    st.caption(dec["notes"])

                    d1, d2 = st.columns(2)
                    with d1:
                        st.markdown("**Bit stream (first 64 bits):**")
                        st.code(dec["bit_preview"], language=None)
                        st.markdown("**ASCII preview:**")
                        st.code(dec["ascii_preview"], language=None)
                    with d2:
                        st.markdown("**Hex preview (first 32 bytes):**")
                        st.code(dec["hex_preview"], language=None)
                        byte_vals = dec.get("byte_values", np.zeros(0, dtype=np.uint8))
                        if byte_vals.size > 0:
                            st.markdown("**Full hex dump:**")
                            st.code(format_hex_dump(bytes(byte_vals.tolist())),
                                    language=None)
            else:
                st.info("Load an IQ capture using the panel on the left to begin analysis.")

        # Drain sniffer queue and auto-rerun if sniffer is alive
        if st.session_state.sniffer_thread and st.session_state.sniffer_thread.is_alive():
            if not st.session_state.scanning:
                time.sleep(0.5)
                st.rerun()

    # ── Auto-refresh when scanning ────────────────────────────────────────────
    if st.session_state.scanning:
        t = st.session_state.scan_thread
        if t and not t.is_alive():
            st.session_state.scanning = False
            st.session_state.scan_thread = None
            st.warning(f"Scan thread stopped: {t.status}")
        time.sleep(0.4)
        st.rerun()


if __name__ == "__main__":
    main()
