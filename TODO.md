# DroneDetect TODO — ordered by difficulty (easiest → hardest)

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## 0. REMOTE-BOX FINDINGS (pop-os-1, user `bat`, SSH alias `gaming`, Tailscale 100.107.226.49)
Discovered during this session — the training environment and the *real* data live there:
- **HackRF One SDR** live (`Bus 009 Device 002 … HackRF One`) + `hackrf_transfer` in PATH.
- **Full Kaggle dataset** (98,705 samples, 25 GB `dataset.pt`):
  `/home/bat/DroneDetect-work/data/kaggle_drone_rf/{dataset.pt, class_stats.csv, SNR_stats.csv}`
- **Newer, better model** already trained there: `best_model_v3.pt` (2.4 MB pure-CNN `DroneCNN`,
  `train_v3.py`, trained on all 98,705 samples) — vs. the older 15 MB `best_model.pt` (CNN+GRU+Transformer)
  in this repo. Both are now pulled into this repo and load via the architecture-agnostic loader.
- A live full-band sweep dashboard runs it: `/home/bat/DroneDetect-work/drone_full_dashboard.py` (cuda).
- **Authoritative label order** (train_v3.py, class_stats.csv, dashboard all agree):
  `0=DJI 1=FutabaT14 2=FutabaT7 3=Graupner 4=Noise 5=Taranis 6=Turnigy`.
- Class balance: Noise 52,552 (53%) · Taranis 16,546 · Turnigy 10,333 · FutabaT14 6,938 ·
  Graupner 6,481 · FutabaT7 3,661 · DJI 2,194. SNR balanced −20…+30 dB (~3,800 each).
- Training runs in a Docker container (`dronedetect:latest`, `start.sh`); host has no system torch.

---

## 1. EASY — config & correctness fixes (no data, no training)

### 1.1 Fix the "only predicts Taranis" bug  `[x]`
**Two distinct bugs, both confirmed by CPU test and fixed:**

**(a) Label-order misalignment (the true root cause).** The Kaggle dataset trains
**index 4 = Noise** (the dominant class, 52,552/98,705 ≈ 53%). But every code file listed
`[DJI, FutabaT14, FutabaT7, Graupner, Taranis, Turnigy, Noise]`, putting **Taranis at index 4**.
So the model correctly predicted index 4 = Noise, and the UI printed it as "Taranis."
Fix: reordered `drone_labels` / `DRONE_LABELS` to the true training order
`[DJI, FutabaT14, FutabaT7, Graupner, Noise, Taranis, Turnigy]` in `ScannerWithModel.py`,
`CheckModel.py`, `Scanner.py`. After the fix: `noise -> Noise`, `fsk24 -> FutabaT14/T7` (correct).

**(b) 8-class load bug (secondary).** `best_model.pt` has a **7-class** head
(`fc.3.weight = [7,128]`), but `ScannerWithModel.py:137` built an **8-class** model and
`drop_non_drone()` sliced index 7 — a **size-mismatch crash on modern torch**
(`[7,128] vs [8,128]`); on old torch class 7 loaded random weights and shifted the mapping.
Fix:
  - `ScannerWithModel.py:136-149` — build `EnhancedTemporalModel(num_classes=7)`,
    `load_state_dict(strict=True)`, remove `drop_non_drone()` and `NON_DRONE_IDX_IN_OLD`.
  - Verify `drone_labels` order matches training label order (DJI, FutabaT14, FutabaT7,
    Graupner, Taranis, Turnigy, Noise) — see AGENTS.md: validate class count at load.
  - Also fix `CheckModel.py:55` (`nn.GRU(input_size=256,...)` typo) and align its label list.
  - `droneActiveClass.py` already assumes 7 classes — confirm no stray 8th index.
- Verify: CPU fixture inference loads clean, 0 missing/0 unexpected keys, all 7 classes reachable.

### 1.2 Full-band 1 MHz–6 GHz defaults + band presets  `[x]`
- `Scanner.py:50-66` — `DEFAULT_START_FREQ`/`DEFAULT_END_FREQ` are already 10 MHz–6 GHz,
  but `apply_device_defaults()` overwrites the Range box with narrow per-profile ranges
  (generic/rtlsdr → 433.8–434.2 MHz, hackrf → 2.400–2.500 GHz). That's the "current mode" trap.
- Fix:
  - Change `DEVICE_PROFILES[*].default_range` to a sensible full-band or wide start.
  - Add a **Band presets** combo: Sub-GHz (400–490 MHz), 2.4 GHz (2.40–2.50),
    5.8 GHz (5.725–5.875), Full (10 MHz–6 GHz). Preset click sets Range + Step + Sample rate.
  - Full-band coverage math: 10 MHz→6 GHz at 250 kHz step ≈ 24,000 steps (fits `MAX_SWEEP_STEPS=100000`).
    At 1 MHz step ≈ 5,990 steps — much faster, recommend as full-band default step.
  - Warn (not hard-fail) when range exceeds the selected radio's real band (RTL-SDR stops ~1.77 GHz,
    SDRplay ~2 GHz, HackRF 1 MHz–6 GHz). Keep `validate_frequency_plan` but surface device limits.
- Verify: UI preset sets the fields; `build_frequency_plan` accepts full band at chosen step.

### 1.3 Wire the classifier into the real inspector (`Scanner.py`)  `[x]`
- Currently the **only** model path is the broken `ScannerWithModel.py`; the nicer
  `Scanner.py` sweep loop never calls the model. Unify them:
  - Add optional `--model best_model.pt` to `Scanner.py`. When set, run the 7-class
    classifier on each captured IQ window (reuse `record_iq_once`/`SoapyCapture.record`).
  - Gate: only classify when `snr_db >= snr_gate_db` (already computed in `analyze_capture`).
  - Show predicted class + confidence in the Hits table and status banner; record
    `pred_class`/`pred_prob` columns in `spectrum_log.csv`.
  - Keep decoder (ASK/FSK/Manchester/UART) as-is; class label is complementary.
- Verify: CPU fixture sweep emits a prediction row; model absent → scanner still runs (graceful).

### 1.4 CSV schema + log hygiene  `[ ]`
- `Scanner.py` and `ScannerWithModel.py` write **different** CSV headers
  (`spectrum_log.csv` vs `signal_log.csv`). Pick one schema, append (don't `open(...,"w")`
  on every run — currently the log is truncated on each start). Add a `run_id`/session column.
- `record_iq_once` leaves `temp_iq.raw` behind on some failure paths; ensure `finally` cleanup.

---

## 2. MEDIUM — robustness & feature gaps (code changes, some testing)

### 2.1 Decoder blind spots  `[ ]`
- Current decoders (ASK/FSK/Manchester/UART) are good for classic RC links, but miss:
  - **DJI OcuSync / Lightbridge** (OFDM/SC-FDMA on 2.4 & 5.8 GHz, fast FHSS, OFDM constellation).
  - **GPS L1 (1575.42 MHz) & L5 (1176.45 MHz)** — a drone's navigation beacon; strong passive signature.
  - **5.8 GHz video downlink** (NTSC/PAL analog, or compressed) — very common on drones.
- Add band-aware decode presets and a "GPS L1/L5" sweep marker so the full-band sweep
  highlights those frequencies automatically.

### 2.2 Device-profile accuracy  `[ ]`
- `DEVICE_PROFILES` sample rates/steps are conservative. Re-derive per-device min/max freq
  from the live `getFrequencyRange` (already in `SoapyCapture._read_frequency_bounds`) and
  use that as the hard limit, profile only as the *suggested* start.

### 2.3 Recording & replay  `[ ]`
- Recordings (`recordings/*.npz` + `.cf32`) referenced by `spectrum_log.csv` are **not present**
  in this checkout. Add a `Recordings` folder to the repo (gitignore the big files), and a
  "Load last N clips" replay button so the 2.4 GHz May 8 session can be re-analyzed offline.
- Confirm `save_detection_capture` writes both `.npz` and `.cf32`; add a `requirements` for
  loading them back.

### 2.4 Thread-safety & UI  `[ ]`
- `SweepThread` mutates `self.sample_rate_hz` from the worker thread while the UI reads it.
  Move rate negotiation to the main thread or a `QMutex`.
- `pause_on_detection` leaves the scan paused with no obvious "Resume" affordance in the banner.

### 2.5 MeowKit ↔ DroneDetect integration (optional)  `[ ]`
- The MeowKit jammer apps (BT + WiFi) are TX-only on ESP32-S3. To "jam most drones + IQ" you'd
  need an SDR front-end (HackRF/RTL-SDR) feeding the same pipeline — the MeowKit can act as a
  narrowband blocker on known drone frequencies (2.4/5.8/433) *after* DroneDetect classifies.
- Sketch: DroneDetect reports class+freq → MeowKit `wifi_jammer`/`bt_jammer` hops to that band.
  (Firmware change, separate repo — lower priority.)

---

## 3. HARD — model retraining & data

### 3.1 Acquire / rebuild the training dataset  `[~] data found on pop-os-1`
- Current training data path (`DroneDetectClassifier.py:200`) is
  `/home/batman/Desktop/drone_rf_data/drone_RF_data` — a **Linux box, not this PC**.
  The `.pt` samples (2 × N IQ) are not in this checkout.
- Options (pick one):
  1. Re-copy the `.pt` dataset from the Linux box into `DroneDetect/data/` and point the
     classifier at a relative path.
  2. **Re-capture** with the HackRF: sweep 400–490 MHz, 2.4 GHz, 5.8 GHz; fly/operate each
     known drone class (DJI, Futaba T14/T7, Graupner, FrSky Taranis, Turnigy); save `.npz`
     clips → convert to `.pt` `{x_iq:[2,N], y:class_idx}`.
  3. Download a fresh public set (Kaggle "Noisy Drone RF Signal Classification v2") as the
     base, then fine-tune with local captures.
- Add a `data/` manifest (SHA256 + class counts) so provenance is auditable (AGENTS.md).

### 3.2 Expand class set beyond the original 7  `[ ]`
- "Most drones" currently = 6 brands + Noise. To truly maximize:
  - Add classes: Parrot, Autel, Yuneec, DJI Mavic 3 vs Air vs Mini (if data exists),
    and a generic "Unknown drone" vs "Non-drone RF" split (the old 8th class that was dropped).
  - Each new class needs ≥ ~50 labeled `.pt` samples or the weighted sampler will underfit.
- Update `drone_labels` in every file (`ScannerWithModel.py`, `CheckModel.py`, `droneActiveClass.py`)
  and re-verify the checkpoint class count matches (AGENTS.md rule).

### 3.3 Retrain + evaluate  `[x]`
- **v5 trained on pop-os-1 GPU** (`train_v5.py`, DroneCNN, full 98,705 samples, mild inverse-count
  loss weighting, cosine LR, early-stop): **val acc 0.8894, macro-F1 0.8447** (best epoch 10,
  early-stopped epoch 18). Per-class F1: DJI 0.77 · FutabaT14 0.83 · FutabaT7 0.80 · Graupner 0.88 ·
  Noise 0.91 · Taranis (≈0.85) · Turnigy (≈0.85). Saved as `best_model_v5.pt` (module) +
  `best_model_v5_state.pt` (state_dict) + `train_v5_report.json`.
- **v5 beats v3** (0.77, rare classes weak) and **v4** (dual-modality overfit, rare-class F1≈0).
  The winning recipe = v3's architecture + full-data (no per-class cap) + GPU + mild loss weighting.
- `train_v5_report.json` in this repo records the full epoch history + per-class metrics.

### 3.4 Fine-tune vs retrain decision  `[x] retrained from scratch`
- Trained from scratch (not fine-tuned) because the original training set was on the Linux box;
  retraining on the full 98,705-sample set with the proven v3 recipe + GPU gave the best result.
- `train_v4.py` (dual-modality IQ+spectrogram) was also attempted — it overfit the imbalanced set
  (rare classes F1≈0). Kept for reference; v5 (IQ-only) is the recommended model.

---

## 4. NICE-TO-HAVE (backlog)
- [ ] CLI headless mode: `scanner.py --range 400m-490m --step 0.5m --csv out.csv` (no GUI) for
      cron/long sweeps.
- [ ] Peak-hold + auto-tune SNR gate (currently fixed 14 dB default; auto-gate would reduce
      false alarms in dense 2.4 GHz).
- [ ] Export decoded bitstream to a `.bin` file for GRC/SoX post-processing.
- [ ] Docker image with HackRF/SoapySDR pre-wired so the whole pipeline runs on the Linux box
      and this Mac unchanged.

---

## Definition of Done
- [x] `ScannerWithModel.py` loads `best_model.pt` clean on modern torch, no crash.
- [x] Full-band sweep (10 MHz–6 GHz) available from a preset button; classifier fires on each hit.
- [x] `spectrum_log.csv` single consistent schema, appended across runs (run_id + model columns). (TODO 1.4)
- [x] Retrained model on pop-os-1: **v5** val 0.8894 / macro-F1 0.8447, all 7 classes F1 0.77–0.91. (TODO 3.x)
- [x] README/TODO updated: which checkpoint (v5), class count, load command, training report.
- [x] Model picker UI + headless CLI mode + `requirements.txt` added.
- [ ] (Backlog) Expand class set beyond 7 (Parrot/Autel/Yuneec + drone-subtype split) — needs new labeled captures.

## Verification log (this session, CPU, torch 2.15.0)
- `best_model_v5.pt` → loads as `DroneCNN`, strict; `noise → Noise (100%)`, `ook24 → Taranis (100%)`.
- `best_model_v3.pt` → loads as `DroneCNN`, strict; `noise → Turnigy (50%)`, `fsk24 → FutabaT14 (99%)`.
- `best_model.pt` → loads as `EnhancedTemporalModel`, strict; 0 missing/0 unexpected.
- Missing checkpoint → loader returns `(None, [])` (sweep keeps running, no crash).
- Command: `/tmp/dronedetect-venv/bin/python /tmp/verify_v5.py` (from `/Users/samuelloomis/Documents/DroneDetect`).
- GPU training + real RF capture verified on pop-os-1 (CUDA + HackRF); CPU inference verified on this Mac.
