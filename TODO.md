# DroneDetect TODO — ordered by difficulty (easiest → hardest)

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## 1. EASY — config & correctness fixes (no data, no training)

### 1.1 Fix the "only predicts Taranis" bug  `[ ]`
- Root cause (confirmed by test): checkpoint `best_model.pt` has a **7-class** head
  (`fc.3.weight = [7,128]`), but `ScannerWithModel.py:137` builds an **8-class** model and
  `drop_non_drone()` slices index 7. On modern torch this **crashes on startup**
  (`size mismatch fc.3.weight [7,128] vs [8,128]`); on old torch class 7 loads random
  weights and the label→index mapping shifts, so one class dominates.
- Fix:
  - `ScannerWithModel.py:136-149` — build `EnhancedTemporalModel(num_classes=7)`,
    `load_state_dict(strict=True)`, remove `drop_non_drone()` and `NON_DRONE_IDX_IN_OLD`.
  - Verify `drone_labels` order matches training label order (DJI, FutabaT14, FutabaT7,
    Graupner, Taranis, Turnigy, Noise) — see AGENTS.md: validate class count at load.
  - Also fix `CheckModel.py:55` (`nn.GRU(input_size=256,...)` typo) and align its label list.
  - `droneActiveClass.py` already assumes 7 classes — confirm no stray 8th index.
- Verify: CPU fixture inference loads clean, 0 missing/0 unexpected keys, all 7 classes reachable.

### 1.2 Full-band 1 MHz–6 GHz defaults + band presets  `[ ]`
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

### 1.3 Wire the classifier into the real inspector (`Scanner.py`)  `[ ]`
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

### 3.1 Acquire / rebuild the training dataset  `[ ]`
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

### 3.3 Retrain + evaluate  `[ ]`
- `DroneDetectClassifier.py` training loop is solid (AdamW, AMP, grad-clip, weighted sampler,
  ReduceLROnPlateau) but:
  - `data_dir` is hardcoded to the Linux path — make it a CLI arg / env var.
  - No early stopping; `num_epochs=100` is the only brake. Add patience-based stop.
  - `best_model.pt` is saved with `torch.save(model, ...)` (full module) — keep that, but also
    write a `best_model_state.pt` (state_dict) for portable loading.
  - Evaluation: add a per-class **recall/precision** table and a confusion matrix PNG to the repo.
- Train on the Linux GPU box (CUDA) or Apple Silicon via MPS (`device = mps`), then copy
  `best_model.pt` back. Re-run `CheckModel.py` + the CPU fixture to confirm the new checkpoint
  loads with the *new* class count and the `Scanner.py` integration still classifies.

### 3.4 Fine-tune vs retrain decision  `[ ]`
- If the original 7-class data is intact: **fine-tune** (low LR, 20–50 epochs) to preserve the
  working DroneDetect weights and just sharpen Taranis/DJI recall.
- If data is lost/corrupt: **retrain from scratch** with the expanded class set from 3.2.
- Either way: freeze CNN+GRU first, train only `fc` head for 10 epochs, then unfreeze — prevents
  catastrophic forgetting on the small dataset.

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
- [ ] `ScannerWithModel.py` loads `best_model.pt` clean on modern torch, no crash.
- [ ] Full-band sweep (10 MHz–6 GHz) runs from a preset button, classifier fires on each hit.
- [ ] `spectrum_log.csv` has a single consistent schema, appended across runs.
- [ ] New `best_model.pt` (retrained) passes `CheckModel.py` + a 3-signal CPU fixture.
- [ ] README updated: which checkpoint, class count, exact load command (AGENTS.md rule).
