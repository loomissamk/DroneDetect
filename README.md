# DroneDetect: Spatiotemporal Classification of Drone RF Signals from Raw IQ Data

**Samuel Loomis**
Technical Research Statement

---

## Abstract

This project investigates the problem of identifying consumer unmanned aerial systems (UAS) using passive radio‑frequency (RF) sensing. Rather than relying on protocol decoding or handcrafted spectral features, DroneDetect operates directly on raw in‑phase and quadrature (IQ) samples captured from software‑defined radios. The core contribution is a spatiotemporal deep‑learning architecture that combines convolutional feature extraction, gated recurrent sequence modeling, and transformer‑based attention to classify RF emitters over short time horizons.

The work is motivated by the observation that drone RF signatures are not static patterns but evolving behaviors. Accurate identification therefore requires models that can jointly represent local signal structure and longer‑range temporal dynamics. This document describes the dataset used, the architectural decisions made, and the rationale behind the final model design.

---

## 1. Background and Motivation

The rapid proliferation of commercial off‑the‑shelf (COTS) drones has created demand for passive, non‑cooperative identification methods. RF‑based approaches are attractive because they operate without line‑of‑sight constraints and do not require active interrogation. However, RF environments are inherently noisy, non‑stationary, and heterogeneous, making classical rule‑based or feature‑engineered approaches brittle.

Recent advances in deep learning for the physical layer suggest that models trained directly on IQ data can learn discriminative signal representations that outperform handcrafted pipelines. This project adopts that philosophy and extends it by explicitly modeling temporal structure, treating drone identification as a spatiotemporal learning problem rather than a static classification task.

---

## 2. Dataset Provenance

### 2.1 Source Dataset

The initial training and evaluation baseline for DroneDetect was derived from the publicly available **Noisy Drone RF Signal Classification v2** dataset hosted on Kaggle. The dataset consists of raw IQ recordings collected from multiple consumer drone and remote‑controller systems under varying noise conditions.

Key characteristics of the dataset include:

* Access to raw IQ samples rather than precomputed spectrograms
* Multiple transmitter classes corresponding to distinct drone or controller types
* Noise augmentation that reflects realistic RF capture conditions

This dataset was selected specifically because it preserves both fine‑grained signal structure and temporal continuity, enabling end‑to‑end learning directly from the physical layer.

### 2.2 Class Structure

The trained checkpoint currently used for inference (`best_model.pt`) is configured for **seven** classes. This is verified at load time by the model-check script, which infers the class count directly from the final classifier layer and binds UI labels accordingly.

**Active labels (7):**

* DJI
* Futaba T14
* Futaba T7
* Graupner
* FrSky Taranis
* Turnigy
* Noise (non-drone / background RF)

The explicit **Noise** class is a deliberate design choice: a practical RF system must be able to reject non-target RF activity rather than force every observation into a drone/controller label.

---

## 3. Problem Formulation

Given a window of raw IQ samples (x \in \mathbb{R}^{2 \times N}), the objective is to infer the emitting source class (y) while accounting for both instantaneous signal characteristics and their evolution over time.

Unlike image‑based classification, where spatial structure is often sufficient, RF signals exhibit discriminative features across multiple temporal scales:

* Short‑term structure (e.g., modulation artifacts, transient bursts)
* Medium‑term dynamics (e.g., packet framing, control cadence)
* Longer‑term behavior (e.g., hopping patterns, repetition intervals)

DroneDetect is designed to explicitly capture these scales within a single unified model.

---

## 4. Model Architecture

DroneDetect uses a deliberately staged spatiotemporal architecture: convolutional compression for local RF structure, gated recurrence for sequential stability, and self-attention for global context within each observation window.

### 4.1 Convolutional Feature Extraction (1D CNN)

The front end consists of 1D convolutions applied directly to the raw IQ time series. These layers learn discriminative local structure in the signal (e.g., burst shapes, phase/amplitude patterns, and hardware/chain-induced artifacts) without requiring protocol decoding.

**Implementation (as trained):**

* `Conv1d(2 → 64, kernel=5, stride=1, padding=2) + ReLU`
* `Conv1d(64 → 128, kernel=7, stride=2, padding=3) + ReLU`
* `Conv1d(128 → 256, kernel=9, stride=2, padding=4) + ReLU`
* `AdaptiveMaxPool1d(64)`

The adaptive pooling stage aggressively reduces sequence length early to keep the temporal stack computationally feasible.

### 4.2 Nonlinear Activations (ReLU, intentionally)

The model uses **ReLU** throughout the convolutional stack and the classifier head. This is an intentional design decision, not a placeholder: ReLU is computationally efficient, widely characterized in deep optimization, and tends to produce stable training dynamics in convolutional front ends paired with sequence models. In this setting, ReLU promotes selective (sparser) feature activations after convolutional filtering, which is useful when learning discriminative structure directly from raw IQ under variable SNR.

### 4.3 Temporal Modeling (GRU)

After convolutional compression, temporal dependencies are modeled with a multi-layer GRU:

* `GRU(input_size=256, hidden_size=256, num_layers=3, dropout=0.3, batch_first=True)`

GRUs are used to capture short- to medium-range dynamics such as framing/cadence and consistent timing behaviors that are not reliably represented by local convolutional filters alone.

### 4.4 Global Context Aggregation (Transformer Encoder)

To capture longer-range relationships within the pooled observation window, the GRU output is passed through a transformer encoder:

* `TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, dropout=0.3)`
* `TransformerEncoder(num_layers=3)`

Self-attention enables the model to re-weight salient temporal segments inside the window (e.g., particularly informative bursts) without being limited to strictly local recurrence.

### 4.5 Classification Head

The classifier uses a compact fully connected head:

* `Linear(256 → 128) + ReLU`
* `Dropout(p=0.4)`
* `Linear(128 → num_classes)`

### 4.6 Verified Tensor Flow (matches `CheckModel.py` / loaded checkpoint)

The following shapes reflect the **actual forward pass** of the trained model used in the checkpoint:

* **Input IQ:** `x ∈ [B, 2, N]`
* **CNN output:** `cnn(x) ∈ [B, 256, 64]` (after `AdaptiveMaxPool1d(64)`)
* **Permute for sequence modeling:** `cnn(x).permute(0,2,1) ∈ [B, 64, 256]`
* **GRU output:** `rnn_out ∈ [B, 64, 256]`
* **Transformer output:** `attn_out ∈ [B, 64, 256]`
* **Global average pool over time:** `mean(attn_out, dim=1) ∈ [B, 256]`
* **Logits:** `fc(...) ∈ [B, 7]`

At runtime, the checkpoint loads cleanly with no missing or unexpected keys, and the class count is inferred from the classifier layer (`fc.3.weight`) as **7**, matching the active UI label list.

---

## 5. Data Representation and Training Setup

Data Representation and Training Setup

All IQ samples are normalized and stored in tensor format with shape:

```
[2, N]
```

where the first dimension corresponds to in‑phase and quadrature components. Training samples are serialized as PyTorch `.pt` files to support efficient loading and batching.

The training loop employs:

* AdamW optimization
* Gradient clipping for recurrent stability
* Class balancing to address dataset skew
* Periodic checkpointing to preserve intermediate states

The current trained checkpoint (`best_model.pt`) loads cleanly with no missing or unexpected parameters and runs on CUDA‑enabled hardware.

---

## 6. Inference and Evaluation

The trained model is integrated into a live scanning pipeline that captures RF data via HackRF or SoapySDR‑compatible devices. Incoming IQ windows are processed in near real time, and classification outputs are gated by confidence thresholds to reduce false positives.

This end‑to‑end pipeline demonstrates the feasibility of deploying spatiotemporal RF classifiers on commodity hardware.

---

## 7. Contributions and Significance

The primary contributions of this work are:

* Demonstration of end‑to‑end learning from raw IQ data for drone identification
* A unified spatiotemporal architecture combining CNNs, GRUs, and transformers
* Practical integration with live SDR capture pipelines

More broadly, this project supports the view that RF emitter identification is fundamentally a sequence modeling problem and benefits from modern attention‑based architectures.

---

## 8. Ethical and Legal Considerations

This research is intended for defensive awareness, academic study, and personal experimentation with owned or authorized equipment. All RF capture must comply with applicable spectrum regulations and privacy laws.

---

## References

1. S. Gluege, *Noisy Drone RF Signal Classification v2*, Kaggle Dataset.
2. T. J. O’Shea and J. Hoydis, “An Introduction to Deep Learning for the Physical Layer,” *IEEE Transactions on Cognitive Communications and Networking*, 2017.
3. D. Hendrycks and K. Gimpel, “Gaussian Error Linear Units (GELUs),” arXiv:1606.08415.
4. J. Chung et al., “Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling,” arXiv:1412.3555.
5. A. Vaswani et al., “Attention Is All You Need,” *NeurIPS*, 2017.
