# 🎙️ Trigger Word Detection — "Help"

A real-time trigger word detection system that identifies the keyword **"Help"** from live audio input using a hybrid **CNN + GRU** deep learning model trained on a custom dataset.

---

## 📌 Project Overview

This project builds a lightweight keyword spotting pipeline that:
- Listens to real-time microphone audio
- Detects whether the spoken word is **"Help"** or any random word
- Uses spectrogram-based feature extraction fed into a CNN-GRU neural network

---

## 🗂️ Dataset

The dataset was created from scratch with two classes:

| Label | Class |
|-------|-------|
| `0` | Random word |
| `1` | "Help" (trigger word) |

**Specs:**
- Format: `.wav` (mono)
- Duration: 1 second per recording
- Sample Rate: 44,100 Hz
- Bit Rate: 814 kb/s

---

## 🔊 Feature Extraction — Spectrogram

Raw audio is converted into a **spectrogram** (frequency vs. time representation) using `matplotlib`'s `specgram` function:

- FFT Window Length: 200 samples
- Window Overlap: 120 samples

This transforms 1D audio signals into 2D feature maps suitable for convolutional processing.

---

## 🧠 Model Architecture

The model combines **Conv1D** for local pattern extraction and **GRU** layers for temporal sequence modeling.

```
Input: (time_steps, features)
    │
    ▼
Conv1D (196 filters, kernel=15, stride=4)
    → Batch Normalization → ReLU → Dropout(0.5)
    │
    ▼
GRU Layer 1 (128 units, return_sequences=True)
    → Dropout(0.5) → Batch Normalization
    │
    ▼
GRU Layer 2 (128 units, return_sequences=True)
    → Dropout(0.5) → Batch Normalization
    │
    ▼
Global Average Pooling
    │
    ▼
Dense (64 units, ReLU) → Dropout(0.5)
    │
    ▼
Dense Output Layer (Binary Classification)
```

**Why this architecture?**
- Conv1D captures local phonetic patterns in the spectrogram
- GRU layers model the temporal progression of speech
- Dropout + Batch Normalization at every stage prevent overfitting

---

## 🎧 Real-Time Audio Processing

The inference pipeline uses **PyAudio** for live audio streaming:

| Parameter | Value |
|-----------|-------|
| Chunk Size | 1024 samples |
| Format | 16-bit integer |
| Channels | Mono |
| Sample Rate | 44,100 Hz |

**How it works:**
1. A callback function continuously monitors microphone input
2. Audio levels are checked against a silence threshold
3. Audio chunks are stored in a circular buffer
4. Buffered data is pushed to a processing queue
5. The model runs inference on each queued chunk
6. The system automatically stops after a **60-second timeout**

---


## 📦 Requirements

Key libraries used:
- `tensorflow` / `keras` — model building and training
- `librosa` — audio processing
- `pyaudio` — real-time microphone input
- `matplotlib` — spectrogram generation
- `numpy` — numerical operations

---


## 💡 Key Concepts

- **Keyword Spotting (KWS):** Detecting a specific word in a continuous audio stream
- **Spectrogram:** A visual representation of audio frequencies over time, used as model input
- **CNN + GRU Hybrid:** Convolutional layers extract local features; recurrent layers capture temporal patterns
- **Circular Buffer:** Ensures smooth real-time processing without gaps in audio capture

---

## 🙋 About

Built as a learning project to explore audio machine learning, deep learning architectures for sequential data, and real-time inference pipelines.
