# HMM_Activity_Recognition


# 🏃 Hidden Markov Model for Human Activity Recognition

> Formative 2 — Sensor Data & Probabilistic Modelling  

---

## 📋 Overview

This project implements a **Hidden Markov Model (HMM)** to infer human activity states from raw smartphone accelerometer and gyroscope signals. The model is trained using the **Baum–Welch algorithm** and decoded using the **Viterbi algorithm**, achieving **78.6% overall accuracy** on unseen test data.

**Activities classified:** Still · Standing · Walking · Jumping

---

## 📱 Data Collection

| Detail | Value |
|---|---|
| App | Sensor Logger |
| Sampling Rate | 100 Hz (10 ms interval) |
| Sensors | Accelerometer (x, y, z) · Gyroscope (x, y, z) |
| Total Sessions | 50 |
| Total Windows | 725 (after 1s windowing, 50% overlap) |

**Activity breakdown:**

| Activity | Sessions | Windows (Train) | Windows (Test) |
|---|---|---|---|
| Still    | 12 | 148 | 9  |
| Standing | 12 | 167 | 6  |
| Walking  | 15 | 229 | 22 |
| Jumping  | 11 | 139 | 5  |
| **Total**| **50** | **683** | **42** |

---

## ⚙️ Features Extracted (25 total per window)

**Time-domain (22):** mean, variance, RMS — per axis (ax, ay, az, gx, gy, gz) · Signal Magnitude Area (SMA) · Pairwise correlations (ax-ay, ax-az, ay-az)

**Frequency-domain (3):** dominant frequency · spectral energy · low/high band energy ratio (FFT of accelerometer magnitude)

All features Z-score normalised (fit on training data only).

---

## 🤖 Model

| Component | Detail |
|---|---|
| Hidden States | 4 (Still, Standing, Walking, Jumping) |
| Emission Model | Gaussian (diagonal covariance) |
| Training Algorithm | Baum–Welch (EM), tol = 1e-4, max 200 iterations |
| Decoding Algorithm | Viterbi |
| Convergence | 26 iterations · log-likelihood = 14,879.28 |

---

## 📊 Results (Unseen Test Data — 42 windows)

| Activity | Sensitivity | Specificity | F1-Score | Accuracy |
|---|---|---|---|---|
| Standing | 1.000 | 0.750 | 0.57 | 0.786 |
| Walking  | 1.000 | 1.000 | 1.00 | 1.000 |
| Jumping  | 1.000 | 1.000 | 1.00 | 1.000 |
| Still    | 0.000 | 1.000 | 0.00 | 0.786 |
| **Overall** | — | — | **0.72** (weighted) | **78.6%** |

> ⚠️ Still was fully misclassified as Standing due to a state collapse during training — no HMM state was assigned exclusively to Still. See the report for discussion.

---

## 🚀 How to Run

1. **Mount Google Drive** in Colab and place the `HMM_data/` folder at:
   ```
   /content/drive/MyDrive/HMM_data
   ```

2. **Open the notebook** `HMM_Activity_Recognition.ipynb` in Google Colab.

3. **Run all cells** in order. The notebook will:
   - Install `hmmlearn`
   - Load and merge all 50 session folders
   - Extract 25 features per window
   - Train the HMM with Baum–Welch
   - Decode test sessions with Viterbi
   - Output evaluation metrics and save 7 figures

4. **Outputs saved:**
   - `hmm_model.pkl` — trained model + scaler
   - `evaluation_results.csv` — per-class metrics
   - `raw_signals.png`, `feature_distributions.png`, `convergence.png`, `transition_matrix.png`, `emission_means.png`, `decoded_sequence.png`, `confusion_matrix.png`

---

## 🛠️ Dependencies

```
hmmlearn
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
```

Install with:
```bash
pip install hmmlearn numpy pandas scipy scikit-learn matplotlib seaborn
```

---

