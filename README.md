## Hidden Markov Model for Human Activity Recognition

Formative 2. Sensor Data & Probabilistic Modelling


## 1. Overview

This project implements a **Hidden Markov Model (HMM)** to recognise human activities from raw smartphone motion data. Using 3‑axis accelerometer and gyroscope readings collected from everyday phones, the model predicts which activity a person is performing at each time window.

- **Activities:** Still · Standing · Walking · Jumping  
- **Model:** Gaussian HMM trained with **Baum-Welch** and decoded with **Viterbi**  
- **Evaluation:** Window-level classification on held-out sessions, achieving **78.6% overall accuracy** on unseen data

The full workflow is implemented in the notebook `notebook/HMM_Activity_Recognition.ipynb`.


## 2. Data

### 2.1 Collection setup

| Detail | Value |
|---|---|
| Devices | iPhone 11 · Pixel 4 |
| App | Sensor Logger |
| Sampling rate | 100 Hz (10 ms interval) |
| Sensors | Accelerometer (x, y, z) · Gyroscope (x, y, z) |
| Total sessions | 50 |

Each recording session corresponds to one subject performing a **single activity** (Still, Standing, Walking, or Jumping). Data from the accelerometer and gyroscope are timestamped and later merged into a single time series per session.

### 2.2 Windowing

Signals are converted into fixed-length windows before feature extraction:

- **Window length:** 1 s (100 samples at 100 Hz)  
- **Overlap:** 50% (step size = 0.5 s)  
- **Label:** each window is assigned the activity label of its parent session

After windowing, there are **725 windows** in total.

### 2.3 Activity breakdown

| Activity | Sessions | Windows (Train) | Windows (Test) |
|---|---|---|---|
| Still    | 12 | 148 | 9  |
| Standing | 12 | 167 | 6  |
| Walking  | 15 | 229 | 22 |
| Jumping  | 11 | 139 | 5  |
| **Total**| **50** | **683** | **42** |

The last session of each activity is held out as **unseen test data**; all remaining sessions are used for training.


## 3. Feature Engineering

For every 1‑second window, we compute a compact set of features that characterise both the **magnitude** and **periodicity** of motion.

### 3.1 Time‑domain features (22)

- **Per-axis statistics** for each of the 6 channels (ax, ay, az, gx, gy, gz):
  - Mean  
  - Variance  
  - Root-mean-square (RMS)
- **Signal Magnitude Area (SMA)** for the accelerometer
- **Pairwise correlations** between accelerometer axes: ax-ay, ax-az, ay-az

These features capture posture (gravity vector orientation), overall motion energy, and coordination between axes.

### 3.2 Frequency‑domain features (3)

Computed from the FFT of the accelerometer magnitude:

- **Dominant frequency** (Hz)  
- **Spectral energy**  
- **Energy ratio** between a low‑frequency band (0-3 Hz) and a higher band (3-10 Hz)

These features separate quasi‑static postures (Still, Standing) from periodic movements (Walking, Jumping).

### 3.3 Normalisation

All features are **Z‑score normalised**:

- Fit `StandardScaler` **only on the training windows**  
- Apply the learned scaling to both training and test windows

This prevents high‑variance features from dominating the Gaussian emission model.


## 4. HMM Model

The activity classifier is a **Gaussian HMM** defined over the engineered feature vectors.

| Component | Detail |
|---|---|
| Hidden states | 4 (Still, Standing, Walking, Jumping) |
| Observations | 25‑dimensional feature vector per window |
| Emission model | Multivariate Gaussian with **diagonal covariance** |
| Training | Baum-Welch (EM), tolerance = 1e‑4, max 200 iterations |
| Decoding | Viterbi algorithm |
| Convergence | 26 iterations · final log‑likelihood ≈ 14,879.28 |

Each hidden state is aligned with one activity class by analysing how often its decoded state sequence co‑occurs with the true labels on the training data (see the notebook for details on the mapping strategy).


## 5. Evaluation and Results

Evaluation is performed on the **held‑out windows** from the last session of each activity (42 windows in total). Metrics are computed at the **window level**.

### 5.1 Per‑class metrics (unseen test data)

| Activity | Sensitivity | Specificity | F1‑score | Accuracy |
|---|---|---|---|---|
| Standing | 1.000 | 0.750 | 0.57 | 0.786 |
| Walking  | 1.000 | 1.000 | 1.00 | 1.000 |
| Jumping  | 1.000 | 1.000 | 1.00 | 1.000 |
| Still    | 0.000 | 1.000 | 0.00 | 0.786 |
| **Overall** | — | — | **0.72** (weighted) | **78.6%** |

The model recognises **Walking** and **Jumping** almost perfectly. The main challenge is distinguishing **Still** from **Standing**: in this particular configuration, Still windows are consistently classified as Standing, leading to 0 sensitivity for Still but high sensitivity for Standing. The notebook discusses this behaviour and experiments with improved state‑to‑activity mappings.


## 6. Repository Structure

- `notebook/HMM_Activity_Recognition.ipynb`: Main end‑to‑end workflow (data loading, feature extraction, HMM training, visualisation, evaluation)  
- `data/HMM_data/`: Raw CSV sessions from Sensor Logger, grouped by activity and session index  
- `requirements.txt`: Python dependencies


## 7. How to Run

### 7.1 Google Colab

1. **Upload data to Drive** and place the `HMM_data/` folder at:
   ```
   /content/drive/MyDrive/HMM_data
   ```
2. Open `notebook/HMM_Activity_Recognition.ipynb` in Google Colab.
3. Set `COLAB = True` at the top of the notebook if needed.
4. Run all cells from top to bottom. The notebook will:
   - Install the required Python packages (e.g. `hmmlearn`)
   - Load and merge all 50 session folders
   - Extract 25 features per 1‑second window
   - Train the HMM with Baum-Welch
   - Decode test sessions with Viterbi
   - Compute and print evaluation metrics
   - Save all generated figures and results to the working directory

### 7.2 Local environment

1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the folder `data/HMM_data/` is present and matches the structure expected by the notebook.
4. Open `notebook/HMM_Activity_Recognition.ipynb` in VS Code, Jupyter Lab, or another notebook environment.
5. Run all cells in order.


## 8. Outputs

Running the notebook produces the following artefacts:

- `hmm_model.pkl` — trained HMM, feature scaler, and mapping information  
- `evaluation_results.csv` — per‑class metrics on the test set  
- Figures for qualitative analysis:
  - `raw_signals.png` — example raw accelerometer/gyroscope traces per activity  
  - `feature_distributions.png` — distributions of normalised features by activity  
  - `convergence.png` — Baum-Welch log‑likelihood over iterations  
  - `transition_matrix.png` — learned state transition probabilities  
  - `emission_means.png` — Gaussian means per state and feature  
  - `decoded_sequence.png` — true vs predicted activity sequences on test data  
  - `confusion_matrix.png` — confusion matrix on unseen test windows


## 9. Limitations and Future Work

- **Data diversity:** data comes from a small number of sessions and devices (iPhone 11, Pixel 4). Extending to more users and activities would improve robustness.  
- **Confusion between low‑motion states:** Still and Standing are difficult to separate based solely on motion, especially when the subject naturally sways while “standing still”. Additional postural or orientation features could help.  
- **Emission model flexibility:** the current diagonal Gaussian assumption is simple and data‑efficient but may underfit complex motion patterns. Full covariance or Gaussian mixture emissions are natural extensions.  
- **Model comparison:** discriminative sequence models (e.g. CRFs, RNNs) could be compared against the HMM baseline to quantify the benefits of generative vs discriminative approaches.

See the notebook for detailed plots, intermediate diagnostics, and alternative configurations.


