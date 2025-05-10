# Chess Evaluation Neural Network – Training Schedule

## 🧠 Phase 0: Base Model Summary

- **Input**: `19×8×8` tensor (12 pieces + 7 auxiliary channels)
- **Model**: 10 residual blocks, Conv2D (128 filters, 3×3)
- **Output**: `Dense(1, activation='tanh')` → scalar eval
- **Loss**: `MeanSquaredError`
- **Eval Metric**: `MAE` or domain-specific (e.g. sign accuracy)

---

## 🚀 Phase 1: Initial Training

### ⚙️ Training Hyperparameters

| Parameter             | Value                            |
|----------------------|----------------------------------|
| Batch size           | 256                              |
| Learning rate        | 1e-3 (Adam)                      |
| Scheduler            | Cosine decay or ReduceLROnPlateau |
| Optimizer            | Adam                             |
| Loss                 | MSE                              |
| Epochs               | 50                               |
| Early stopping       | Patience = 5                     |
| Shuffling            | Yes (reshuffle each epoch)       |
| Caching / Prefetch   | Yes (for performance)            |
| Data split           | 95% train / 5% val               |

### 🧪 Monitoring Metrics

- Train & val loss (MSE, MAE)
- Eval distribution (histogram of predictions)
- Eval direction accuracy (sign match)

---

## 🧭 Phase 2: Evaluation After First Run

| Symptom                                 | Action                                                    |
|----------------------------------------|-----------------------------------------------------------|
| ⚠️ Overfitting                         | Add dropout (0.2–0.4), reduce model width                 |
| 📉 Underfitting                        | Add more residual blocks or filters                      |
| 🔁 Early learning plateau              | Use warmup + cosine decay, reduce LR                     |
| 🔄 Predictions saturate at ±1.0        | Check label normalization, add residual depth            |
| 🚫 MAE plateaus at ~0.3–0.4            | Use Huber loss or scaled log-cosh                        |

---

## 🛠 Phase 3: Improvements if Needed

### 🔹 Option 1: Expand Model Capacity
- Add more residual blocks (15–20)
- Increase Conv2D filters (128 → 256)
- Add Squeeze-and-Excitation layers

### 🔹 Option 2: Improve Label Strategy
- Apply soft label smoothing near ±1.0
- Cap centipawn evals more conservatively (e.g. ±10,000)

### 🔹 Option 3: Multitask Learning
- Add policy head (distribution over legal moves)
- Jointly train value + policy heads

### 🔹 Option 4: Curriculum Learning
- Train first on “simple” positions (low cp eval)
- Gradually introduce harder, high-magnitude examples

---

## 📊 Visualization & Logging

- TensorBoard:
  - Loss curves
  - Learning rate
  - Output histograms
- Save samples: prediction vs ground truth, especially near 0 and ±1