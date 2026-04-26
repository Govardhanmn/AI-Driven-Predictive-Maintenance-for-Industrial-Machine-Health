# ⚙️ AI-Driven Predictive Maintenance for Industrial Machine Health

> A production-ready, end-to-end machine learning system that predicts industrial machine failure from real-time sensor telemetry — before the breakdown happens.

---

## 📌 Project Overview

Unplanned machine downtime costs manufacturers an estimated **$50 billion annually**. This project builds a full predictive maintenance pipeline that ingests live sensor data — temperature, rotational speed, torque, and tool wear — and predicts failure probability using a tuned **XGBoost classifier** deployed via a **Streamlit dashboard**.

The system enables maintenance teams to shift from reactive ("fix it after it breaks") to proactive ("fix it before it breaks") operations, dramatically reducing downtime and operational costs.

---

## 🖥️ Live Dashboard Preview

| Status | Green (Healthy) | Yellow (Warning) | Red (Failure) |
|---|---|---|---|
| **Prediction** | No Breakdown | Monitor Required | Failure Imminent |
| **Action** | Continue operation | Schedule inspection | Stop machine immediately |
| **Condition** | `prob_fail < 20%` | `20% ≤ prob_fail < 50%` | `prob_fail ≥ 50%` |

> The **Health Score** displayed on the dashboard is derived as `(1 - failure_probability) × 100`. A machine with 13% failure probability shows a Health Score of 87%.

---

## 🚀 Features

- **Real-time inference** — Adjusts failure probability live as sensor values change
- **Three-tier alert system** — Green / Yellow / Red status with dynamic dashboard theming
- **Mechanical factor analysis** — Individual breakdowns for Torque, Speed, Tool Wear, and Temperature
- **Premium glassmorphism UI** — Dark-themed dashboard with industrial background imagery
- **Hyperparameter-tuned model** — XGBoost optimized via `RandomizedSearchCV` (50 iterations × 5-fold CV)

---

## 🧠 Machine Learning Pipeline

```
Raw Data → EDA → Preprocessing → Benchmarking → Hyperparameter Tuning → Deployment
```

### Models Benchmarked
| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Interpretable, depth-limited |
| Random Forest | Ensemble with balanced class weights |
| SVM | Kernel-based, probability calibrated |
| **XGBoost ✅** | **Champion model — tuned & deployed** |

### Why XGBoost?
Tree-based ensemble methods dominate physical sensor data classification. They carve non-linear decision boundaries across multi-dimensional feature spaces, capturing the complex interaction between temperature drift, torque spikes, and tool wear that linear models miss.

### Handling Class Imbalance
The dataset has a **96% Healthy / 4% Failure** split. To prevent the model from ignoring the minority failure class:
- `scale_pos_weight` set to the class ratio
- Primary metric: **F1-Score & Recall** (not Accuracy)
- Goal: Maximize **True Positive Rate** — catching every real failure

---

## 🔧 Feature Engineering & Preprocessing

| Feature | Transformation | Reason |
|---|---|---|
| `Rotational speed [rpm]` | `np.log1p()` | Normalizes right-skewed distribution |
| All features | `RobustScaler` | Scales using median/IQR — preserves failure-signature outliers |
| `Type` (L/M/H) | One-hot encoding | Converts categorical machine quality to numeric |


---

## 📈 Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted failures, how many were real? (avoids false alarms) |
| **Recall** | Of actual failures, how many did we catch? (critical for safety) |
| **F1-Score** | Harmonic mean of Precision and Recall — our primary metric |

> ⚠️ **Why Recall matters most:** A False Negative (missed failure) results in catastrophic unplanned downtime. We always prefer a cautious false alarm over a missed breakdown.

---

## 🎯 How Predictions Are Made

The dashboard uses `predict_proba()` from the tuned XGBoost model to return a **continuous failure probability** between 0.0 and 1.0. This is then mapped to three actionable states:

```python
prob_fail = model.predict_proba(input_scaled)[0][1]  # Probability of class 1 (Failure)

if prob_fail < 0.20:        # < 20% failure chance
    status = "NO BREAKDOWN"       # 🟢 Green
elif prob_fail < 0.50:      # 20–50% failure chance
    status = "WARNING STATE"      # 🟡 Yellow
else:                        # ≥ 50% failure chance
    status = "FAILURE IMMINENT"   # 🔴 Red

health_score = int((1.0 - prob_fail) * 100)  # e.g., prob=0.13 → score=87%
```

The thresholds (`0.20` and `0.50`) are **engineering decisions**, not model outputs. Lowering the yellow threshold (e.g., to `0.10`) makes the alert system more conservative — triggering warnings earlier at the cost of more false alarms. This trade-off is configurable based on the facility's risk tolerance.
