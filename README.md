# 📊 Experimental Setup — KD-ALIGN *(24/04 Final Run)*

## ⚙️ Configuration

| Parameter | Setting |
|----------|---------|
| **Modality** | EEG-only + Knowledge Distillation Alignment |
| **Mixup** | Enabled |
| **Evaluation Protocol** | Within-Subject |
| **Cross-Attention** | Disabled |
| **Data Split** | No Split *(Validation = Test)* |
| **Seeds** | 5 *(multi-seed averaging)* |
| **Teacher Guidance** | Enabled |
| **Experiment Date** | 24/04 |

---

# 📈 Baseline Results — EEG Only

| Subject | Accuracy (%) | Std Dev | Cohen’s Kappa |
|--------|-------------:|--------:|--------------:|
| S01 | 87.08 | 1.38 | 0.8278 |
| S02* | 56.81 | 1.71 | 0.4241 |
| S03 | 96.46 | 0.51 | 0.9528 |
| S04 | 80.76 | 1.52 | 0.7435 |
| S05 | 67.85 | 1.21 | 0.5713 |
| S06* | 60.97 | 1.74 | 0.4796 |
| S07 | 94.10 | 0.38 | 0.9213 |
| S08 | 85.14 | 1.75 | 0.8019 |
| S09 | 88.06 | 0.56 | 0.8407 |
| **Average** | **79.69** | **1.20** | **0.7292** |

> *\* Difficult subjects (CFG override enabled)*

---

# 🧪 Standard Knowledge Distillation Results

| Subject | Baseline | Distillation ON | Δ Improvement |
|--------|---------:|----------------:|-------------:|
| S01 | 87.08 | 87.78 | +0.70 |
| S02 | 56.81 | 58.61 | +1.80 |
| S03 | 96.46 | 96.53 | +0.07 |
| S04 | 80.76 | 79.51 | -1.25 |
| S05 | 67.85 | 67.85 | 0.00 |
| S06 | 60.97 | 61.53 | +0.56 |
| S07 | 94.10 | 93.33 | -0.77 |
| S08 | 85.14 | 85.21 | +0.07 |
| S09 | 88.06 | 87.50 | -0.56 |
| **Average** | **79.69** | **79.76** | **+0.07** |

---

# 🚀 KD-ALIGN Results *(24/04 Final)*

| Subject | Teacher (%) | Student (%) | vs EEG-Only |
|--------|------------:|------------:|------------:|
| S01 | 73.26 | 86.81 | -0.27 |
| S02 | 42.71 | 58.68 | +1.87 |
| S03 | 78.47 | 96.53 | +0.07 |
| S04 | 42.01 | 65.62 | -15.14 |
| S05 | 47.22 | 72.57 | +4.72 |
| S06 | 36.81 | 57.64 | -3.33 |
| S07 | 77.78 | 94.79 | +0.69 |
| S08 | 72.22 | 82.99 | -2.15 |
| S09 | 69.44 | 86.11 | -1.95 |

---

# 📌 Global Metrics

| Metric | Accuracy (%) |
|-------|-------------:|
| **Average Teacher** | **59.99** |
| **Average Student KD-ALIGN** | **77.97** |
| **Best EEG-Only Reference** | **79.76** |
| **Gap vs EEG-Only** | **-1.79 pp** |

---

# 🧠 Analysis

## KD-ALIGN vs Standard Distillation

| Method | Accuracy (%) |
|-------|-------------:|
| Baseline EEG-Only | 79.69 |
| Standard KD | **79.76** |
| KD-ALIGN | 77.97 |

**KD-ALIGN underperforms Standard KD by -1.79 percentage points.**

---

## ✅ Positive Subject-Level Gains

KD-ALIGN improved several difficult or mid-performing subjects:

- **S02** → +1.87 pp  
- **S05** → +4.72 pp  
- **S07** → +0.69 pp  

---

## ❌ Major Performance Drops

Significant degradations observed on:

- **S04** → -15.14 pp  
- **S06** → -3.33 pp  
- **S08** → -2.15 pp  
- **S09** → -1.95 pp  

---

## 🔍 Interpretation

KD-ALIGN appears beneficial when:

- the teacher provides informative guidance on weak subjects  
- class boundaries are harder to learn directly from EEG  
- student benefits from representation regularization  

However, it becomes harmful when:

- teacher accuracy is low  
- latent spaces are poorly aligned  
- noisy supervision dominates learning  

The average teacher performance (**59.99%**) is substantially below student standalone accuracy, indicating that **teacher quality is the main bottleneck**.

---

# 🏆 Final Conclusion

## Best Overall Method: **Standard Knowledge Distillation**

| Method | Accuracy (%) |
|-------|-------------:|
| EEG-Only Baseline | 79.69 |
| Standard KD | **79.76** |
| KD-ALIGN | 77.97 |

KD-ALIGN shows **subject-specific potential**, especially for difficult users, but reduces overall performance.

---

# 🚀 Future Improvements for KD-ALIGN

Possible directions:

- Stronger teacher architectures  
- Confidence-aware teacher filtering  
- Adaptive alignment weighting  
- Subject-wise gating mechanisms  
- Dynamic loss scheduling  
- Better feature-space normalization  

---

# 📁 README Summary

```text
EEG Within-Subject Classification Benchmark

Baseline EEG-Only      : 79.69%
Standard Distillation  : 79.76%
KD-ALIGN (24/04)       : 77.97%

Best Method            : Standard KD

KD-ALIGN helps difficult subjects
but lowers the global average.

Teacher quality is likely the main bottleneck.
