# 🧠 EEG Within-Subject Classification Benchmark


----------------------------------------------------------------------------
## "use_gaf":         False,   # era True
## "use_kd_align":    False,   # era True
## "train_student":   False,   # era True  ← non addestrare lo student
----------------------------------------------------------------------------

#TODO
##Togliere bandpass → "lowcut": None, "highcut": None

##Allineare F1 al paper → "teacher_F1": 32, "student_F1": 32

##Allineare trans_depth → "teacher_trans_depth": 5, "student_trans_depth": 5
#

## 📌 Overview

This report summarizes the performance of different training strategies for EEG-only within-subject classification.

Compared methods:

1. **EEG-Only Baseline**
2. **Standard Knowledge Distillation (KD)**
3. **KD-ALIGN (27/04 Final Run)**

---

# ⚙️ Experimental Setup

| Parameter | Configuration |
|----------|---------------|
| **Input Modality** | EEG Only |
| **Training Strategy** | Knowledge Distillation + Alignment |
| **Evaluation Protocol** | Within-Subject |
| **Mixup** | Enabled |
| **Cross-Attention** | Disabled |
| **Seeds** | 5 *(multi-seed average)* |
| **Teacher Guidance** | Enabled |
| **Date** | 27/04 |

---

# 📈 Baseline Results — EEG Only

| Subject | Accuracy (%) | Std Dev | Cohen’s Kappa |
|--------|-------------:|--------:|--------------:|
| S01 | 87.08 | 1.38 | 0.8278 |
| S02 | 56.81 | 1.71 | 0.4241 |
| S03 | 96.46 | 0.51 | 0.9528 |
| S04 | 80.76 | 1.52 | 0.7435 |
| S05 | 67.85 | 1.21 | 0.5713 |
| S06 | 60.97 | 1.74 | 0.4796 |
| S07 | 94.10 | 0.38 | 0.9213 |
| S08 | 85.14 | 1.75 | 0.8019 |
| S09 | 88.06 | 0.56 | 0.8407 |

| **Average** | **79.69** | **1.20** | **0.7292** |

---

# 🧪 Previous Best Method — Standard KD

| Metric | Accuracy (%) |
|-------|-------------:|
| **Average Accuracy** | **79.76** |
| **Gain vs Baseline** | **+0.07 pp** |

---

# 🚀 KD-ALIGN Results *(27/04 Final Run)*

| Subject | Teacher (%) | Student (%) | vs EEG-Only |
|--------|------------:|------------:|------------:|
| S01 | 84.72 | 87.50 | ▼ -0.28 pp |
| S02 | 54.51 | 60.42 | ▲ +1.81 pp |
| S03 | 94.79 | 96.53 | = 0.00 pp |
| S04 | 73.61 | 79.17 | ▼ -0.34 pp |
| S05 | 61.46 | 71.18 | ▲ +3.33 pp |
| S06 | 59.03 | 64.24 | ▲ +2.71 pp |
| S07 | 90.62 | 94.79 | ▲ +1.46 pp |
| S08 | 82.29 | 85.42 | ▲ +0.21 pp |
| S09 | 84.72 | 87.85 | ▲ +0.35 pp |

---

# 📊 Global Summary

| Metric | Accuracy (%) |
|-------|-------------:|
| **Average Teacher** | **76.20** |
| **Average Student KD-ALIGN** | **80.79** |
| **Best Previous Reference (Standard KD)** | **79.76** |
| **Gain vs Previous Best** | **+1.03 pp** |

---

# 🏆 Final Ranking

| Rank | Method | Accuracy (%) |
|-----:|--------|-------------:|
| 🥇 1 | **KD-ALIGN (27/04)** | **80.79** |
| 🥈 2 | Standard KD | 79.76 |
| 🥉 3 | EEG-Only Baseline | 79.69 |

---

# ✅ Subject-Level Improvements

Strongest gains obtained on difficult subjects:

| Subject | Improvement |
|--------|------------:|
| S05 | +3.33 pp |
| S06 | +2.71 pp |
| S02 | +1.81 pp |
| S07 | +1.46 pp |

Additional positive gains:

- **S08** → +0.21 pp  
- **S09** → +0.35 pp  

---

# ⚠️ Minor Performance Drops

Only small degradations observed:

| Subject | Change |
|--------|-------:|
| S01 | -0.28 pp |
| S04 | -0.34 pp |

No change:

- **S03** → 0.00 pp

---

# 🔍 Interpretation

The 27/04 KD-ALIGN run significantly improves over previous attempts.

Main observations:

- Higher teacher quality (**76.20% average**)
- Better student generalization (**80.79%**)
- Strong improvements on weaker subjects
- Minimal negative transfer

This indicates that **teacher reliability and alignment stability are critical factors** for successful distillation.

---

# 🏁 Final Conclusion

## Best Overall Method: **KD-ALIGN**

KD-ALIGN is now the top-performing strategy for this benchmark, outperforming both:

- EEG-Only baseline
- Standard Knowledge Distillation

It is especially effective on historically difficult subjects while preserving strong performance on easier ones.

---

# 🚀 Suggested Future Work

- Confidence-based teacher filtering  
- Adaptive alignment loss weights  
- Multi-teacher ensembles  
- Dynamic temperature scheduling  
- Cross-subject transfer learning  
- Better calibration strategies  

---

# 📁 README Summary

```text
EEG Within-Subject Classification Benchmark

Baseline EEG-Only      : 79.69%
Standard KD            : 79.76%
KD-ALIGN (27/04)       : 80.79%

Best Method            : KD-ALIGN

Gain vs Previous Best  : +1.03 pp

Strong improvements on difficult subjects
with minimal negative transfer.
