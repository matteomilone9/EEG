## 📊 Experimental Setup *(KD-ALIGN – 24/04)*

* **Modality:** EEG-only + Knowledge Distillation Alignment
* **Mixup:** ON
* **Evaluation:** Within-Subject
* **Cross-Attention:** OFF
* **Data Split:** No Split (Evaluation = Validation + Test)
* **Seeds:** 5 (multi-seed averaging)
* **Teacher Guidance:** Enabled
* **Experiment Date:** 24/04

---

## 📈 Results per Subject (Baseline EEG-Only)

| Subject | Mean Accuracy (%) |  Std Dev | Cohen’s Kappa |
| ------- | ----------------: | -------: | ------------: |
| S01     |             87.08 |     1.38 |        0.8278 |
| S02 *   |             56.81 |     1.71 |        0.4241 |
| S03     |             96.46 |     0.51 |        0.9528 |
| S04     |             80.76 |     1.52 |        0.7435 |
| S05     |             67.85 |     1.21 |        0.5713 |
| S06 *   |             60.97 |     1.74 |        0.4796 |
| S07     |             94.10 |     0.38 |        0.9213 |
| S08     |             85.14 |     1.75 |        0.8019 |
| S09     |             88.06 |     0.56 |        0.8407 |
| **Avg** |         **79.69** | **1.20** |    **0.7292** |

> *Difficult subjects (CFG override enabled)*

---

## 🔬 Distillation Impact (Previous KD)

| Sub     |  Baseline | Distill=ON |         Δ |
| ------- | --------: | ---------: | --------: |
| S01     |     87.08 |      87.78 |     +0.70 |
| S02     |     56.81 |      58.61 |     +1.80 |
| S03     |     96.46 |      96.53 |     +0.07 |
| S04     |     80.76 |      79.51 |     -1.25 |
| S05     |     67.85 |      67.85 |      0.00 |
| S06     |     60.97 |      61.53 |     +0.56 |
| S07     |     94.10 |      93.33 |     -0.77 |
| S08     |     85.14 |      85.21 |     +0.07 |
| S09     |     88.06 |      87.50 |     -0.56 |
| **Avg** | **79.69** |  **79.76** | **+0.07** |

---

## 🚀 KD-ALIGN Results *(24/04 Final Run)*

| Sub | Teacher (%) | Student (%) | vs EEG-Only |
| --- | ----------: | ----------: | ----------: |
| S01 |       73.26 |       86.81 |      ▼ 0.27 |
| S02 |       42.71 |       58.68 |      ▲ 1.87 |
| S03 |       78.47 |       96.53 |      ▲ 0.07 |
| S04 |       42.01 |       65.62 |     ▼ 15.14 |
| S05 |       47.22 |       72.57 |      ▲ 4.72 |
| S06 |       36.81 |       57.64 |      ▼ 3.33 |
| S07 |       77.78 |       94.79 |      ▲ 0.69 |
| S08 |       72.22 |       82.99 |      ▼ 2.15 |
| S09 |       69.44 |       86.11 |      ▼ 1.95 |

| Metric                      | Accuracy (%) |
| --------------------------- | -----------: |
| **Avg Teacher**             |    **59.99** |
| **Avg Student KD-ALIGN**    |    **77.97** |
| **Best EEG-Only Reference** |    **79.76** |
| **Gap vs EEG-Only**         | **-1.79 pp** |

---

## 🧠 Analysis

### KD-ALIGN vs Standard Distillation

* Standard KD achieved **79.76%**
* KD-ALIGN achieved **77.97%**
* Performance drop: **-1.79 pp**

### Positive Cases

KD-ALIGN improved difficult or mid-level subjects:

* **S02:** +1.87
* **S05:** +4.72
* **S07:** +0.69

### Negative Cases

Large degradation on:

* **S04:** -15.14
* **S06:** -3.33
* **S08:** -2.15
* **S09:** -1.95

### Interpretation

KD-ALIGN appears to help when teacher signals remain informative on weaker subjects, but harms when teacher representations are misaligned or underperforming.

Teacher average accuracy (**59.99%**) is significantly below student standalone performance, suggesting imperfect teacher supervision quality.

---

## 📌 Final Conclusion

* **Best overall method remains Standard Distillation (79.76%)**
* **KD-ALIGN underperforms baseline by -1.79 pp**
* However, KD-ALIGN shows promise for **hard subjects** and could improve with:

  * stronger teacher models
  * adaptive alignment weights
  * subject-wise gating
  * confidence-based teacher filtering

---

## 📁 README Summary
EEG Within-Subject Classification Benchmark

Baseline EEG-Only         : 79.69%
Standard Distillation    : 79.76%
KD-ALIGN (24/04)         : 77.97%

Best Method: Standard KD

KD-ALIGN helps difficult subjects but reduces global average.
Teacher quality likely bottleneck.
