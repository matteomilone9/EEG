## 📊 Experimental Setup

- **Modality:** EEG-only  
- **Mixup:** ON  
- **Evaluation:** Within-Subject  
- **Cross-Attention:** OFF  
- **Data Split:** No Split (Evaluation = Validation + Test)  
- **Seeds:** 5 (multi-seed averaging)

---

## 📈 Results per Subject (Baseline)

| Subject | Mean Accuracy (%) | Std Dev | Cohen’s Kappa |
|---------|------------------|---------|---------------|
| S01     | 87.08            | 1.38    | 0.8278        |
| S02 *   | 56.81            | 1.71    | 0.4241        |
| S03     | 96.46            | 0.51    | 0.9528        |
| S04     | 80.76            | 1.52    | 0.7435        |
| S05     | 67.85            | 1.21    | 0.5713        |
| S06 *   | 60.97            | 1.74    | 0.4796        |
| S07     | 94.10            | 0.38    | 0.9213        |
| S08     | 85.14            | 1.75    | 0.8019        |
| S09     | 88.06            | 0.56    | 0.8407        |
|---------|------------------|---------|---------------|
| **Avg** | **79.69**        | **1.20**| **0.7292**    |

> \* Difficult subjects (CFG override enabled)

---

## 🔬 Distillation Impact

| Sub | Baseline | Distill=ON | Δ     |
|-----|----------|------------|-------|
| S01 | 87.08    | 87.78      | +0.70 |
| S02 | 56.81    | 58.61      | +1.80 |
| S03 | 96.46    | 96.53      | +0.07 |
| S04 | 80.76    | 79.51      | −1.25 |
| S05 | 67.85    | 67.85      | 0.00  |
| S06 | 60.97    | 61.53      | +0.56 |
| S07 | 94.10    | 93.33      | −0.77 |
| S08 | 85.14    | 85.21      | +0.07 |
| S09 | 88.06    | 87.50      | −0.56 |
|-----|----------|------------|-------|
| **Avg** | **79.69** | **79.76** | **+0.07** |

---

## 🧠 Notes

- Distillation provides **marginal overall improvement (+0.07%)**
- Gains are more evident on **difficult subjects (e.g., S02, S06)**
- Slight degradation observed on some high-performing subjects (e.g., S04, S07)
