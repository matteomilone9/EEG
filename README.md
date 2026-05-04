Ecco tutto riscritto in modo ordinato e coerente:

---

# Risultati Sperimentali — BCI Motor Imagery

---

## 1. EEG Only (Split 80-20)

**Configurazione:**
```python
{
    "split_train": True,   # 80-20
    "use_gaf":     False,
    "use_kd_align": False,
    "train_student": False
}
```

| Subject | Teacher  | Student |
|---------|----------|---------|
| S01     | 87.85%   | skipped |
| S02     | 56.25%   | skipped |
| S03     | 95.83%   | skipped |
| S04     | 80.90%   | skipped |
| S05     | 71.88%   | skipped |
| S06     | 62.50%   | skipped |
| S07     | 94.79%   | skipped |
| S08     | 85.42%   | skipped |
| S09     | 89.24%   | skipped |
| **Media** | **80.52%** | — |

---

## 2. EEG Teacher + Student (Split 80-20)

**Configurazione:**
```python
{
    "split_train": True,   # 80-20
    "use_gaf":     False,
    "use_kd_align": False,
    "train_student": True
}
```

| Subject | Teacher  | Student  | Δ vs EEG Only |
|---------|----------|----------|---------------|
| S01     | 87.85%   | 88.89%   | ▲ +1.11pp     |
| S02     | 57.64%   | 60.42%   | ▲ +1.81pp     |
| S03     | 96.18%   | 96.88%   | ▲ +0.34pp     |
| S04     | 81.25%   | 82.29%   | ▲ +2.78pp     |
| S05     | 70.49%   | 74.31%   | ▲ +6.46pp     |
| S06     | 60.07%   | 59.72%   | ▼ −1.81pp     |
| S07     | 94.44%   | 91.67%   | ▼ −1.66pp     |
| S08     | 85.07%   | 87.15%   | ▲ +1.94pp     |
| S09     | 88.19%   | 86.81%   | ▼ −0.69pp     |
| **Media** | **80.13%** | **80.90%** | ▲ +0.38pp |

---

## 3. EEG + GAF + KD Align — Teacher + Student (Split 80-20)

**Configurazione:**
```python
{
    "split_train": True,   # 80-20
    "use_gaf":     True,
    "use_kd_align": True,
    "train_student": True
}
```

| Subject | Teacher  | Student  |
|---------|----------|----------|
| S01     | 86.11%   | 88.19%   |
| S02     | 55.56%   | 62.15%   |
| S03     | 96.18%   | 96.53%   |
| S04     | 75.69%   | 80.90%   |
| S05     | 66.32%   | 71.18%   |
| S06     | 56.25%   | 65.28%   |
| S07     | 92.01%   | 95.83%   |
| S08     | 84.04%   | 85.42%   |
| S09     | 85.07%   | 86.81%   |
| **Media** | **77.47%** | **81.37%** |

---

## 4. EEG Only — LOSO (Leave-One-Subject-Out)

**Configurazione:** `BCI-2a_fff_loso_kd_multival_seed42`

| Subject | Teacher Acc | Teacher κ | Student Acc | Student κ |
|---------|-------------|-----------|-------------|-----------|
| S01     | 64.80%      | 0.5307    | —           | —         |
| S02     | 30.47%      | 0.0729    | —           | —         |
| S03     | 72.40%      | 0.6319    | —           | —         |
| S04     | 36.98%      | 0.1597    | —           | —         |
| S05     | 30.25%      | 0.0700    | —           | —         |
| S06     | 29.30%      | 0.0573    | —           | —         |
| S07     | 35.85%      | 0.1447    | —           | —         |
| S08     | 58.07%      | 0.4410    | —           | —         |
| S09     | 55.43%      | 0.4057    | —           | —         |
| **Media** | **45.95%** | **0.2793** | —         | —         |

---

## 5. EEG Teacher + Student — LOSO

**Configurazione:** `BCI-2a_loso_kd_multival_seed42`

| Subject | Teacher Acc | Teacher κ | Student Acc | Student κ |
|---------|-------------|-----------|-------------|-----------|
| S01     | 66.67%      | 0.5556    | 66.71%      | 0.5561    |
| S02     | 30.95%      | 0.0793    | 31.90%      | 0.0920    |
| S03     | 69.40%      | 0.5920    | 72.79%      | 0.6372    |
| S04     | 38.32%      | 0.1777    | 36.55%      | 0.1539    |
| S05     | 30.64%      | 0.0752    | 30.08%      | 0.0677    |
| S06     | 29.51%      | 0.0602    | 28.34%      | 0.0446    |
| S07     | 37.80%      | 0.1707    | 38.54%      | 0.1806    |
| S08     | 56.99%      | 0.4265    | 59.81%      | 0.4641    |
| S09     | 54.56%      | 0.3941    | 53.26%      | 0.3767    |
| **Media** | **46.10%** | **0.2813** | **46.44%** | **0.2859** |

---

## 6. EEG + GAF + KD Align — Teacher + Student — LOSO

**Configurazione:** `BCI-2a_ttt_loso_kd_multival_seed42`

| Subject | Teacher Acc | Teacher κ | Student Acc | Student κ |
|---------|-------------|-----------|-------------|-----------|
| S01     | 62.59%      | 0.5012    | 66.10%      | 0.5480    |
| S02     | 27.73%      | 0.0365    | 27.86%      | 0.0382    |
| S03     | 64.93%      | 0.5324    | 70.92%      | 0.6123    |
| S04     | 32.42%      | 0.0990    | 35.33%      | 0.1377    |
| S05     | 28.08%      | 0.0411    | 28.65%      | 0.0486    |
| S06     | 27.34%      | 0.0313    | 25.13%      | 0.0017    |
| S07     | 32.73%      | 0.1030    | 35.33%      | 0.1377    |
| S08     | 57.25%      | 0.4300    | 59.16%      | 0.4554    |
| S09     | 49.57%      | 0.3275    | 52.65%      | 0.3686    |
| **Media** | **42.52%** | **0.2447** | **44.57%** | **0.2609** |
