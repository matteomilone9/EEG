# EEG - ONLY
## Parametri Utilizzati

```python
{
    "split_train": False (Session T+E)
    "use_gaf": False,        
    "use_kd_align": False,   
    "train_student": False    
}

| Subject | Accuracy (EEG)   | Student | 
| ------- | ---------------- | ------- | 
| S01     | 87.85%           | skipped | 
| S02     | 56.25%           | skipped | 
| S03     | 95.83%           | skipped | 
| S04     | 80.90%           | skipped |
| S05     | 71.88%           | skipped |
| S06     | 62.50%           | skipped | 
| S07     | 94.79%           | skipped | 
| S08     | 85.42%           | skipped |
| S09     | 89.24%           | skipped |


| Modello | Accuracy |
| ------- | -------- |
| Teacher | 80.52%   |
| Student | Skipped  |

```


# EEG + KD Align (Teacher & Student) 
## Parametri Utilizzati

```python
{
    "split_train": False (Session T+E)
    "use_gaf": True,       
    "use_kd_align": True,  
    "train_student": True   
}

| Subject | Teacher   | Student | 
| ------- | ----------| ------- | 
| S01     | 86.11%    | 88.19%  | 
| S02     | 55.56%    | 62.15%  | 
| S03     | 96.18%    | 96.53%  | 
| S04     | 75.69%    | 80.90%  |
| S05     | 66.32%    | 71.18%  |
| S06     | 56.25%    | 65.28%  | 
| S07     | 92.01%    | 95.83%  | 
| S08     | 84.04%    | 85.42%  |
| S09     | 85.07%    | 86.81%  |


| Modello | Accuracy |
| ------- | -------- |
| Teacher | 77.47%   |
| Student | 81.37%   |
```
