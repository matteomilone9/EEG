# EEG - ONLY
## Parametri Utilizzati

```python
{
    "split_train": True (80-20)
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
    "split_train": True (80-20)
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


EEG (Teacher & Student) 

```python
{
    "split_train": True (80-20)
    "use_gaf": False,       
    "use_kd_align": False,  
    "train_student": True   
}

================================================================================
Sub          Teacher              Student   vs EEG-Only
--------------------------------------------------------------------------------
S01 | Teacher 87.85% | Student 88.89% | ▲ 1.11pp
S02 | Teacher 57.64% | Student 60.42% | ▲ 1.81pp
S03 | Teacher 96.18% | Student 96.88% | ▲ 0.34pp
S04 | Teacher 81.25% | Student 82.29% | ▲ 2.78pp
S05 | Teacher 70.49% | Student 74.31% | ▲ 6.46pp
S06 | Teacher 60.07% | Student 59.72% | ▼ 1.81pp
S07 | Teacher 94.44% | Student 91.67% | ▼ 1.66pp
S08 | Teacher 85.07% | Student 87.15% | ▲ 1.94pp
S09 | Teacher 88.19% | Student 86.81% | ▼ 0.69pp
--------------------------------------------------------------------------------
Media Teacher         : 80.13%
Media Student KD/Align: 80.90%
```

EEG-ONLY
BCI-2a_fff_loso_kd_multival_seed42 student_acc / student_kappa non disponibili
Subject	teacher_acc	teacher_kappa	student_acc	student_kappa
1	64.8003	0.5307	—	—
2	30.4688	0.0729	—	—
3	72.3958	0.6319	—	—
4	36.9792	0.1597	—	—
5	30.2517	0.0700	—	—
6	29.2969	0.0573	—	—
7	35.8507	0.1447	—	—
8	58.0729	0.4410	—	—
9	55.4253	0.4057	—	—

EEG TEACHER-STUDENT
BCI-2a_loso_kd_multival_seed42
Subject	teacher_acc	teacher_kappa	student_acc	student_kappa
1	66.6667	0.5556	66.7101	0.5561
2	30.9462	0.0793	31.9010	0.0920
3	69.4010	0.5920	72.7865	0.6372
4	38.3247	0.1777	36.5451	0.1539
5	30.6424	0.0752	30.0781	0.0677
6	29.5139	0.0602	28.3420	0.0446
7	37.8038	0.1707	38.5417	0.1806
8	56.9878	0.4265	59.8090	0.4641
9	54.5573	0.3941	53.2552	0.3767

EEG TEACHER STUDENT  (GAF+ KD_ALIGN)
BCI-2a_ttt_loso_kd_multival_seed42
Subject	teacher_acc	teacher_kappa	student_acc	student_kappa
1	62.5868	0.5012	66.1024	0.5480
2	27.7344	0.0365	27.8646	0.0382
3	64.9306	0.5324	70.9201	0.6123
4	32.4219	0.0990	35.3299	0.1377
5	28.0816	0.0411	28.6458	0.0486
6	27.3438	0.0313	25.1302	0.0017
7	32.7257	0.1030	35.3299	0.1377
8	57.2483	0.4300	59.1580	0.4554
9	49.5660	0.3275	52.6476	0.3686
