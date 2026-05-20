# IFRS 9 PD Modeling Framework via Empirical Migration Matrix and Credit Cycle Index Approach 

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&style=for-the-badge)
![Pandas](https://img.shields.io/badge/pandas-Data%20Analysis-purple?logo=pandas&style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-green?logo=numpy&style=for-the-badge)
![SciPy](https://img.shields.io/badge/SciPy-Scientific%20Computing-blue?logo=scipy&style=for-the-badge)
![statsmodels](https://img.shields.io/badge/statsmodels-Statistical%20Modeling-red?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-teal?style=for-the-badge)
![MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)



<p align="center">
<img width="1536" height="1024" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ด้วย transition matrix แบบ credit cycle index ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/20c9f374-a844-45a5-ba3e-8e9f30ed4ed1" />
</p>

## Overview

## Project Structure
```
pd_emm_cci_model/
├── model/                                        #Trainned model and parameters (pkl.)
│   ├── rho.pkl
│   └── fwl_model.pkl  
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_credit_cycle_index.ipynb
│   ├── 03_fwl_model.ipynb
│   └── 04_markov_liftetime.ipynb
├── src/
│   ├── data_prep.py
│   ├── migration_matrix_cci.py
│   ├── regression_model.py
│   ├── lifetime_model.py
│   ├── stats_testing.py
│   └── plot_function.py
├── data/          
│   ├── processed/
|   |   ├── train_data.parquet                    #Not tracked by git
|   |   ├── migration_count.parquet
|   |   ├── average_matrix.parquet
|   |   ├── monthly_cci.parquet
|   |   ├── mev_transformed.parquet
|   |   └── mev_sign_transformed.parquet
│   └── raw/
|   |   ├── usedcar_transaction_score.parquet     #Not tracked by git
|   └── └── mev_data.csv
├── requirements.txt
└── README.md
```

## Project Details
### 0. Model Segmentation
### 1. Unbias Model
#### 1.1 Empirical Migration Matrix
<p align="center">
<img width="1536" height="1024" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ด้วย transition matrix แบบ credit cycle index ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/a6388850-e891-42ac-a696-c6409f8a1a9e" />
</p>

#### 1.2 Credit Cycle Index
<p align="center">
<img width="1536" height="1024" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ด้วย transition matrix แบบ credit cycle index ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/465f0b6f-95d8-4bfe-8c71-9b54643acff7" />
</p>

$$ P_t(ij) = \Phi\left(\frac{x_{i+1}^{\,j} - \sqrt{\rho}\, z_t}{\sqrt{1 - \rho}}\right) - \Phi\left(\frac{x_i^{\,j} - \sqrt{\rho}\, z_t}{\sqrt{1 - \rho}}\right) $$

$$
\min_{z_t} \sum_j \sum_i n_{t,G} \frac{\left[P_t(i,j) - \Delta\left(x_{i+1}^j, x_i^j, z_t\right)\right]^2}{\Delta\left(x_{i+1}^j, x_i^j, z_t\right)\left[1 - \Delta\left(x_{i+1}^j, x_i^j, z_t\right)\right]}
$$


<p align="center">
<img width="989" height="593" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ด้วย transition matrix แบบ credit cycle index ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/c8359005-8fb0-4238-bdcd-cdbf3141a19c" />
</p>

### 2. Forward-looking Model
The forward-looking model processes are finding the relationship between Credit Cycle Index (CCI, Z-Index) with macroeconomics varialbes (MEV). The processes are similar to others ODR Model but changed the dependence variabale from ODR to CCI. In this repository is not covered the forward-looking model but it can refer to [this repository](https://github.com/naenumtou/pd_cohort_model/blob/main/README.md#2-forward-looking-model) for the forward-looking model consideration.

#### 2.1 Model Back-testing
The model back-testing of actual CCI and predicted CCI from the regression model have been displayed in the following section. The visualisation of model back-testing in the following:

<p align="center">
<img width="989" height="593" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ด้วย transition matrix แบบ credit cycle index ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/e64e6d97-cafc-4de5-b31e-9d8a16c284c7" />
</p>

> Note: It is a ramdom selection model. No expert input in this model.

### 3. Lifetime Model
<p align="center">
<img width="1536" height="1024" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ด้วย transition matrix แบบ credit cycle index ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/9ecee832-a2f1-4597-8b24-a4f3ba1a1fc9" />
</p>


$$ P_t(i,j) = \Phi\left(\frac{x_{i+1}^j - \sqrt{\rho}\, \hat{z}(t)}{\sqrt{1 - \rho}}\right) - \Phi\left(\frac{x_i^j - \sqrt{\rho}\, \hat{z}(t)}{\sqrt{1 - \rho}}\right) $$
