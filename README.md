# IFRS 9 PD Modeling Framework via Empirical Migration Matrix and Credit Cycle Index Approach 

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&style=for-the-badge)
![Pandas](https://img.shields.io/badge/pandas-Data%20Analysis-purple?logo=pandas&style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-green?logo=numpy&style=for-the-badge)
![SciPy](https://img.shields.io/badge/SciPy-Scientific%20Computing-blue?logo=scipy&style=for-the-badge)
![statsmodels](https://img.shields.io/badge/statsmodels-Statistical%20Modeling-red?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet?style=for-the-badge&logo=Plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-3775a9?style=for-the-badge&logo=plotly&logoColor=white)
![MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

This repository implements a **migration matrix-based Probability of Default (PD)** model aligned with IFRS 9 Expected Credit Loss (ECL) requirements. The model estimates an **Empirical Migration Matrix (EMM)** from historical loan-level data and applies a **Credit Cycle Index (CCI)** that derived from the Vasicek (1987) single factor framework to condition the Through-the-Cycle (TTC) matrix into a Point-in-Time (PiT) matrix under multiple macroeconomic scenarios. The resulting **PD term structures** are suitable for Stage 1, Stage 2, and lifetime ECL calculation.

<p align="center">
<img width="1536" height="1024" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ด้วย transition matrix แบบ credit cycle index ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/20c9f374-a844-45a5-ba3e-8e9f30ed4ed1" />
</p>

## Overview
his project implements an **EMM CCI PD Model** designed to support IFRS 9 ECL calculation. The model estimates a TTC migration matrix from observed loan grade transitions, extracts a time series of Credit Cycle Index (Z-Score, Z-Index) that summarises the systematic credit environment at each period, and links it to macroeconomic variables to generate the forward-looking PiT migration matrices and cumulative PD Term structures.

The implementation emphasises:
- Transparent, auditable migration matrix construction suitable for model governance
- Closed-form and optimisation-based CCI Estimation
- Vectorised numerical computation for efficiency and scalability
- Flexible scenario conditioning across baseline, adverse, and severe paths

The resulting PD Term structures can be directly used in Stage 1 and Stage 2 ECL Calculation. The project is intended to serve as a practical reference implementation for credit risk practitioners, model developers, and validators. All calculations are made explicit, facilitating validation, backtesting, and model explainability.

## Project Structure
```
pd_emm_cci_model/
├── model/                                        #Trainned model and parameters (pkl.)
│   ├── rho.pkl
│   ├── fwl_model.pkl
│   └── lifetime_pd_term_structure.pkl  
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


#### 1.1 Empirical Migration Matrix (TTC)
<p align="center">
<img width="1536" height="1024" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ด้วย transition matrix แบบ credit cycle index ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/a6388850-e891-42ac-a696-c6409f8a1a9e" />
</p>

**Empirical Migration Matrix:** A Through-the-Cycle (TTC) Migration matrix is estimated from historical loan level data by tracking grade transitions over a defined observation window. For each period, the number of transitions from grade $i$ to grade $j$ is counted and normalised by the total population starting in grade $i$. The long-run average across all periods forms the TTC matrix $\mathbf{P}_{\text{TTC}}$, with the final column representing the observed default rate per grade.

<p align="center">
<img width="838" height="585" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ด้วย transition matrix แบบ credit cycle index ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/b33b9451-c83f-44ff-831d-66fa38fd7ff1" />
</p>

#### 1.2 Credit Cycle Index
<p align="center">
<img width="1536" height="1024" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ด้วย transition matrix แบบ credit cycle index ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/465f0b6f-95d8-4bfe-8c71-9b54643acff7" />
</p>

**CCI Model:** The Credit Cycle Index is estimated by fitting a time-varying Z-Score to each period's observed migration matrix. Based on the Vasicek (1987) single-factor model:

$$ P_t(ij) = \Phi\left(\frac{x_{i+1}^{\,j} - \sqrt{\rho}\, z_t}{\sqrt{1 - \rho}}\right) - \Phi\left(\frac{x_i^{\,j} - \sqrt{\rho}\, z_t}{\sqrt{1 - \rho}}\right) $$

$$
\min_{z_t} \sum_j \sum_i n_{t,G} \frac{\left[P_t(i,j) - \Delta\left(x_{i+1}^j, x_i^j, z_t\right)\right]^2}{\Delta\left(x_{i+1}^j, x_i^j, z_t\right)\left[1 - \Delta\left(x_{i+1}^j, x_i^j, z_t\right)\right]}
$$

The transition probabilities are expressed as threshold crossings of a standard normal distribution (Belkin, Suchower & Forest, 1998). For each period $t$, the CCI ($Z_t$) and asset correlation ($\rho$) are jointly estimated by minimising the weighted squared error between observed and model implied migration probabilities, subject to the constraint $\text{std}(\{Z_t\}) = 1$. A simplified closed-form variant is also provided for cases where only $\rho$ optimisation is required.

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
**Conditional PIT Matrix:** For each forecast horizon $h$ and scenario, the TTC matrix is shifted using the forecast $\hat{Z}_{t+h}$ to produce a period-specific PiT migration matrix by the formula below:

$$ P_t(i,j) = \Phi\left(\frac{x_{i+1}^j - \sqrt{\rho}\, \hat{z_t}}{\sqrt{1 - \rho}}\right) - \Phi\left(\frac{x_i^j - \sqrt{\rho}\, \hat{z_t}}{\sqrt{1 - \rho}}\right) $$

<p align="center">
<img width="1536" height="1024" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ด้วย transition matrix แบบ credit cycle index ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/9ecee832-a2f1-4597-8b24-a4f3ba1a1fc9" />
</p>

The cumulative multi-periods of PD Term structures are derived by chaining:

$$\mathbf{P}(h) = \prod_{t=1}^{h} \mathbf{P}_t^{\text{PiT}}$$

The last column of $\mathbf{P}(h)$ gives the cumulative PD per grade at horizon $h$, directly applicable to IFRS 9 Stage 1 (12-month) and Stage 2 (lifetime) ECL Calculation.

<p align="center">
<img width="989" height="592" alt="การพัฒนาแบบจำลอง IFRS 9 PD Model ด้วย transition matrix แบบ credit cycle index ตั้งแต่ต้นจนจบ" src="https://github.com/user-attachments/assets/0e73d859-3076-4e15-87a3-126ee729ba1e" />
</p>

## License
MIT · Built for learning purposes
