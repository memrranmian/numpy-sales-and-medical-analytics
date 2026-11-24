# 📊 Enterprise-Grade NumPy Analytics: Retail & Clinical Systems
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.x-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Domain](https://img.shields.io/badge/Domain-Retail%20%26%20Healthcare%20Analytics-brightgreen.svg)]()
[![Code Quality](https://img.shields.io/badge/Vectorized-100%25%20No--Loop%20Math-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-orange.svg)](LICENSE)

An end-to-end numerical computing, data analytics, and vectorization project engineered in pure **NumPy**. This repository demonstrates production-grade numerical processing across two complex real-world domains: **Omnichannel E-Commerce Inventory/Revenue Systems** and **Hospital Patient Clinical Monitoring**.

Designed for **Data Analysts, Data Scientists, and Scientific Computing Researchers** to showcase mastery over memory layouts, broadcasting rules, multi-condition boolean filtering, matrix transformations, and persistent I/O.

---

## 🌟 Executive Summary & Impact

| Scenario | Real-World Problem | Key Mathematical & NumPy Techniques | Business / Clinical Value |
| :--- | :--- | :--- | :--- |
| **System 1: E-Commerce Analytics** | Real-time multi-branch revenue tracking, stock deduction, and inventory replenishment alerts. | Tensor broadcasting `(5, 4) * (5, 1)`, Axis-reductions (`axis=0`, `axis=1`), `np.where()`, Dynamic manipulation (`append`, `insert`, `delete`), Concatenation & Horizontal/Vertical Stacking. | Automated stock alerts for SKU depletion (< 20 units), branch revenue benchmarking (> Rs. 100K), and multi-day trend merging. |
| **System 2: Clinical ICU Monitoring** | Multiparametric vital signs surveillance (Temp, Systolic BP, Heart Rate, SpO2) across hospital wards. | 3D Tensor Reshaping `(2, 4, 4)`, Axis Swapping (`np.swapaxes`), Universal Functions (`ufunc`), Bitwise Multi-Condition Boolean Masks, Calibration Broadcasting. | Automated triage classification: Instant identification of high-risk patients meeting multiple critical thresholds (Tachycardia + Hypoxemia + Pyrexia). |

---

## 🛠️ Deployed Skills & Technical Competencies

### 1. Vectorized Computation & Array Broadcasting
- **N-Dimensional Matrix Alignment:** Applied broadcasting mechanics to seamlessly compute element-wise products across rank-differing arrays `(5, 4)` and `(5, 1)` without allocating redundant loop overhead.
- **Micro-Calibration via Broadcasted Scalars:** Applied offset corrections across high-throughput sensor telemetry in zero-copy execution pipelines.

### 2. Multi-Condition Boolean Indexing & Search
- **Bitwise Logic Gates (`&`, `|`, `~`):** Formulated complex composite filtering pipelines across multiple continuous clinical vitals (e.g. Body Temp > 38.0°C, Heart Rate > 100 bpm, SpO2 < 95%).
- **Axis-Summing Boolean Matrix Evaluation:** Leveraged binary truth arithmetic (`sum(axis=1) > 1`) to detect simultaneous multi-morbidity alerts and prioritize emergency interventions.
- **Index Retrieval (`np.where`, `np.argmax`, `np.argmin`):** Pinpointed exact entity coordinates for highest revenue branches, maximum tachycardia outliers, and depleted stock items.

### 3. Tensor Reshaping & Multidimensional Geometry
- **3D Ward Tensor Partitioning:** Reshaped continuous patient streams into multidimensional tensor structures `(2 Wards, 4 Beds, 4 Clinical Signals)`.
- **Axis Swapping & Transposition:** Utilized `np.transpose(2, 0, 1)` and `np.swapaxes(0, 1)` to reorient coordinates for longitudinal patient vs measurement comparisons.
- **Flattening & Splitting:** Restructured data between serialized storage formats (`flatten()`) and compartmentalized cohorts (`np.split()`).

### 4. Memory Management & Defense Against Data Corruption
- **Pointer Aliasing vs Views vs Deep Copies:** Explicitly validated mutation isolation:
  - **Alias (`b = a`):** Reference pointer sharing.
  - **Shallow View (`b = a.view()`):** Shared internal buffer; changes propagate.
  - **Deep Copy (`b = a.copy()`):** Independent data block for non-destructive exploratory analysis and simulations.

### 5. Robust Scientific File I/O
- High-precision export/import pipelines using `np.savetxt()` and `np.loadtxt(skiprows=1)` with delimiter standards, validated through floating-point epsilon checks via `np.allclose(atol=1e-2)`.

---

## 📂 Repository Architecture

```text
numpy-sales-and-medical-analytics/
├── .gitignore                     # Production ignore rules
├── Assignment_06_IDS_Numpy.pdf    # Complete project specification
├── README.md                      # Professional technical documentation (this file)
├── question_1/                    # Module 1: E-Commerce Sales & Inventory Analytics
│   ├── question_1.py              # Fully documented NumPy execution script
│   └── question_1_output.txt      # Verified runtime output log
└── question_2/                    # Module 2: Hospital Patient Monitoring System
    ├── question_2.py              # Fully documented NumPy execution script
    ├── question_2_output.txt      # Verified runtime output log
    └── patient_data.csv           # Structured exported clinical data
```

---

## 🚀 Execution & Quick Start

Ensure you have Python 3.9+ and NumPy installed:

```bash
# Clone the repository
git clone https://github.com/memrranmian/numpy-sales-and-medical-analytics.git
cd numpy-sales-and-medical-analytics

# Verify NumPy installation
python -c "import numpy as np; print(f'NumPy Version: {np.__version__}')"
```

### Running Module 1: E-Commerce Analytics
```bash
python question_1/question_1.py
```

### Running Module 2: Hospital Patient Monitoring
```bash
python question_2/question_2.py
```

---

## 📈 Sample Results & System Output

### Module 1: E-Commerce Executive Summary
```text
======================= EXECUTIVE SALES REPORT =======================
1. Total Company Revenue: Rs. 1,040,600.00
----------------------------------------------------------------------
2. Product-wise Sales Quantities & Total Revenues:
   - Product 0: Total Quantity Sold =  70 units | Revenue = Rs.  84,000.00
   - Product 1: Total Quantity Sold =  82 units | Revenue = Rs. 287,000.00
   - Product 2: Total Quantity Sold =  40 units | Revenue = Rs. 300,000.00
   - Product 3: Total Quantity Sold =  90 units | Revenue = Rs.  72,000.00
   - Product 4: Total Quantity Sold =  48 units | Revenue = Rs. 297,600.00
----------------------------------------------------------------------
3. Revenue Highlights:
   - Highest Revenue Product: Product 2 (Rs. 300,000.00)
   - Lowest Revenue Product:  Product 3 (Rs.  72,000.00)
   - Top Performing Branch:   Branch 3 (Rs. 308,200.00)
======================================================================
```

### Module 2: ICU Triage Classification
```text
================== HOSPITAL PATIENT MONITORING REPORT ================
1. SUMMARY OF OVERALL WARD VITALS:
   - Average Temperature:       37.62 °C (Std: 0.94)
   - Average Systolic BP:       125.62 mmHg (Std: 8.14)
   - Average Heart Rate:        90.88 bpm (Std: 15.07)
   - Average Oxygen Saturation: 95.88 % (Std: 2.42)
----------------------------------------------------------------------
2. PATIENT TRIAGE ALERTS:
   - Patient 0 [WARNING]:  HIGH TEMP (38.5°C > 38.0°C)
   - Patient 1 [CRITICAL]: HIGH TEMP (39.1°C) | HIGH HR (115 bpm) | LOW SpO2 (92.0%)
   - Patient 2 [STABLE]:   All vitals in normal range
   - Patient 3 [WARNING]:  HIGH HR (108 bpm > 100 bpm)
   - Patient 4 [WARNING]:  LOW SpO2 (93.0% < 95.0%)
   - Patient 6 [CRITICAL]: HIGH TEMP (38.8°C) | HIGH HR (104 bpm) | LOW SpO2 (94.0%)
======================================================================
```

---

## 👤 Author & Connect
- **Developer:** Imran Mian
- **GitHub:** [@memrranmian](https://github.com/memrranmian)
- **Focus:** Data Analytics, Data Science, Scientific Computing, & Machine Learning
