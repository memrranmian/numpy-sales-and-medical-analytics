# Assignment 6: NumPy Hands-on Practice
**Course:** Data Analytics / Intro to Data Science (Sem 3)  
**Topic:** Array Computing, Numerical Manipulation, and Analytics with NumPy  
**Environment:** Python 3.12+ / NumPy 2.x  

---

## 📌 Project Overview
This repository contains student-level, step-by-step implementations and verified execution results for **Assignment #6: NumPy Foundations and Practical Analytics**. 

The assignment is divided into two primary real-world problem scenarios:
1. **Question 1: E-Commerce Sales and Inventory Analytics System**
   - 2D array representation of 5 products sold across 4 retail branches.
   - Price calculation, broadcasting, revenue aggregations, inventory deduction, and restocking alerts.
   - Dynamic array manipulation (`np.append`, `np.insert`, `np.delete`), concatenation, stacking, splitting, sorting, and reporting.
2. **Question 2: Hospital Patient Monitoring and Medical Data Analysis System**
   - 2D array representing clinical vital signs across 8 patients (Temperature, Blood Pressure, Heart Rate, Oxygen Saturation).
   - Statistical summaries (mean, median, min, max, std dev, universal functions).
   - Complex multi-condition Boolean indexing (Fever > 38°C, Tachycardia > 100 bpm, Hypoxemia SpO2 < 95%).
   - Broadcasting calibration factors, 3D ward reshaping `(2, 4, 4)`, axis swapping, transposition, flattening, memory referencing (alias vs shallow view vs deep copy), CSV file I/O (`savetxt`, `loadtxt`), and automated medical reporting.

---

## 📂 Repository Structure

```text
numpy-sales-health-analytics/
├── Assignment_06_IDS_Numpy.pdf    # Original Assignment specification document
├── README.md                      # Assignment overview & execution instructions (this file)
├── question_1/                    # Question 1: E-Commerce Sales & Inventory Analytics
│   ├── question_1.py              # Working script with beginner-friendly comments
│   └── question_1_output.txt      # Full captured terminal output
└── question_2/                    # Question 2: Hospital Patient Monitoring System
    ├── question_2.py              # Working script with beginner-friendly comments
    ├── question_2_output.txt      # Full captured terminal output
    └── patient_data.csv           # Exported CSV generated during file handling tasks
```

---

## 🚀 How to Run the Code

Make sure you have Python and NumPy installed in your environment:
```powershell
# Verify Python and NumPy
python --version
python -c "import numpy; print(numpy.__version__)"
```

### Running Question 1
Navigate to the root directory or inside `question_1` and execute:
```powershell
python question_1/question_1.py
```
To verify the output against the saved execution log:
```powershell
Get-Content question_1/question_1_output.txt
```

### Running Question 2
Execute Question 2:
```powershell
python question_2/question_2.py
```
This script will also create and verify `question_2/patient_data.csv`. To inspect the saved outputs:
```powershell
Get-Content question_2/question_2_output.txt
Get-Content question_2/patient_data.csv
```

---

## 🎯 Key Concepts Practiced
- **Broadcasting Rules:** Handling mismatched array dimensions (e.g., multiplying `(5, 4)` sales matrix by `(5,)` product prices requiring reshaping to `(5, 1)`).
- **Axis-Specific Reductions:** Knowing when to use `axis=0` (across columns/branches/vitals) vs `axis=1` (across rows/products/patients).
- **Boolean Indexing:** Constructing masks, combining bitwise operators (`&`, `|`), and using `np.where()` for positional index retrieval.
- **Memory Management:** Observing the critical difference between pointer aliasing (`b = a`), shallow views (`b = a.view()`), and independent deep copies (`b = a.copy()`).
- **Data Persistence:** Using `np.savetxt()` with custom CSV delimiters/headers and verifying with `np.loadtxt()` and `np.allclose()`.

---

## 💡 Key Takeaways
- **Broadcasting Dynamics:** Reshaping arrays to align singleton dimensions enables vectorized element-wise arithmetic without expensive loops.
- **Memory Integrity:** Using `.copy()` is vital when preserving baseline data structures against in-place mutations or slice view modifications.
- **Data Persistence:** Seamlessly exporting and importing raw array state using standard delimited formats ensures reproducibility in scientific and analytics pipelines.
