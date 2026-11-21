# ==============================================================================
# Assignment 6 - Question 2: Hospital Patient Monitoring and Medical Data Analysis System
# Course: Data Analytics / Intro to Data Science (Numpy Practice)
# Author: Student (practicing numpy hands-on after class lectures)
# ==============================================================================

import numpy as np

# Setting seed so random generation and testing stays predictable
np.random.seed(101)

print("=" * 70)
print("QUESTION 2: HOSPITAL PATIENT MONITORING AND MEDICAL DATA ANALYSIS SYSTEM")
print("=" * 70)

# ------------------------------------------------------------------------------
# PART A: Array Creation and Initialization
# ------------------------------------------------------------------------------
print("\n>>> PART A: Array Creation and Initialization")

# A1 & A2: Create a 2D NumPy array containing health measurements of 8 patients using np.array().
# The 4 measurements for each patient are:
# Index 0: Temperature (°C) - normal around 36.5 to 37.5, fever > 38.0
# Index 1: Blood Pressure (Systolic in mmHg) - e.g. 110 - 145
# Index 2: Heart Rate (bpm) - normal 60-100, elevated/tachycardia > 100
# Index 3: Oxygen Saturation (SpO2 in %) - normal 95-100, hypoxemia < 95
#
# Let's populate 8 patients with diverse realistic values so the thresholds will trigger:
# Patient 0: Mild fever (38.5), Normal HR (85), SpO2 (96)
# Patient 1: High fever (39.1), High HR (115), Low SpO2 (92) -> Multiple conditions!
# Patient 2: Normal (36.8, 118, 72, 98)
# Patient 3: High HR only (37.2, 130, 108, 97)
# Patient 4: Low SpO2 only (37.0, 115, 88, 93)
# Patient 5: Normal (36.5, 122, 75, 99)
# Patient 6: Fever (38.8) and High HR (104), SpO2 (94) -> All 3 conditions!
# Patient 7: Normal (37.1, 120, 80, 98)

patient_data = np.array([
    [38.5, 125.0,  85.0, 96.0],  # Patient 0
    [39.1, 140.0, 115.0, 92.0],  # Patient 1
    [36.8, 118.0,  72.0, 98.0],  # Patient 2
    [37.2, 130.0, 108.0, 97.0],  # Patient 3
    [37.0, 115.0,  88.0, 93.0],  # Patient 4
    [36.5, 122.0,  75.0, 99.0],  # Patient 5
    [38.8, 135.0, 104.0, 94.0],  # Patient 6
    [37.1, 120.0,  80.0, 98.0]   # Patient 7
], dtype=float)

print(f"\nA1 & A2. Patient Health Measurements Array (Shape: {patient_data.shape}):")
print("Columns: [Temp (°C), Systolic BP (mmHg), Heart Rate (bpm), SpO2 (%)]")
print(patient_data)

# A3. Create arrays using np.zeros(), np.ones(), and np.eye() for suitable auxiliary calculations or illustrative data.
# For example:
# - np.zeros((8, 4)) can represent baseline adjustments or change tracker
zero_tracker = np.zeros((8, 4))
# - np.ones(4) can represent unit weights or scale factors for the 4 features
feature_weights = np.ones(4)
# - np.eye(4) identity matrix can represent feature covariance or orthogonal indicator for the 4 medical metrics
metric_identity = np.eye(4)

print("\nA3. Auxiliary Arrays Created:")
print("   np.zeros((8, 4)) - Change Tracker (first 2 rows shown):")
print(zero_tracker[:2])
print("   np.ones(4) - Metric Weights:")
print(feature_weights)
print("   np.eye(4) - 4x4 Metric Identity Matrix:")
print(metric_identity)

# A4. Generate an array using np.linspace() to represent a range of normal temperature values.
# Normal clinical body temp range is approximately 36.0°C to 37.5°C, say sampled at 7 reference points
normal_temp_range = np.linspace(36.0, 37.5, num=7)
print("\nA4. Normal Temperature Range (Generated using np.linspace(36.0, 37.5, 7)):")
print(np.round(normal_temp_range, 2))


# ------------------------------------------------------------------------------
# PART B: Statistical Analysis
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> PART B: Statistical Analysis")

metrics = ["Temperature (°C)", "Blood Pressure (mmHg)", "Heart Rate (bpm)", "Oxygen Saturation (%)"]

# B1. Calculate mean, median, maximum, and minimum values of each health measurement.
# Measurements are along columns, so axis=0 computes stats across all patients for each measurement.
means = np.mean(patient_data, axis=0)
medians = np.median(patient_data, axis=0)
maxs = np.max(patient_data, axis=0)
mins = np.min(patient_data, axis=0)

# B2. Calculate standard deviation of each measurement.
stds = np.std(patient_data, axis=0)

print("\nB1 & B2. Summary Statistics per Health Measurement:")
print(f"{'Measurement':<26} | {'Mean':>7} | {'Median':>7} | {'Min':>7} | {'Max':>7} | {'Std Dev':>7}")
print("-" * 75)
for i in range(4):
    print(f"{metrics[i]:<26} | {means[i]:7.2f} | {medians[i]:7.2f} | {mins[i]:7.2f} | {maxs[i]:7.2f} | {stds[i]:7.2f}")

# B3. Identify the patient with the highest heart rate.
# Heart rate is column index 2
heart_rates = patient_data[:, 2]
highest_hr_patient = np.argmax(heart_rates)
print(f"\nB3. Patient with Highest Heart Rate: Patient {highest_hr_patient} ({heart_rates[highest_hr_patient]} bpm)")

# B4. Calculate the average oxygen saturation of all patients.
# Oxygen saturation is column index 3
avg_spo2 = np.mean(patient_data[:, 3])
print(f"\nB4. Average Oxygen Saturation across all patients: {avg_spo2:.2f}%")

# B5. Use appropriate NumPy universal functions for numerical calculations.
# Let's show ufuncs like np.round(), np.sqrt() of variances, np.exp() / np.abs(), np.add() etc.
variances = np.var(patient_data, axis=0)
ufunc_stds = np.sqrt(variances)  # np.sqrt is a universal function!
print(f"\nB5. Universal Function Demo (np.sqrt(np.var(patient_data, axis=0)) == np.std):")
print(f"   Calculated with np.sqrt: {np.round(ufunc_stds, 4)}")
print(f"   Calculated with np.std:  {np.round(stds, 4)}")
print(f"   Are they identical? {np.allclose(ufunc_stds, stds)}")


# ------------------------------------------------------------------------------
# PART C: Boolean Indexing and Patient Identification
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> PART C: Boolean Indexing and Patient Identification")

# Thresholds:
# Temperature > 38.0 °C
# Heart rate > 100 bpm
# Oxygen saturation < 95%

temp_col = patient_data[:, 0]
hr_col = patient_data[:, 2]
spo2_col = patient_data[:, 3]

# C1. Identify patients whose temperature exceeds 38°C.
fever_mask = temp_col > 38.0
fever_patients = np.where(fever_mask)[0]
print(f"\nC1. Patients with Temperature > 38.0°C (Fever):")
print(f"   Boolean Mask: {fever_mask}")
print(f"   Patient IDs:  {fever_patients}")
for p in fever_patients:
    print(f"   - Patient {p}: Temp = {temp_col[p]}°C")

# C2. Identify patients whose heart rate exceeds 100 beats per minute.
high_hr_mask = hr_col > 100.0
high_hr_patients = np.where(high_hr_mask)[0]
print(f"\nC2. Patients with Heart Rate > 100 bpm (Tachycardia):")
print(f"   Boolean Mask: {high_hr_mask}")
print(f"   Patient IDs:  {high_hr_patients}")
for p in high_hr_patients:
    print(f"   - Patient {p}: Heart Rate = {hr_col[p]} bpm")

# C3. Identify patients whose oxygen saturation is below 95%.
low_spo2_mask = spo2_col < 95.0
low_spo2_patients = np.where(low_spo2_mask)[0]
print(f"\nC3. Patients with SpO2 < 95% (Hypoxemia):")
print(f"   Boolean Mask: {low_spo2_mask}")
print(f"   Patient IDs:  {low_spo2_patients}")
for p in low_spo2_patients:
    print(f"   - Patient {p}: SpO2 = {spo2_col[p]}%")

# C4. Use combined Boolean conditions to identify patients meeting more than one of these conditions.
# In Python, each condition returns a boolean array of shape (8,).
# Converting True to 1 and False to 0, we can sum the conditions across each patient!
# If sum > 1, the patient meets more than one critical condition!
condition_matrix = np.column_stack((fever_mask, high_hr_mask, low_spo2_mask))
critical_count = np.sum(condition_matrix, axis=1)
more_than_one_mask = critical_count > 1
critical_patients = np.where(more_than_one_mask)[0]

print(f"\nC4. Patients meeting MORE THAN ONE critical condition:")
print(f"   Condition Counts per patient (Fever + High HR + Low SpO2): {critical_count}")
print(f"   Multi-condition Mask: {more_than_one_mask}")
print(f"   Critical Patient IDs: {critical_patients}")
for p in critical_patients:
    reasons = []
    if fever_mask[p]: reasons.append(f"Fever ({temp_col[p]}°C)")
    if high_hr_mask[p]: reasons.append(f"High HR ({hr_col[p]} bpm)")
    if low_spo2_mask[p]: reasons.append(f"Low SpO2 ({spo2_col[p]}%)")
    print(f"   - Patient {p} meets {critical_count[p]} conditions: {', '.join(reasons)}")

# C5. Use np.where() to obtain positions of patients requiring attention.
# Any patient meeting AT LEAST ONE condition (fever | high_hr | low_spo2) requires attention.
attention_mask = fever_mask | high_hr_mask | low_spo2_mask
attention_patients = np.where(attention_mask)[0]
print(f"\nC5. Positions (Indices) of Patients Requiring Medical Attention (via np.where):")
print(f"   Indices: {attention_patients}")
print(f"   Total patients requiring attention: {len(attention_patients)} out of 8")


# ------------------------------------------------------------------------------
# PART D: Broadcasting and Array Updating
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> PART D: Broadcasting and Array Updating")

# D1. Apply a correction factor to the temperature measurements using broadcasting.
# Suppose digital thermometer had a calibration offset of -0.2°C that needs to be subtracted.
# Temperature is in column 0. We can broadcast a 1D correction vector of length 4: [-0.2, 0.0, 0.0, 0.0]
# or broadcast directly onto column 0.
temp_correction = -0.2
# Broadcasting scalar addition across patient_data[:, 0]:
corrected_temps = patient_data[:, 0] + temp_correction
print(f"\nD1. Temperature Measurements with Calibration Correction ({temp_correction}°C):")
print(f"   Original Temps:  {patient_data[:, 0]}")
print(f"   Corrected Temps: {corrected_temps}")

# D2. Increase or decrease selected measurement values using NumPy array operations.
# For example, let's say patients receiving supplemental oxygen get their SpO2 increased by +3%
# (capping at 100% using np.minimum).
# We can do this on a test slice or array.
spo2_boosted = np.minimum(patient_data[:, 3] + 3.0, 100.0)
print(f"\nD2. Oxygen saturation boosted by +3% (capped at 100%):")
print(f"   Original SpO2: {patient_data[:, 3]}")
print(f"   Boosted SpO2:  {spo2_boosted}")

# D3. Create a new array containing updated patient measurements without changing the original array.
# Must use .copy() so original is not modified!
updated_patient_data = patient_data.copy()
# Apply temperature correction and oxygen boost to the new array:
updated_patient_data[:, 0] = updated_patient_data[:, 0] + temp_correction
updated_patient_data[low_spo2_patients, 3] = np.minimum(updated_patient_data[low_spo2_patients, 3] + 4.0, 100.0)

print("\nD3. Created 'updated_patient_data' as an independent copy and modified it.")

# D4. Compare original and updated measurements.
print("\nD4. Comparison between Original and Updated Patient Measurements:")
print("   Original Patient 1 (before updates):", patient_data[1])
print("   Updated  Patient 1 (after updates): ", updated_patient_data[1])
print(f"   Is original patient_data modified? {not np.array_equal(patient_data[1], [39.1, 140.0, 115.0, 92.0])}")
print("   Difference Matrix (Updated - Original):")
print(updated_patient_data - patient_data)


# ------------------------------------------------------------------------------
# PART E: Reshaping, Transposing, and Array Manipulation
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> PART E: Reshaping, Transposing, and Array Manipulation")

# E1. Reshape the patient data into a 3D array representing two hospital wards,
# each containing four patients and four measurements -> Shape: (2, 4, 4)
# Ward 0: Patients 0, 1, 2, 3
# Ward 1: Patients 4, 5, 6, 7
wards_3d = patient_data.reshape(2, 4, 4)
print(f"\nE1. Reshaped Patient Data into 3D Array (Shape: {wards_3d.shape}):")
print("   --- Ward 0 (Patients 0-3) ---")
print(wards_3d[0])
print("   --- Ward 1 (Patients 4-7) ---")
print(wards_3d[1])

# E2. Transpose the array to change the arrangement of wards, patients, and measurements.
# Original shape: (2 wards, 4 patients, 4 measurements).
# Let's transpose to (4 measurements, 2 wards, 4 patients) using axes (2, 0, 1) or simple .T
transposed_wards = wards_3d.transpose(2, 0, 1)
print(f"\nE2. Transposed Array shape: {transposed_wards.shape} [Measurements x Wards x Patients]")
print("   Transposed Temperature slice across Ward 0 and Ward 1:")
print(transposed_wards[0])

# E3. Use np.swapaxes() to exchange two selected axes.
# Let's swap axis 0 (wards) and axis 1 (patients) in wards_3d -> new shape (4, 2, 4)
swapped_axes_data = np.swapaxes(wards_3d, 0, 1)
print(f"\nE3. Swapped Axes (0 and 1) - Shape changed from (2, 4, 4) to {swapped_axes_data.shape}:")
print("   First patient position across both wards:")
print(swapped_axes_data[0])

# E4. Use flatten() to convert the complete patient data into a 1D array.
flattened_data = patient_data.flatten()
print(f"\nE4. Flattened Patient Data into 1D (Length: {len(flattened_data)}):")
print(flattened_data)

# E5. Sort patient temperature measurements using np.sort().
sorted_temperatures = np.sort(patient_data[:, 0])
print("\nE5. Patient Temperatures Sorted Ascending (using np.sort):")
print(sorted_temperatures)


# ------------------------------------------------------------------------------
# PART F: Copying and Array Operations
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> PART F: Copying and Array Operations")

# F1. Create an alias, shallow copy, and deep copy of the patient measurement array.
# - Alias (view/reference): assignment '='
alias_data = patient_data
# - Shallow copy (view): array.view()
shallow_copy_data = patient_data.view()
# - Deep copy (independent data buffer): array.copy()
deep_copy_data = patient_data.copy()

print("\nF1. Created:")
print("   - alias_data = patient_data (Direct assignment)")
print("   - shallow_copy_data = patient_data.view() (View/shallow copy)")
print("   - deep_copy_data = patient_data.copy() (Independent deep copy)")

# F2. Modify selected patient measurements and observe the impact on the original array.
print("\nF2. Testing Mutation Impact:")
# Let's temporarily change deep_copy_data[0, 0] = 999.0 and check patient_data[0, 0]
deep_copy_data[0, 0] = 999.0
print(f"   Modified deep_copy_data[0, 0] = 999.0")
print(f"   Original patient_data[0, 0] is still: {patient_data[0, 0]} (Protected!)")

# Now let's test shallow_copy_data: if we modify shallow copy, base array changes!
original_val = patient_data[0, 1]
shallow_copy_data[0, 1] = 777.0
print(f"\n   Modified shallow_copy_data[0, 1] = 777.0")
print(f"   Original patient_data[0, 1] changed to: {patient_data[0, 1]} (Because views share memory!)")
# Reverting back original value
patient_data[0, 1] = original_val
print(f"   Reverted patient_data[0, 1] back to: {patient_data[0, 1]}")

# F3. Use np.concatenate() and np.stack() to combine patient records from two different wards.
ward_A = wards_3d[0]  # Patients 0-3 (shape: 4, 4)
ward_B = wards_3d[1]  # Patients 4-7 (shape: 4, 4)

# Concatenate along rows (axis=0) gives back 8 patients x 4 measurements -> (8, 4)
combined_wards_concat = np.concatenate((ward_A, ward_B), axis=0)
print(f"\nF3. np.concatenate((ward_A, ward_B), axis=0) shape: {combined_wards_concat.shape}")

# np.stack along a new dimension (axis=0) creates a 3D array -> (2, 4, 4)
combined_wards_stack = np.stack((ward_A, ward_B), axis=0)
print(f"    np.stack((ward_A, ward_B), axis=0) shape: {combined_wards_stack.shape}")

# F4. Use np.split() to divide the patient records into two groups.
group_1, group_2 = np.split(patient_data, 2, axis=0)
print(f"\nF4. np.split() into two equal patient groups (axis=0):")
print(f"    Group 1 shape: {group_1.shape} (Patients 0-3)")
print(f"    Group 2 shape: {group_2.shape} (Patients 4-7)")


# ------------------------------------------------------------------------------
# PART G: File Handling and Reporting
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print(">>> PART G: File Handling and Reporting")

csv_filename = "question_2/patient_data.csv"

# G1. Save the patient measurement array in a CSV file using np.savetxt().
header_text = "Temperature_C,Systolic_BP_mmHg,Heart_Rate_bpm,Oxygen_Saturation_pct"
np.savetxt(csv_filename, patient_data, delimiter=",", header=header_text, comments="", fmt="%.2f")
print(f"\nG1. Saved patient measurement array to file: '{csv_filename}'")

# G2. Load the saved data using np.loadtxt().
# skiprows=1 skips the header line
loaded_patient_data = np.loadtxt(csv_filename, delimiter=",", skiprows=1)
print(f"\nG2. Successfully loaded data back from '{csv_filename}' (Shape: {loaded_patient_data.shape}):")
print(loaded_patient_data)

# G3. Verify that the loaded data matches the original data.
# Since we saved with %.2f formatting, np.allclose verifies floating point match within tolerance
matches = np.allclose(patient_data, loaded_patient_data, atol=1e-2)
print(f"\nG3. Verification: Does loaded data match original patient data? -> {matches}")

# G4. Generate a patient monitoring report containing average measurements,
# patients requiring attention, and statistical summaries.
print("\n" + "=" * 70)
print("================== HOSPITAL PATIENT MONITORING REPORT ================")
print("=" * 70)
print("1. SUMMARY OF OVERALL WARD VITALS:")
print(f"   - Average Temperature:       {np.mean(patient_data[:, 0]):.2f} °C (Std: {np.std(patient_data[:, 0]):.2f})")
print(f"   - Average Systolic BP:       {np.mean(patient_data[:, 1]):.2f} mmHg (Std: {np.std(patient_data[:, 1]):.2f})")
print(f"   - Average Heart Rate:        {np.mean(patient_data[:, 2]):.2f} bpm (Std: {np.std(patient_data[:, 2]):.2f})")
print(f"   - Average Oxygen Saturation: {np.mean(patient_data[:, 3]):.2f} % (Std: {np.std(patient_data[:, 3]):.2f})")

print("-" * 70)
print("2. PATIENTS REQUIRING ATTENTION (Threshold Alerts):")
print(f"   Total flagged patients: {len(attention_patients)} out of 8")
for p in range(8):
    flags = []
    if fever_mask[p]:
        flags.append(f"HIGH TEMP ({patient_data[p, 0]:.1f}°C > 38.0°C)")
    if high_hr_mask[p]:
        flags.append(f"HIGH HR ({patient_data[p, 2]:.0f} bpm > 100 bpm)")
    if low_spo2_mask[p]:
        flags.append(f"LOW SpO2 ({patient_data[p, 3]:.1f}% < 95.0%)")
    
    if len(flags) > 1:
        severity = "CRITICAL (MULTIPLE CONDITIONS)"
    elif len(flags) == 1:
        severity = "WARNING (SINGLE CONDITION)"
    else:
        severity = "NORMAL / STABLE"
        
    flag_str = " | ".join(flags) if flags else "All vitals in normal range"
    print(f"   - Patient {p} [{severity}]: {flag_str}")

print("-" * 70)
print("3. EXTREME VALUES RECORDED:")
print(f"   - Highest Temp:       Patient {np.argmax(patient_data[:, 0])} ({np.max(patient_data[:, 0])} °C)")
print(f"   - Highest Heart Rate: Patient {np.argmax(patient_data[:, 2])} ({np.max(patient_data[:, 2])} bpm)")
print(f"   - Lowest SpO2:        Patient {np.argmin(patient_data[:, 3])} ({np.min(patient_data[:, 3])} %)")
print("=" * 70)
print("\nQuestion 2 completed successfully!")
