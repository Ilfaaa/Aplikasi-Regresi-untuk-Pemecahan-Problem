import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# Data sampel (Durasi waktu belajar dalam jam dan nilai ujian)
TB = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
NT = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95])

# Model Linear: NT = a * TB + b
def linear_model(TB, a, b):
    return a * TB + b

# Fit data ke model linear
params_linear, _ = curve_fit(linear_model, TB, NT)
a_linear, b_linear = params_linear

# Prediksi menggunakan model linear
NT_pred_linear = linear_model(TB, a_linear, b_linear)

# Model Eksponensial: NT = a * exp(b * TB)
def exponential_model(TB, a, b):
    return a * np.exp(b * TB)

# Fit data ke model eksponensial
params_exponential, _ = curve_fit(exponential_model, TB, NT, p0=(1, 0.1))
a_exponential, b_exponential = params_exponential

# Prediksi menggunakan model eksponensial
NT_pred_exponential = exponential_model(TB, a_exponential, b_exponential)

# Hitung galat RMS
rms_linear = np.sqrt(mean_squared_error(NT, NT_pred_linear))
rms_exponential = np.sqrt(mean_squared_error(NT, NT_pred_exponential))

print(f"Model Linear: NT = {a_linear:.2f} * TB + {b_linear:.2f}")
print(f"Model Eksponensial: NT = {a_exponential:.2f} * exp({b_exponential:.2f} * TB)")
print(f"Galat RMS Model Linear: {rms_linear:.2f}")
print(f"Galat RMS Model Eksponensial: {rms_exponential:.2f}")

# Plot data dan hasil regresi
plt.figure(figsize=(12, 6))

# Plot titik data
plt.scatter(TB, NT, color='blue', label='Data')

# Plot hasil regresi linear
plt.plot(TB, NT_pred_linear, color='red', label='Regresi Linear')

# Plot hasil regresi eksponensial
plt.plot(TB, NT_pred_exponential, color='green', label='Regresi Eksponensial')

# Tambahkan label dan legend
plt.xlabel('Durasi Waktu Belajar (TB)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Linear dan Eksponensial pada Data Nilai Ujian')
plt.legend()

# Tampilkan plot
plt.show()