import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import os
import random

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Lấy danh sách tệp tin CSV trong thư mục
csv_files_folder1 = glob.glob('Data_Train/*.csv')
csv_files_folder2 = glob.glob('Data_Test/*.csv')

fig, axs = plt.subplots(3, 2, figsize=(10, 10))
axs = axs.flatten()

# Tạo thư mục mới để lưu tệp tin CSV
new_folder_folder1 = 'Processed_Data_Train'
new_folder_folder2 = 'Processed_Data_Test'
Path(new_folder_folder1).mkdir(parents=True, exist_ok=True)
Path(new_folder_folder2).mkdir(parents=True, exist_ok=True)

for file_path in csv_files_folder1:
    # Đọc dữ liệu từ tệp tin CSV
    data = pd.read_csv(file_path, header=None, skiprows=2, nrows=200)

    # Chuyển dữ liệu thành mảng numpy
    samples = data.values.astype(float)

    # Biến đổi Fourier
    fft_result = np.fft.fft(samples, axis=0)
    freq = np.fft.fftfreq(samples.shape[0])
    fft_magnitude = np.abs(fft_result)
    freq_positive = freq[np.where(freq >= 0)]

    for i in range(samples.shape[1]):
        axs[i].plot(freq_positive, abs(fft_result[:len(freq_positive), i]))
        axs[i].set_xlabel('Frequency')
        axs[i].set_ylabel('Magnitude')
        axs[i].set_title(f'Fourier Transform of Column {i + 1}')
        axs[i].grid(True)

    # Lấy 100 giá trị
    samples_100 = fft_magnitude[:100]

    # Tạo DataFrame từ mảng numpy
    df_100 = pd.DataFrame(samples_100, columns=["Gx", "Gy", "Gz", "Ax", "Ay", "Az"])

    # Trích xuất tên file từ file_path
    file_name = Path(file_path).name

    # Tạo đường dẫn tệp tin CSV mới
    new_file_path = os.path.join(new_folder_folder1, file_name)

    # Lưu DataFrame vào tệp tin CSV mới
    df_100.to_csv(new_file_path, index=False)

for file_path in csv_files_folder2:
    # Đọc dữ liệu từ tệp tin CSV
    data = pd.read_csv(file_path, header=None, skiprows=2, nrows=200)

    # Chuyển dữ liệu thành mảng numpy
    samples = data.values.astype(float)

    # Biến đổi Fourier
    fft_result = np.fft.fft(samples, axis=0)
    freq = np.fft.fftfreq(samples.shape[0])
    fft_magnitude = np.abs(fft_result)
    freq_positive = freq[np.where(freq >= 0)]

    for i in range(samples.shape[1]):
        axs[i].plot(freq_positive, abs(fft_result[:len(freq_positive), i]))
        axs[i].set_xlabel('Frequency')
        axs[i].set_ylabel('Magnitude')
        axs[i].set_title(f'Fourier Transform of Column {i + 1}')
        axs[i].grid(True)

    # Lấy 100 giá trị
    samples_100 = fft_magnitude[:100]

    # Tạo DataFrame từ mảng numpy
    df_100 = pd.DataFrame(samples_100, columns=["Gx", "Gy", "Gz", "Ax", "Ay", "Az"])

    # Trích xuất tên file từ file_path
    file_name = Path(file_path).name

    # Tạo đường dẫn tệp tin CSV mới
    new_file_path = os.path.join(new_folder_folder2, file_name)

    # Lưu DataFrame vào tệp tin CSV mới
    df_100.to_csv(new_file_path, index=False)

plt.tight_layout()
plt.show()

# SVM
train_features = []
train_labels = []
test_features = []
test_labels = []

for file_path in glob.glob('Processed_Data_Train/*.csv'):
    data = pd.read_csv(file_path)
    data.columns = ["Gx", "Gy", "Gz", "Ax", "Ay", "Az"]
    train_features.append(data.values.flatten())

    file_name = Path(file_path).name
    gender_label = random.choice(['0', '1'])
    train_labels.append((file_name, gender_label))

for file_path in glob.glob('Processed_Data_Test/*.csv'):
    data = pd.read_csv(file_path)
    data.columns = ["Gx", "Gy", "Gz", "Ax", "Ay", "Az"]
    test_features.append(data.values.flatten())

    file_name = Path(file_path).name
    gender_label = random.choice(['0', '1'])
    test_labels.append((file_name, gender_label))

svm = SVC(kernel='linear')
svm.fit(train_features, [label for _, label in train_labels])

predictions = svm.predict(test_features)

for (file, label), prediction in zip(test_labels, predictions):
    print(f"File: {file}, Label: {label}")
    gender = 'Female' if prediction == '0' else 'Male'
    print(f"Predicted Gender: {gender}")

accuracy = accuracy_score([label for _, label in test_labels], predictions)
print("Accuracy:", accuracy)
