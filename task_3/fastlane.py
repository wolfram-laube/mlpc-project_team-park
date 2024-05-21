import pandas as pd

# Load the CSV file to inspect its structure
file_path = '../dataset/development.csv'  # Update with your actual file path
metadata = pd.read_csv(file_path)

# Display the first few rows of the metadata
print(metadata.head())

##############

import os
import pandas as pd
import librosa

# Define the root directory where the dataset is located
root_dir = '../dataset'  # Replace <root> with the actual path to your root directory

# Load the CSV file
file_path = '../dataset/development.csv'  # Update with your actual file path
metadata = pd.read_csv(file_path)

# Function to load a WAV file using the full path
def load_wav(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

# Get the relative file path from the CSV and construct the full path
relative_file_path = metadata.loc[0, 'filename']  # Assuming the column name is 'filename'
full_file_path = os.path.join(root_dir, relative_file_path)

# Load the first audio file
audio, sr = load_wav(full_file_path)

# Print the shape of the audio array and the sample rate
print(audio.shape, sr)

#################

import pandas as pd
import librosa
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the root directory where the dataset is located
root_dir = '../dataset'  # Update with your actual path to the dataset directory

# Load the CSV file
file_path = os.path.join(root_dir, 'development.csv')
metadata = pd.read_csv(file_path)


# Function to load a WAV file using the full path
def load_wav(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr


# Directory to save preprocessed data
preprocessed_dir = 'preprocessed_data'
os.makedirs(preprocessed_dir, exist_ok=True)


# Function to extract the label from the directory name
def extract_label(file_path):
    # Assuming the structure is "root_dir/label/filename.wav"
    return os.path.basename(os.path.dirname(file_path))


# Function to check and handle NaNs and Infs
def check_and_handle_nans_infs(array):
    if np.isnan(array).any() or np.isinf(array).any():
        logging.warning(f"NaNs or Infs found in array: replacing with 0s")
        # Replace NaNs with 0
        array = np.nan_to_num(array)
        # Replace Infs with 0
        array[np.isinf(array)] = 0
    return array


# Function to preprocess and save all audio files
def preprocess_and_save(metadata, root_dir):
    scaler = StandardScaler()
    ica = FastICA(n_components=1, whiten='unit-variance')

    for i, row in metadata.iterrows():
        relative_file_path = row['filename']
        full_file_path = os.path.join(root_dir, relative_file_path)
        label = extract_label(full_file_path)

        try:
            # Load and preprocess audio
            logging.info(f"Processing file {full_file_path}")
            audio, sr = load_wav(full_file_path)
            logging.info(f"Loaded audio shape: {audio.shape}, sample rate: {sr}")

            # Check for NaNs or Infs in the original audio
            audio = check_and_handle_nans_infs(audio)

            audio_scaled = scaler.fit_transform(audio.reshape(-1, 1)).flatten()
            logging.info(f"Scaled audio shape: {audio_scaled.shape}")

            # Check for NaNs or Infs in the scaled audio
            audio_scaled = check_and_handle_nans_infs(audio_scaled)

            audio_ica = ica.fit_transform(audio_scaled.reshape(-1, 1)).flatten()
            logging.info(f"ICA transformed audio shape: {audio_ica.shape}")

            # Save preprocessed audio and label
            np.save(os.path.join(preprocessed_dir, f'audio_{i}.npy'), audio_ica)
            np.save(os.path.join(preprocessed_dir, f'label_{i}.npy'), label)
        except Exception as e:
            logging.error(f"Error processing file {full_file_path}: {e}")
            continue


# Preprocess and save all audio files
preprocess_and_save(metadata, root_dir)


# Function to load preprocessed data
def load_preprocessed_data(preprocessed_dir):
    X = []
    y = []

    for file_name in os.listdir(preprocessed_dir):
        if file_name.startswith('audio'):
            audio = np.load(os.path.join(preprocessed_dir, file_name))
            label_file = file_name.replace('audio', 'label')
            label = np.load(os.path.join(preprocessed_dir, label_file))

            X.append(audio)
            y.append(label)

    return np.array(X), np.array(y)


# Load preprocessed data
X, y = load_preprocessed_data(preprocessed_dir)

# Split data into training, validation, and test sets
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f'Training set size: {len(X_train)}')
print(f'Validation set size: {len(X_val)}')
print(f'Test set size: {len(X_test)}')

import os
import numpy as np


# Function to load preprocessed data
def load_preprocessed_data(preprocessed_dir):
    X = []
    y = []

    for file_name in os.listdir(preprocessed_dir):
        if file_name.startswith('audio'):
            audio = np.load(os.path.join(preprocessed_dir, file_name))
            label_file = file_name.replace('audio', 'label')
            label = np.load(os.path.join(preprocessed_dir, label_file))

            X.append(audio)
            y.append(label)

    return np.array(X), np.array(y)


# Load preprocessed data
preprocessed_dir = 'preprocessed_data'
X, y = load_preprocessed_data(preprocessed_dir)

print(f'Loaded {len(X)} preprocessed audio files.')
print(f'Labels: {set(y)}')

# Split data into training, validation, and test sets
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f'Training set size: {len(X_train)}')
print(f'Validation set size: {len(X_val)}')
print(f'Test set size: {len(X_test)}')



