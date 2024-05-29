### Step 1: Load Metadata and Annotations

import pandas as pd

# Define file paths
metadata_file = '../dataset/development_scenes.csv'
annotations_file = '../dataset/development_scene_annotations.csv'

# Load the CSV files
metadata = pd.read_csv(metadata_file)
annotations = pd.read_csv(annotations_file)

# Display the first few rows of each file to understand their structure
print("Metadata:")
print(metadata.head())
print("\nAnnotations:")
print(annotations.head())

### Step 2: Preprocess Audio Files (WAV)

import os
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
import numpy as np

# Define directories
scenes_dir = '../dataset/scenes/wav'  # Updated path


# Function to load and preprocess audio
def load_and_preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    scaler = StandardScaler()
    audio = scaler.fit_transform(audio.reshape(-1, 1)).flatten()

    ica = FastICA(n_components=1, whiten='unit-variance')
    audio = ica.fit_transform(audio.reshape(-1, 1)).flatten()

    return audio, sr


# Function to segment audio based on annotations
def segment_audio(audio, sr, start, end):
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    return audio[start_sample:end_sample]


# Preprocess and segment all audio files
preprocessed_segments = []

for idx, row in annotations.iterrows():
    file_name = row['filename']
    command = row['command']
    start = row['start']
    end = row['end']

    file_path = os.path.join(scenes_dir, file_name + '.wav')

    try:
        audio, sr = load_and_preprocess_audio(file_path)
        segment = segment_audio(audio, sr, start, end)
        preprocessed_segments.append((file_name, command, segment, sr))
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Example: display the first segment
print(f"First segment: {preprocessed_segments[0]}")

### Step 3: Feature Extraction

import librosa
import numpy as np

# Function to extract features with padding/truncation and dynamic n_fft
def extract_features(segment, sr, max_length):
    # Use a smaller n_fft for short segments
    n_fft = min(2048, len(segment))
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13, n_fft=n_fft)
    mfccs_flat = mfccs.flatten()
    if len(mfccs_flat) < max_length:
        # Pad with zeros
        padded_mfccs = np.pad(mfccs_flat, (0, max_length - len(mfccs_flat)), mode='constant')
        return padded_mfccs
    else:
        # Truncate to max_length
        return mfccs_flat[:max_length]

# Determine the maximum length for padding/truncation
max_length = 0
for _, _, segment, sr in preprocessed_segments:
    n_fft = min(2048, len(segment))
    features = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13, n_fft=n_fft).flatten()
    if len(features) > max_length:
        max_length = len(features)

# Extract features for all segments with padding/truncation
feature_data = []

for file_name, command, segment, sr in preprocessed_segments:
    features = extract_features(segment, sr, max_length)
    feature_data.append((file_name, command, features))

# Example: display the first feature set
print(f"First feature set: {feature_data[0]}")

### Step 4: Model Training and Evaluation
#### Step 4.1: Investigate Class Distribution

import numpy as np
import pandas as pd
from collections import Counter

# Prepare data for training
X = np.array([features for _, _, features in feature_data])
y = np.array([command for _, command, _ in feature_data])

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Check class distribution in training and validation sets
train_distribution = Counter(y_train)
val_distribution = Counter(y_val)

print("Training set distribution:")
print(pd.DataFrame(train_distribution.items(), columns=['Command', 'Count']))

print("\nValidation set distribution:")
print(pd.DataFrame(val_distribution.items(), columns=['Command', 'Count']))

#### Step 4.2: Adjust `zero_division` Parameter and Analyze Model Performance

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the classifier with zero_division set to handle undefined metrics
y_pred = rf.predict(X_val)
print(classification_report(y_val, y_pred, zero_division=0))

# Identify which classes are not being predicted
missing_classes = set(val_distribution.keys()) - set(np.unique(y_pred))
print("Classes not predicted at all:")
print(missing_classes)

#### Optional: Handle Class Imbalance

from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}

# Train a Random Forest classifier with class weights
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
rf.fit(X_train, y_train)

# Evaluate the classifier with zero_division set to handle undefined metrics
y_pred = rf.predict(X_val)
print(classification_report(y_val, y_pred, zero_division=0))
