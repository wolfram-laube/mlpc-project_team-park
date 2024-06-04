import os
import numpy as np
import pandas as pd
import librosa
import json
import csv
from tqdm import tqdm
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA

# Configuration
data_dir = '../dataset'
wav_dir = os.path.join(data_dir, 'scenes/wav')
annotations_file = os.path.join(data_dir, 'development_scene_annotations.csv')
model_save_path = os.path.join(data_dir, 'command_model.h5')
label_metadata_path = os.path.join(data_dir, 'command_class_names.json')
output_csv = 'my_predictions.csv'
window_size = 1.5  # Window size in seconds
stride = 0.2  # Stride in seconds
confidence_threshold = 0.8  # Confidence threshold for predictions

# Load the model
model = load_model(model_save_path)

# Get the expected input length for the model
input_length = model.input_shape[1]

# Load the class names
with open(label_metadata_path, 'r') as f:
    command_class_names = json.load(f)


# Function to pad or trim audio segments to a fixed length
def pad_or_trim(segment, target_length):
    if len(segment) > target_length:
        return segment[:target_length]
    elif len(segment) < target_length:
        return np.pad(segment, (0, target_length - len(segment)), mode='constant')
    else:
        return segment


# Function to normalize and apply ICA
def preprocess_segment(segment):
    scaler = StandardScaler()
    ica = FastICA(n_components=1, whiten='unit-variance')

    segment_scaled = scaler.fit_transform(segment.reshape(-1, 1)).flatten()
    segment_scaled = np.nan_to_num(segment_scaled)
    segment_scaled[np.isinf(segment_scaled)] = 0

    segment_ica = ica.fit_transform(segment_scaled.reshape(-1, 1)).flatten()
    segment_ica = np.nan_to_num(segment_ica)
    segment_ica[np.isinf(segment_ica)] = 0

    return segment_ica


# Function to detect command pairs with high confidence
def detect_command_pairs(audio, sample_rate, model, window_size=1.5, step_size=0.2, threshold=0.8):
    window_samples = int(window_size * sample_rate)
    step_samples = int(step_size * sample_rate)

    windows = librosa.util.frame(audio, frame_length=window_samples, hop_length=step_samples).T
    windows = windows.reshape((windows.shape[0], windows.shape[1], 1))

    predictions = []
    segments = []

    for i, window in enumerate(windows):
        # Normalize and apply ICA to the segment
        segment = preprocess_segment(window)

        # Pad or trim the segment to the required input length
        segment = pad_or_trim(segment, input_length)

        # Ensure correct input shape for the model
        segment_input = segment.reshape(1, -1, 1)

        # Command classification
        command_prediction = model.predict(segment_input)
        predictions.append(command_prediction)
        segments.append((i * step_samples, segment_input))

    predictions = np.array(predictions).reshape(-1, model.output_shape[-1])
    boundaries = np.where(predictions > threshold)[0]

    command_segments = []
    for start in boundaries:
        timestamp = start * step_samples / sample_rate
        command_segments.append((timestamp, segments[start][1]))

    return command_segments


# Load the annotations file
annotations = pd.read_csv(annotations_file)

# Process each file in the annotations and write predictions to CSV
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['filename', 'command', 'timestamp', 'confidence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for _, row in tqdm(annotations.iterrows(), desc="Processing files", total=annotations.shape[0]):
        wav_file = row['filename']
        file_path = os.path.join(wav_dir, f"{wav_file}.wav")
        audio, sample_rate = librosa.load(file_path, sr=None)

        # Detect command segments with high confidence
        command_segments = detect_command_pairs(audio, sample_rate, model, window_size=window_size, step_size=stride,
                                                threshold=confidence_threshold)

        # Process the detected command segments
        for timestamp, segment in command_segments:
            command_prediction = model.predict(segment)
            predicted_command_idx = np.argmax(command_prediction)
            predicted_command = command_class_names[predicted_command_idx]
            confidence = command_prediction[0][predicted_command_idx]

            if predicted_command != 'command' and confidence >= confidence_threshold:
                writer.writerow({'filename': wav_file, 'command': predicted_command, 'timestamp': timestamp,
                                 'confidence': confidence})

print(f"Predictions saved to {output_csv}")
