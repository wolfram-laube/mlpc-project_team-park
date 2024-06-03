import os
import random
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

# Function to process audio stream with sliding window
def process_audio_stream(audio, sample_rate, model, input_length, window_size=1.0, stride=0.5, threshold=0.8):
    window_samples = int(window_size * sample_rate)
    stride_samples = int(stride * sample_rate)
    current_position = 0
    command_detections = []

    while current_position + window_samples <= len(audio):
        segment = audio[current_position:current_position + window_samples]

        # Normalize and apply ICA to the segment
        segment = preprocess_segment(segment)

        # Pad or trim the segment to the required input length
        segment = pad_or_trim(segment, input_length)

        # Ensure correct input shape for the model
        segment_input = segment.reshape(1, -1, 1)

        # Command classification
        command_prediction = model.predict(segment_input)
        predicted_command_idx = np.argmax(command_prediction)
        predicted_command = command_class_names[predicted_command_idx]  # Map index to class name
        confidence = command_prediction[0][predicted_command_idx]

        # If the predicted command is not 'command' and confidence is above the threshold, store the detection
        if predicted_command != 'command' and confidence >= threshold:
            timestamp = current_position / sample_rate
            command_detections.append((timestamp, predicted_command))

        current_position += stride_samples

    return command_detections

# Load the annotations file
annotations = pd.read_csv(annotations_file)

# Process each file in the annotations and write predictions to CSV
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['filename', 'command', 'timestamp']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for _, row in tqdm(annotations.iterrows(), desc="Processing files", total=annotations.shape[0]):
        wav_file = row['filename']
        file_path = os.path.join(wav_dir, f"{wav_file}.wav")
        audio, sample_rate = librosa.load(file_path, sr=None)

        # Get the expected input length for the model
        input_length = model.input_shape[1]

        # Process the audio stream
        detections = process_audio_stream(audio, sample_rate, model, input_length, window_size, stride, confidence_threshold)

        # Write results to CSV
        for detection in detections:
            writer.writerow({'filename': wav_file, 'command': detection[1], 'timestamp': detection[0]})

print(f"Predictions saved to {output_csv}")
