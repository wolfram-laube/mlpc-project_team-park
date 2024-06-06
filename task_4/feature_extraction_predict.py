import os
import numpy as np
import tensorflow as tf
import logging
import pandas as pd
from tqdm import tqdm
import librosa
import csv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_dir = '../dataset'
model_save_path = 'best_command_model.keras'
annotations_file = f'development_scene_annotations.csv'
output_csv = 'my_predictions.csv'
window_size = 2  # window size in seconds
stride = 1  # stride size in seconds
confidence_threshold = 0.5  # confidence threshold for predictions
target_frames = 1437  # The number of frames used in training

# Load annotations to get command mapping
logging.info('Loading annotations...')
annotations = pd.read_csv(annotations_file)
logging.info('Annotations loaded.')

# Extract command mapping
command_mapping = {}
current_label = 0
for command in annotations['command'].unique():
    command_mapping[command] = current_label
    current_label += 1

# Reverse command mapping
reverse_command_mapping = {v: k for k, v in command_mapping.items()}

# Load the best model
logging.info('Loading the best model...')
command_model = tf.keras.models.load_model(model_save_path)
logging.info('Model loaded.')


# Function to prepare feature data (from the training script)
def prepare_feature_data(annotations, data_dir, feature_dir):
    command_features = []
    command_labels = []
    command_mapping = {}  # Mapping of command texts to numerical labels
    current_label = 0
    max_len = 0  # To determine the maximum length of features

    logging.info('Preparing command data...')
    for index, row in tqdm(annotations.iterrows(), total=annotations.shape[0]):
        feature_path = os.path.join(feature_dir, row['filename'] + '.npy')
        features = np.load(feature_path)
        max_len = max(max_len, features.shape[1])  # Update max_len

        command_text = row['command']
        if command_text not in command_mapping:
            command_mapping[command_text] = current_label
            current_label += 1

        command_label = command_mapping[command_text]

        command_features.append(features)
        command_labels.append(command_label)

    # Pad features to the same length
    padded_features = []
    for feature in command_features:
        pad_width = max_len - feature.shape[1]
        if pad_width > 0:
            feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
        padded_features.append(feature)

    logging.info('Command data prepared.')
    return np.array(padded_features), np.array(command_labels), command_mapping, max_len


# Calculate mean, std, and max_len from training data
feature_dir = f'{data_dir}/scenes/npy'
command_features, _, _, max_len = prepare_feature_data(annotations, data_dir, feature_dir)
mean = np.mean(command_features, axis=(0, 2), keepdims=True)
std = np.std(command_features, axis=(0, 2), keepdims=True)

logging.info(f'mean shape: {mean.shape}, std shape: {std.shape}')


# Function to prepare new data for prediction from audio
def prepare_segment(audio_segment, sample_rate, mean, std, target_frames):
    features = librosa.feature.melspectrogram(y=audio_segment, sr=sample_rate, n_mels=175)
    features_db = librosa.power_to_db(features, ref=np.max)
    # Pad or trim features to target_frames
    if features_db.shape[1] < target_frames:
        features_padded = np.pad(features_db, ((0, 0), (0, target_frames - features_db.shape[1])), mode='constant')
    else:
        features_padded = features_db[:, :target_frames]

    logging.info(
        f'features_padded shape: {features_padded.shape}, mean shape: {mean.squeeze().shape}, std shape: {std.squeeze().shape}')

    features_normalized = (features_padded - mean.squeeze(axis=0)) / std.squeeze(axis=0)  # Remove singleton dimensions
    features_normalized = features_normalized.T  # Ensure the shape matches (time, features)
    features_normalized = features_normalized.reshape(1, features_normalized.shape[1], features_normalized.shape[0])
    return features_normalized


# Function to detect command pairs in the audio
def detect_command_pairs(audio, sample_rate, model, window_size, step_size, threshold, target_frames):
    command_segments = []
    audio_length = len(audio)
    window_length = int(window_size * sample_rate)
    step_length = int(step_size * sample_rate)

    for start in range(0, audio_length - window_length + 1, step_length):
        end = start + window_length
        segment = audio[start:end]
        segment_features = prepare_segment(segment, sample_rate, mean, std, target_frames)
        prediction = model.predict(segment_features)
        confidence = np.max(prediction)
        if confidence >= threshold:
            command_segments.append(
                (start / sample_rate, segment_features, start, end))  # Include start time and end time
    return command_segments


# Process each file in the annotations and write predictions to CSV
with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['filename', 'command', 'timestamp', 'confidence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for _, row in tqdm(annotations.iterrows(), desc="Processing files", total=annotations.shape[0]):
        wav_file = row['filename']
        file_path = os.path.join(f'{data_dir}/scenes/wav', f"{wav_file}.wav")
        audio, sample_rate = librosa.load(file_path, sr=None)

        # Detect command segments with high confidence
        command_segments = detect_command_pairs(audio, sample_rate, command_model, window_size=window_size,
                                                step_size=stride, threshold=confidence_threshold,
                                                target_frames=target_frames)

        # Process the detected command segments
        for timestamp, segment_features, start, end in command_segments:
            command_prediction = command_model.predict(segment_features)
            predicted_command_idx = np.argmax(command_prediction)
            predicted_command = reverse_command_mapping[predicted_command_idx]
            confidence = command_prediction[0][predicted_command_idx]

            if confidence >= confidence_threshold:
                writer.writerow({'filename': wav_file, 'command': predicted_command, 'timestamp': timestamp,
                                 'confidence': confidence})

print(f"Predictions saved to {output_csv}")
