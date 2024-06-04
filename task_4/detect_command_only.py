import numpy as np
import librosa
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
import json
import os

# Configuration
data_dir = '../dataset'
model_save_path = os.path.join(data_dir, 'command_model.h5')
label_metadata_path = os.path.join(data_dir, 'command_class_names.json')
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


# Function to detect command pairs
def detect_command_pairs(audio_path, model, sr=16000, window_size=1.5, step_size=0.2):
    y, _ = librosa.load(audio_path, sr=sr)
    window_samples = int(window_size * sr)
    step_samples = int(step_size * sr)

    windows = librosa.util.frame(y, frame_length=window_samples, hop_length=step_samples)
    windows = windows.T.reshape((windows.shape[1], windows.shape[0], 1))

    predictions = model.predict(windows)

    command_segments = []
    command_predictions = []

    for i, window in enumerate(windows):
        segment = y[i * step_samples:i * step_samples + window_samples]
        segment = preprocess_segment(segment)
        segment = pad_or_trim(segment, model.input_shape[1])

        prediction = model.predict(segment.reshape(1, -1, 1))
        predicted_command_idx = np.argmax(prediction)
        predicted_command = command_class_names[predicted_command_idx]
        confidence = prediction[0][predicted_command_idx]

        if confidence >= confidence_threshold:
            command_segments.append(segment)
            command_predictions.append((predicted_command, confidence))

    return command_segments, command_predictions


# Example usage
audio_path = os.path.join(data_dir, 'scenes/wav/2023_speech_true_Licht_an.wav')
command_segments, command_predictions = detect_command_pairs(audio_path, model)

# Print the recognized commands
for command, confidence in command_predictions:
    print(f"Command: {command}, Confidence: {confidence}")
