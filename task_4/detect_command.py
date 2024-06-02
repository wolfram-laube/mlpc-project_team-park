import os
import random
import numpy as np
import librosa
import matplotlib.pyplot as plt
from keras.models import load_model
import json
from tqdm import tqdm

# Configuration
data_dir = '../dataset'
wav_dir = os.path.join(data_dir, 'scenes/wav')
model_save_path = os.path.join(data_dir, 'command_model.h5')
label_metadata_path = os.path.join(data_dir, 'command_class_names.json')
num_files_to_process = 5  # Number of files to process

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


# Function to process audio stream with sliding window
def process_audio_stream(audio, sample_rate, model, input_length, window_size=1.0, stride=0.5):
    window_samples = int(window_size * sample_rate)
    stride_samples = int(stride * sample_rate)
    current_position = 0
    command_detections = []

    while current_position + window_samples <= len(audio):
        segment = audio[current_position:current_position + window_samples]

        # Pad or trim the segment to the required input length
        segment = pad_or_trim(segment, input_length)

        # Ensure correct input shape for the model
        segment_input = segment.reshape(1, -1)

        # Command classification
        command_prediction = model.predict(segment_input)
        predicted_command_idx = np.argmax(command_prediction)
        predicted_command = command_class_names[predicted_command_idx]  # Map index to class name

        # If the predicted command is not 'unrecognized_command', store the detection
        if predicted_command != 'unrecognized_command':
            timestamp = current_position / sample_rate
            command_detections.append((timestamp, predicted_command))

        current_position += stride_samples

    return command_detections


# Get a list of all WAV files in the directory
wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]

# Randomly select a subset of files to process
selected_files = random.sample(wav_files, min(num_files_to_process, len(wav_files)))

# Process each selected file
for wav_file in tqdm(selected_files, desc="Processing files"):
    file_path = os.path.join(wav_dir, wav_file)
    audio, sample_rate = librosa.load(file_path, sr=None)

    # Get the expected input length for the model
    input_length = model.input_shape[1]

    # Process the audio stream
    detections = process_audio_stream(audio, sample_rate, model, input_length, stride=0.1)

    # Output results
    print(f"\nResults for {wav_file}:")
    for detection in detections:
        print(f"Detected command '{detection[1]}' at {detection[0]:.2f} seconds")
