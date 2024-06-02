import numpy as np
import librosa
import json
from keras.models import load_model

data_dir = '../dataset'

# Load the model
command_model = load_model(f'{data_dir}/command_model.h5')

# Load the class names
with open(f'{data_dir}/command_class_names.json', 'r') as f:
    command_class_names = json.load(f)

print(f'Command Class Names: {command_class_names}')


# Function to pad or trim audio segments to a fixed length
def pad_or_trim(segment, target_length):
    if len(segment) > target_length:
        return segment[:target_length]
    elif len(segment) < target_length:
        return np.pad(segment, (0, target_length - len(segment)), mode='constant')
    else:
        return segment


# Function to process audio stream with sliding window
def process_audio_stream(audio, sample_rate, command_model, input_length, command_class_names, window_size=1.0,
                         stride=0.5, framework='keras'):
    window_samples = int(window_size * sample_rate)
    stride_samples = int(stride * sample_rate)
    current_position = 0
    command_detections = []

    while current_position + window_samples <= len(audio):
        segment = audio[current_position:current_position + window_samples]

        # Pad or trim the segment to the required input length for the command model
        segment_padded = pad_or_trim(segment, input_length)

        # Ensure correct input shape for the command model
        segment_input = segment_padded.reshape(1, -1, 1)

        if framework == 'keras':
            # Command classification with Keras
            command_prediction = command_model.predict(segment_input)
        elif framework == 'pytorch':
            # Command classification with PyTorch
            segment_tensor = torch.tensor(segment_input, dtype=torch.float32)
            command_prediction = command_model(segment_tensor).detach().numpy()

        predicted_command_idx = np.argmax(command_prediction)
        predicted_command = command_class_names[predicted_command_idx]  # Map index to class name
        print(f'Predicted command: {predicted_command}')

        # Store the command and its timestamp if it is a recognized command
        if predicted_command != 'unrecognized_command':  # Replace with your actual command class / negative detection
            timestamp = current_position / sample_rate
            command_detections.append((timestamp, predicted_command))

        current_position += stride_samples

    return command_detections


# Load the WAV file
wav_file = f'{data_dir}/scenes/wav/2023_speech_true_Licht_an.wav'
audio, sample_rate = librosa.load(wav_file, sr=None)

# Determine the input length expected by the command model
input_length = command_model.input_shape[1]  # Exclude batch dimension

# Process the audio stream
detections = process_audio_stream(audio, sample_rate, command_model, input_length, command_class_names)

# Output results
for detection in detections:
    print(f"Detected command '{detection[1]}' at {detection[0]:.2f} seconds")
