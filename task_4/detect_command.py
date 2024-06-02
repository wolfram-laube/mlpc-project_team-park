data_dir = '../dataset'

import numpy as np
import librosa
import matplotlib.pyplot as plt
from keras.models import load_model

# Function to pad or trim audio segments to a fixed length
def pad_or_trim(segment, target_length):
    if len(segment) > target_length:
        return segment[:target_length]
    elif len(segment) < target_length:
        return np.pad(segment, (0, target_length - len(segment)), mode='constant')
    else:
        return segment

# Function to process audio stream with sliding window
def process_audio_stream(audio, sample_rate, word_model, command_model, word_input_length, command_input_length, window_size=1.0, stride=0.5, framework='keras'):
    window_samples = int(window_size * sample_rate)
    stride_samples = int(stride * sample_rate)
    current_position = 0
    word_buffer = []
    command_detections = []

    while current_position + window_samples <= len(audio):
        segment = audio[current_position:current_position + window_samples]

        # Pad or trim the segment to the required input length for the word model
        word_segment = pad_or_trim(segment, word_input_length)

        # Ensure correct input shape for the word model
        word_segment_input = word_segment.reshape(1, -1)

        if framework == 'keras':
            # Word classification with Keras
            word_prediction = word_model.predict(word_segment_input)
        elif framework == 'pytorch':
            # Word classification with PyTorch
            word_segment_tensor = torch.tensor(word_segment_input, dtype=torch.float32)
            word_prediction = word_model(word_segment_tensor).detach().numpy()

        print(f'Word prediction: {word_prediction}')
        predicted_word = np.argmax(word_prediction)  # Assuming the model returns class probabilities
        print(f'Predicted word: {predicted_word}')

        # Store the word and its timestamp if it is a recognized word
        if predicted_word != 'other':  # Replace 'other' with your actual class for unrecognized words
            timestamp = current_position / sample_rate
            word_buffer.append((timestamp, predicted_word))

            # Check if the buffered words form a recognized command
            if len(word_buffer) > 1:
                words = [w[1] for w in word_buffer]
                command_audio = np.concatenate(
                    [audio[int(w[0] * sample_rate):int((w[0] + window_size) * sample_rate)] for w in word_buffer])

                # Pad or trim the command_audio to the required input length for the command model
                command_audio = pad_or_trim(command_audio, command_input_length)
                command_input = command_audio.reshape(1, -1)

                if framework == 'keras':
                    command_prediction = command_model.predict(command_input)
                elif framework == 'pytorch':
                    command_tensor = torch.tensor(command_input, dtype=torch.float32)
                    command_prediction = command_model(command_tensor).detach().numpy()

                print(f'Command prediction: {command_prediction}')
                predicted_command = np.argmax(command_prediction)
                if predicted_command != 'unrecognized_command':  # Replace with your actual command class / negative detection
                    command_detections.append((word_buffer[0][0], ' '.join(words)))
                    word_buffer = []  # Clear the buffer after recognizing a command

        current_position += stride_samples

    return command_detections

# Load your models and inspect input shapes
# For Keras:
word_model = load_model(f'{data_dir}/word_model.h5')
command_model = load_model(f'{data_dir}/command_model.h5')

word_model_input_shape = word_model.input_shape[1]  # Exclude batch dimension
command_model_input_shape = command_model.input_shape[1]  # Exclude batch dimension

print(f'Word Model Expected Input Shape: {word_model_input_shape}')
print(f'Command Model Expected Input Shape: {command_model_input_shape}')

# For PyTorch:
# word_model = torch.load(f'{data_dir}/word_model.pth')
# command_model = torch.load(f'{data_dir}/command_model.pth')
# word_model.eval()
# command_model.eval()
# word_model_input_shape = word_model.layers[0].in_features
# command_model_input_shape = command_model.layers[0].in_features
# print(f'Word Model Expected Input Shape: {word_model_input_shape}')
# print(f'Command Model Expected Input Shape: {command_model_input_shape}')

# Load the WAV file
wav_file = f'{data_dir}/scenes/wav/2023_speech_true_Licht_an.wav'
audio, sample_rate = librosa.load(wav_file, sr=None)

# Process the audio stream
detections = process_audio_stream(audio, sample_rate, word_model, command_model, word_model_input_shape, command_model_input_shape)

# Output results
for detection in detections:
    print(f"Detected command '{detection[1]}' at {detection[0]:.2f} seconds")
