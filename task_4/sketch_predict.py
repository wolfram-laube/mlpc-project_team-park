import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm
import logging
import random
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_dir = '../dataset'
audio_dir = f'{data_dir}/scenes/wav'
meta_dir = f'{data_dir}/meta'
kan_model_path = 'best_command_kan_model.keras'
output_file = os.path.join(data_dir, 'predictions/predictions.csv')
command_mapping_path = os.path.join(meta_dir, 'command_mapping.json')

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Load mean and std for normalization
mean = np.load(os.path.join(meta_dir, 'mean.npy'))
std = np.load(os.path.join(meta_dir, 'std.npy'))


# Define KolmogorovArnoldNetwork class
class KolmogorovArnoldNetwork(tf.keras.Model):
    def __init__(self, input_shape, num_univariate_functions, hidden_units, num_classes, **kwargs):
        super(KolmogorovArnoldNetwork, self).__init__(**kwargs)
        self.num_univariate_functions = num_univariate_functions
        self.hidden_units = hidden_units

        # Univariate functions layers
        self.univariate_layers = [tf.keras.layers.Dense(hidden_units, activation='tanh') for _ in
                                  range(num_univariate_functions)]

        # Combination layer
        self.combination_layer = tf.keras.layers.Dense(hidden_units, activation='relu')

        # Output layer
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # Apply univariate functions
        univariate_outputs = [layer(inputs) for layer in self.univariate_layers]

        # Combine outputs
        combined_output = tf.concat(univariate_outputs, axis=-1)
        combined_output = self.combination_layer(combined_output)

        # Flatten and apply final output layer
        combined_output = tf.reduce_mean(combined_output, axis=2)  # Average over feature dimension
        return self.output_layer(combined_output)


# Load the KAN model
logging.info('Loading KAN model...')
kan_model = load_model(kan_model_path, custom_objects={'KolmogorovArnoldNetwork': KolmogorovArnoldNetwork})
logging.info('KAN model loaded.')

# Load command mapping
with open(command_mapping_path, 'r') as f:
    command_mapping = json.load(f)
inverse_command_mapping = {v: k for k, v in command_mapping.items()}


# Function to extract features from an audio segment
def extract_features(y, sr, max_len=None):
    features = []

    # Extract various audio features
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    for i in range(mel_spec.shape[0]):
        features.append(mel_spec[i])

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)
    for i in range(mfcc.shape[0]):
        features.append(mfcc[i])

    mfcc_delta = librosa.feature.delta(mfcc)
    for i in range(mfcc_delta.shape[0]):
        features.append(mfcc_delta[i])

    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    for i in range(mfcc_delta2.shape[0]):
        features.append(mfcc_delta2[i])

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    for i in range(bandwidth.shape[0]):
        features.append(bandwidth[i])

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    for i in range(centroid.shape[0]):
        features.append(centroid[i])

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(contrast.shape[0]):
        features.append(contrast[i])

    flatness = librosa.feature.spectral_flatness(y=y)
    for i in range(flatness.shape[0]):
        features.append(flatness[i])

    flux = librosa.onset.onset_strength(y=y, sr=sr)
    features.append(np.broadcast_to(flux, (1, len(flux))))

    rms = librosa.feature.rms(y=y)
    for i in range(rms.shape[0]):
        features.append(rms[i])

    zcr = librosa.feature.zero_crossing_rate(y)
    for i in range(zcr.shape[0]):
        features.append(zcr[i])

    power = librosa.feature.rms(y=y)
    for i in range(power.shape[0]):
        features.append(power[i])

    yin = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    features.append(np.broadcast_to(yin, (1, len(yin))))

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(chroma_stft.shape[0]):
        features.append(chroma_stft[i])

    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    for i in range(chroma_cqt.shape[0]):
        features.append(chroma_cqt[i])

    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    for i in range(chroma_cens.shape[0]):
        features.append(chroma_cens[i])

    harmonics = librosa.effects.harmonic(y)
    if max_len and len(harmonics) > max_len:
        harmonics = harmonics[:max_len]
    features.append(np.broadcast_to(harmonics, (1, len(harmonics))))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    for i in range(rolloff.shape[0]):
        features.append(rolloff[i])

    return features


# Function to pad features to the maximum length
def pad_features(features, max_len):
    padded_features = []
    for feature in features:
        if isinstance(feature, np.ndarray):
            if feature.ndim == 1:
                feature = np.expand_dims(feature, axis=0)
            if feature.shape[1] < max_len:
                feature = np.pad(feature, ((0, 0), (0, max_len - feature.shape[1])), mode='constant')
        else:
            feature = np.array([feature] * max_len)
        padded_features.append(feature)
    return np.array(padded_features)


# Function to prepare a segment for prediction
def prepare_segment(segment, sample_rate, mean, std, max_feature_len):
    features = extract_features(segment, sample_rate, max_feature_len)
    features = pad_features(features, max_feature_len)
    features = (features - mean) / std  # Normalize features
    return np.expand_dims(features, axis=0)  # Add batch dimension


# Function to detect commands in an audio file
def detect_command_pairs(audio, sample_rate, model, max_feature_len, window_size=1.0, step_size=0.5, threshold=0.5):
    audio_len = len(audio) / sample_rate
    command_segments = []

    for start in np.arange(0, audio_len, step_size):
        end = min(start + window_size, audio_len)
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segment = audio[start_sample:end_sample]

        if len(segment) == 0:
            continue

        segment_features = prepare_segment(segment, sample_rate, mean, std, max_feature_len)
        prediction = model.predict(segment_features)
        predicted_label = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]

        if confidence > threshold:
            command_segments.append((start, end, inverse_command_mapping[predicted_label], confidence))

    return command_segments


# Load annotations
annotations = pd.read_csv(os.path.join(meta_dir, 'development_scene_annotations.csv'))

# Randomly select ten files
selected_annotations = annotations.sample(n=10, random_state=42)

# Find maximum feature length from the selected files
max_feature_len = 0
for index, row in tqdm(selected_annotations.iterrows(), total=selected_annotations.shape[0]):
    file_path = os.path.join(audio_dir, f"{row['filename']}.wav")
    y, sr = librosa.load(file_path, sr=None)
    features = extract_features(y, sr)
    for feature in features:
        if isinstance(feature, np.ndarray):
            if feature.ndim > 1:
                max_feature_len = max(max_feature_len, feature.shape[1])
            else:
                max_feature_len = max(max_feature_len, feature.shape[0])

# List to store all predictions
all_predictions = []

# Predict commands for each selected audio file
for index, row in tqdm(selected_annotations.iterrows(), total=selected_annotations.shape[0]):
    file_path = os.path.join(audio_dir, f"{row['filename']}.wav")
    y, sr = librosa.load(file_path, sr=None)
    predictions = detect_command_pairs(y, sr, kan_model, max_feature_len)
    for pred in predictions:
        all_predictions.append([row['filename']] + list(pred))

# Save all predictions to a single CSV file
df = pd.DataFrame(all_predictions, columns=['filename', 'start', 'end', 'command', 'confidence'])
df.to_csv(output_file, index=False)

logging.info('Prediction complete.')
