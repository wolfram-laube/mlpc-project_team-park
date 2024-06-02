import time

start = time.time()
#######################################
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json

data_dir = '../dataset'
model_save_path = f'{data_dir}/command_model.h5'

# Load the CSV file containing the annotations
annotations = pd.read_csv(f'{data_dir}/development_scene_annotations.csv')


# Function to check and handle NaNs and Infs
def check_and_handle_nans_infs(array):
    if np.isnan(array).any() or np.isinf(array).any():
        array = np.nan_to_num(array)
        array[np.isinf(array)] = 0
    return array


# Function to augment audio data
def augment_audio(audio, sample_rate):
    augmented_audio = []

    # Original
    augmented_audio.append(audio)

    # Add noise
    noise = np.random.randn(len(audio))
    audio_noise = audio + 0.005 * noise
    augmented_audio.append(audio_noise)

    # Time shift
    shift_range = int(sample_rate * 0.1)  # shift by up to 10% of sample rate
    shift = np.random.randint(-shift_range, shift_range)
    audio_shift = np.roll(audio, shift)
    augmented_audio.append(audio_shift)

    # Change pitch
    pitch_factor = np.random.uniform(-2, 2)
    audio_pitch = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_factor)
    augmented_audio.append(audio_pitch)

    # Time stretch
    stretch_factor = np.random.uniform(0.8, 1.2)
    audio_stretch = librosa.effects.time_stretch(audio, rate=stretch_factor)
    augmented_audio.append(audio_stretch)

    # Dynamic Range Compression
    audio_drc = librosa.effects.percussive(audio)
    augmented_audio.append(audio_drc)

    # Random Erasing (Zeroing out a part of the audio)
    erase_length = np.random.randint(0, int(len(audio) * 0.1))
    erase_start = np.random.randint(0, len(audio) - erase_length)
    audio_erased = audio.copy()
    audio_erased[erase_start:erase_start + erase_length] = 0
    augmented_audio.append(audio_erased)

    return augmented_audio


# First pass to determine max_length after augmentation
def first_pass(annotations):
    all_lengths = []

    for index, row in annotations.iterrows():
        file_name = f"{data_dir}/scenes/wav/{row['filename']}.wav"
        start_time = row['start']
        end_time = row['end']

        # Load audio file
        audio, sample_rate = librosa.load(file_name, sr=None)

        # Extract the relevant segment
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        audio_segment = audio[start_sample:end_sample]

        # Normalize the audio segment
        audio_segment = librosa.util.normalize(audio_segment)

        # Apply data augmentation
        augmented_segments = augment_audio(audio_segment, sample_rate)

        for segment in augmented_segments:
            all_lengths.append(len(segment))

    return all_lengths


# Determine the maximum length after augmentation
lengths = first_pass(annotations)
max_length = int(np.percentile(lengths, 95))
print(f"Max length determined statistically: {max_length}")


# Second pass to preprocess and pad/crop audio segments
def second_pass(annotations, max_length):
    scaler = StandardScaler()
    ica = FastICA(n_components=1, whiten='unit-variance')
    audio_data = []
    labels = []

    for index, row in annotations.iterrows():
        file_name = f"{data_dir}/scenes/wav/{row['filename']}.wav"
        start_time = row['start']
        end_time = row['end']

        # Load audio file
        audio, sample_rate = librosa.load(file_name, sr=None)

        # Extract the relevant segment
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        audio_segment = audio[start_sample:end_sample]

        # Normalize the audio segment
        audio_segment = librosa.util.normalize(audio_segment)

        # Apply data augmentation
        augmented_segments = augment_audio(audio_segment, sample_rate)

        for segment in augmented_segments:
            # Scale and apply ICA
            segment_scaled = scaler.fit_transform(segment.reshape(-1, 1)).flatten()
            segment_scaled = check_and_handle_nans_infs(segment_scaled)
            segment_ica = ica.fit_transform(segment_scaled.reshape(-1, 1)).flatten()
            segment_ica = check_and_handle_nans_infs(segment_ica)

            # Pad/crop to max_length
            if len(segment_ica) > max_length:
                segment_ica = segment_ica[:max_length]
            else:
                segment_ica = np.pad(segment_ica, (0, max_length - len(segment_ica)), 'constant')

            audio_data.append(segment_ica)
            labels.append(row['command'])

    return np.array(audio_data), labels


# Load and preprocess the audio data with padding/cropping
X, y = second_pass(annotations, max_length)

# Reshape the ICA output for Conv1D
X_ica_reshaped = X.reshape(X.shape[0], X.shape[1], 1)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the class names
command_class_names = list(label_encoder.classes_)
with open(f'{data_dir}/command_class_names.json', 'w') as f:
    json.dump(command_class_names, f)

# Convert labels to categorical
num_classes = len(np.unique(y_encoded))
y_categorical = to_categorical(y_encoded, num_classes)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_ica_reshaped, y_categorical, test_size=0.2, random_state=42)


# Define the Conv1D model
def create_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=8, kernel_size=13, padding='valid', activation='relu', strides=1),
        MaxPooling1D(pool_size=3),
        Dropout(0.3),
        Conv1D(filters=16, kernel_size=11, padding='valid', activation='relu', strides=1),
        MaxPooling1D(pool_size=3),
        Dropout(0.3),
        Conv1D(filters=32, kernel_size=9, padding='valid', activation='relu', strides=1),
        MaxPooling1D(pool_size=3),
        Dropout(0.3),
        Conv1D(filters=64, kernel_size=7, padding='valid', activation='relu', strides=1),
        MaxPooling1D(pool_size=3),
        Dropout(0.3),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model


# Create the model
model = create_model((X_train.shape[1], 1), num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_val_classes = np.argmax(y_val, axis=1)

# Classification report
print("Conv1D Neural Network Classifier Report")
print(classification_report(y_val_classes, y_pred_classes, zero_division=0))

# Confusion matrix
conf_matrix = confusion_matrix(y_val_classes, y_pred_classes)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Learning curves
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.axvline(np.argmin(history.history['val_loss']), color='r', linestyle='--', label='Early Stopping Checkpoint')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.show()

# Load the previous best model and evaluate its performance
if os.path.exists(model_save_path):
    previous_model = load_model(model_save_path)
    previous_val_loss, previous_val_accuracy = previous_model.evaluate(X_val, y_val)
    print(f"Previous model validation accuracy: {previous_val_accuracy}")
else:
    previous_val_accuracy = 0
    print("No previous model found. Saving current model as the best model.")

# Compare the performance of the current model with the previous best model
current_val_loss, current_val_accuracy = model.evaluate(X_val, y_val)
print(f"Current model validation accuracy: {current_val_accuracy}")

if current_val_accuracy > previous_val_accuracy:
    print("Current model outperforms the previous model. Saving the current model.")
    model.save(model_save_path)
else:
    print("Previous model outperforms the current model. Keeping the previous model.")
##############################################
stop = time.time()
print(f'Finished in {stop - start} secs')