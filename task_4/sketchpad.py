data_dir = '../dataset'

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
import librosa

# Load the CSV file containing the annotations
annotations = pd.read_csv(f'{data_dir}/development_scene_annotations.csv')


# Load and preprocess the audio data
def load_and_preprocess_audio(annotations, max_length=22050):
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

        # Reshape the segment to have a fixed length
        if len(audio_segment) > max_length:
            audio_segment = audio_segment[:max_length]
        else:
            audio_segment = np.pad(audio_segment, (0, max_length - len(audio_segment)), 'constant')

        audio_data.append(audio_segment)
        labels.append(row['command'])

    return np.array(audio_data), labels


# Load and preprocess the audio data
X, y = load_and_preprocess_audio(annotations)

# Check the shape of the data
print(f"Shape of data after preprocessing: {X.shape}")

# Apply FastICA
ica = FastICA(n_components=1)
X_ica = ica.fit_transform(X)

# Check the shape of the data after ICA
print(f"Shape of data after ICA: {X_ica.shape}")

# Reshape the ICA output for Conv1D
X_ica_reshaped = X_ica.reshape(X_ica.shape[0], X_ica.shape[1], 1)

# Check the shape of the data after re-shaping
print(f"Shape of data after re-shaping: {X_ica_reshaped.shape}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert labels to categorical
num_classes = len(np.unique(y_encoded))
y_categorical = to_categorical(y_encoded, num_classes)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_ica_reshaped, y_categorical, test_size=0.2, random_state=42)

# Define the Conv1D model
model = Sequential([
    Input(shape=(X_train.shape[1], 1)),
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

from sklearn.metrics import classification_report

print("Conv1D Neural Network Classifier Report")
print(classification_report(y_val_classes, y_pred_classes, zero_division=0))
