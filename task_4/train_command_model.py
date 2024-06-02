import os
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
import soundfile as sf
import glob
from tqdm import tqdm
import logging
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_dir = '../dataset'
preprocessed_data_dir = os.path.join(data_dir, 'preprocessed_audio')
model_save_path = os.path.join(data_dir, 'command_model.h5')
label_metadata_path = os.path.join(data_dir, 'command_class_names.json')


# Load preprocessed audio files
def load_preprocessed_data(preprocessed_data_dir):
    audio_data = []
    labels = []
    files = glob.glob(os.path.join(preprocessed_data_dir, '*.wav'))

    with tqdm(total=len(files), desc="Loading preprocessed data") as pbar:
        for file in files:
            label = file.split('_')[-1].split('.')[0]  # Extract label from filename
            audio, sample_rate = sf.read(file)
            audio_data.append(audio)
            labels.append(label)
            pbar.update(1)

    return audio_data, labels


# Function to pad or trim audio segments to a fixed length
def pad_or_trim(segment, target_length):
    if len(segment) > target_length:
        return segment[:target_length]
    elif len(segment) < target_length:
        return np.pad(segment, (0, target_length - len(segment)), mode='constant')
    else:
        return segment


# Load and preprocess the data
logging.info("Loading preprocessed audio data...")
X_list, y = load_preprocessed_data(preprocessed_data_dir)

# Determine the 95th percentile length from the preprocessed data
lengths = [len(x) for x in X_list]
max_length = int(np.percentile(lengths, 95))
logging.info(f"95th percentile length of preprocessed data: {max_length}")

# Pad or trim the data to the 95th percentile length
X = np.array([pad_or_trim(x, max_length) for x in X_list])

# Reshape the data for Conv1D
X = X.reshape(X.shape[0], X.shape[1], 1)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save class label metadata
with open(label_metadata_path, 'w') as f:
    json.dump(label_encoder.classes_.tolist(), f)

# Convert labels to categorical
num_classes = len(np.unique(y_encoded))
y_categorical = to_categorical(y_encoded, num_classes)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)


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
logging.info("Starting training...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_val_classes = np.argmax(y_val, axis=1)

# Classification report
logging.info("Conv1D Neural Network Classifier Report")
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

logging.info("Training completed.")
