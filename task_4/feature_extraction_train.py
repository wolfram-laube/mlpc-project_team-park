import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import json
from utils import augment_data
from models import KolmogorovArnoldNetwork

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_dir = '../dataset'
annotations_file = f'{data_dir}/development_scene_annotations.csv'
kan_model_save_path = 'best_command_kan_model.keras'
feature_dir = f'{data_dir}/scenes/extracted_features'
meta_dir = f'{data_dir}/meta'
command_mapping_path = os.path.join(meta_dir, 'command_mapping.json')

# Load annotations
logging.info('Loading annotations...')
annotations = pd.read_csv(annotations_file)
logging.info('Annotations loaded.')

# Load mean and std
mean = np.load(os.path.join(meta_dir, 'mean.npy'))
std = np.load(os.path.join(meta_dir, 'std.npy'))

# Check class distribution
class_counts = annotations['command'].value_counts()
plt.figure(figsize=(12, 6))
class_counts.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Commands')
plt.ylabel('Count')
plt.show()

def prepare_feature_data(annotations, feature_dir):
    command_features = []
    command_labels = []
    command_mapping = {}  # Mapping of command texts to numerical labels
    current_label = 0
    max_len = 0  # To determine the maximum length of features

    logging.info('Preparing command data...')
    for index, row in tqdm(annotations.iterrows(), total=annotations.shape[0]):
        feature_path = os.path.join(feature_dir, row['filename'] + '.npy')
        features = np.load(feature_path)

        if features.ndim == 1:  # Handling case where features are not properly loaded
            logging.warning(f"Feature file {row['filename']} has unexpected shape {features.shape} and will be skipped.")
            continue

        max_len = max(max_len, features.shape[2])  # Update max_len

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
        pad_width = max_len - feature.shape[2]
        if pad_width > 0:
            feature = np.pad(feature, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        padded_features.append(feature)

    logging.info('Command data prepared.')
    return np.array(padded_features), np.array(command_labels), command_mapping, max_len

# Prepare feature-based command data
command_features, command_labels, command_mapping, max_len = prepare_feature_data(annotations, feature_dir)

# Save command mapping
with open(command_mapping_path, 'w') as f:
    json.dump(command_mapping, f)

# Normalize features across each feature dimension
command_features = (command_features - mean) / std

# One-hot encode labels
num_classes = len(command_mapping)
command_labels = to_categorical(command_labels, num_classes=num_classes)

logging.info(f'Command mapping: {command_mapping}')

# Data Augmentation Function
augmented_features = augment_data(command_features)
combined_features = np.concatenate((command_features, augmented_features), axis=0)
combined_labels = np.concatenate((command_labels, command_labels), axis=0)

# Calculate class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(np.argmax(command_labels, axis=1)),
                                                  y=np.argmax(command_labels, axis=1))
class_weights = dict(enumerate(class_weights))

# Adjust input data shape
combined_features = np.squeeze(combined_features, axis=1)
print(f'Adjusted combined_features shape: {combined_features.shape}')

input_shape = (combined_features.shape[1], combined_features.shape[2])
num_univariate_functions = 5
hidden_units = 128

kan_model = KolmogorovArnoldNetwork(input_shape, num_univariate_functions, hidden_units, num_classes)

kan_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

logging.info('Training command recognition model...')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(kan_model_save_path, save_best_only=True, monitor='val_loss')

history = kan_model.fit(combined_features, combined_labels, epochs=200, batch_size=48, validation_split=0.2,
                        callbacks=[early_stopping, model_checkpoint], class_weight=class_weights)
logging.info('Command recognition model trained.')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
