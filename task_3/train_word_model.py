import time
start = time.time()
########################

import os
import numpy as np

# Function to load preprocessed data
def load_preprocessed_data(preprocessed_dir):
    X = []
    y = []

    for file_name in os.listdir(preprocessed_dir):
        if file_name.startswith('audio'):
            audio = np.load(os.path.join(preprocessed_dir, file_name))
            label_file = file_name.replace('audio', 'label')
            label = np.load(os.path.join(preprocessed_dir, label_file))

            X.append(audio)
            y.append(label)

    return np.array(X), np.array(y)

# Load preprocessed data
preprocessed_dir = 'preprocessed_data'
X, y = load_preprocessed_data(preprocessed_dir)

print(f'Loaded {len(X)} preprocessed audio files.')
print(f'Labels: {set(y)}')

# Split data into training, validation, and test sets
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f'Training set size: {len(X_train)}')
print(f'Validation set size: {len(X_val)}')
print(f'Test set size: {len(X_test)}')

import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

data_dir = '../dataset'
# Assuming the data is already loaded into X_train, X_val, X_test, y_train, y_val, y_test

# Reshape data for 1D CNN
X_train_cnn = X_train.reshape(-1, X_train.shape[1], 1)
X_val_cnn = X_val.reshape(-1, X_val.shape[1], 1)
X_test_cnn = X_test.reshape(-1, X_test.shape[1], 1)

# Encode labels as numeric
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

# Save the class names
word_class_names = list(label_encoder.classes_)
with open(f'{data_dir}/word_class_names.json', 'w') as f:
    json.dump(word_class_names, f)

# Convert labels to categorical
num_classes = len(np.unique(y_train_encoded))
y_train_cnn = to_categorical(y_train_encoded, num_classes)
y_val_cnn = to_categorical(y_val_encoded, num_classes)
y_test_cnn = to_categorical(y_test_encoded, num_classes)

# Define the modified model
model_v2 = Sequential([
    # First Conv1D layer
    Conv1D(filters=8, kernel_size=13, padding='valid', activation='relu', strides=1, input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=3),
    Dropout(0.3),

    # Second Conv1D layer
    Conv1D(filters=16, kernel_size=11, padding='valid', activation='relu', strides=1),
    MaxPooling1D(pool_size=3),
    Dropout(0.3),

    # Third Conv1D layer
    Conv1D(filters=32, kernel_size=9, padding='valid', activation='relu', strides=1),
    MaxPooling1D(pool_size=3),
    Dropout(0.3),

    # Fourth Conv1D layer
    Conv1D(filters=64, kernel_size=7, padding='valid', activation='relu', strides=1),
    MaxPooling1D(pool_size=3),
    Dropout(0.3),

    # Flatten layer
    Flatten(),

    # Dense Layer 1
    Dense(256, activation='relu'),
    Dropout(0.3),

    # Dense Layer 2
    Dense(128, activation='relu'),
    Dropout(0.3),

    # Output layer
    Dense(num_classes, activation='softmax')
])

# Compile the model
model_v2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history_v2 = model_v2.fit(X_train_cnn, y_train_cnn, validation_data=(X_val_cnn, y_val_cnn), epochs=20, batch_size=32)

# Evaluate the model on the validation set
val_loss_v2, val_accuracy_v2 = model_v2.evaluate(X_val_cnn, y_val_cnn)
print(f'Modified 1D CNN Validation Accuracy: {val_accuracy_v2}')
print(f'Modified 1D CNN Validation Loss: {val_loss_v2}')

# Evaluate the model on the test set
test_loss_v2, test_accuracy_v2 = model_v2.evaluate(X_test_cnn, y_test_cnn)
print(f'Modified 1D CNN Test Accuracy: {test_accuracy_v2}')
print(f'Modified 1D CNN Test Loss: {test_loss_v2}')

# Save the model if performing better
model_save_path = f'{data_dir}/word_model.h5'

# Evaluate the model
y_pred = model_v2.predict(X_val_cnn)
y_pred_classes = np.argmax(y_pred, axis=1)
y_val_classes = np.argmax(y_val_cnn, axis=1)

# Classification report
print("Conv1D Neural Network Classifier Report")
print(classification_report(y_val_classes, y_pred_classes, zero_division=0))

# Confusion matrix
conf_matrix = confusion_matrix(y_val_classes, y_pred_classes)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Learning curves
plt.figure(figsize=(12, 6))
plt.plot(history_v2.history['loss'], label='Training Loss')
plt.plot(history_v2.history['val_loss'], label='Validation Loss')
plt.plot(history_v2.history['accuracy'], label='Training Accuracy')
plt.plot(history_v2.history['val_accuracy'], label='Validation Accuracy')
plt.axvline(np.argmin(history_v2.history['val_loss']), color='r', linestyle='--', label='Early Stopping Checkpoint')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.show()

# Load the previous best model and evaluate its performance
if os.path.exists(model_save_path):
    previous_model = load_model(model_save_path)
    previous_val_loss, previous_val_accuracy = previous_model.evaluate(X_val_cnn, y_val_cnn)
    print(f"Previous model validation accuracy: {previous_val_accuracy}")
else:
    previous_val_accuracy = 0
    print("No previous model found. Saving current model as the best model.")

# Compare the performance of the current model with the previous best model
current_val_loss, current_val_accuracy = model_v2.evaluate(X_val_cnn, y_val_cnn)
print(f"Current model validation accuracy: {current_val_accuracy}")

if current_val_accuracy > previous_val_accuracy:
    print("Current model outperforms the previous model. Saving the current model.")
    model_v2.save(model_save_path)
else:
    print("Previous model outperforms the current model. Keeping the previous model.")

#####################
stop = time.time()
print(f'Finished in {stop - start} secs')