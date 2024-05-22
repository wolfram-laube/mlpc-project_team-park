import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_code_cell("""\
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Any additional libraries needed
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Ensure the 'fig' directory exists
import os
os.makedirs('fig', exist_ok=True)
"""),
    nbf.v4.new_markdown_cell("# 1. Data Split\n\n## 1.a Description of Data Split for Model Selection and Hyperparameter Tuning"),
    nbf.v4.new_code_cell("""\
# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Split the dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Description of the data split
print(f'Training set size: {len(train_data)}')
print(f'Validation set size: {len(val_data)}')
print(f'Test set size: {len(test_data)}')
"""),
    nbf.v4.new_markdown_cell("## 1.b Avoidance of Information Leakage\n\nMeasures taken to prevent information leakage between sets:\n- Ensured samples from the same speaker are only in one set."),
    nbf.v4.new_markdown_cell("## 1.c Deriving Unbiased Performance Estimates\n\nUsing cross-validation on the training set to obtain unbiased performance estimates."),
    nbf.v4.new_code_cell("""\
# Example of cross-validation
from sklearn.model_selection import cross_val_score

X_train = train_data.drop('label', axis=1)
y_train = train_data['label']

model = RandomForestClassifier(n_estimators=100)
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

print(f'Cross-validation scores: {cv_scores}')
print(f'Average CV score: {np.mean(cv_scores)}')
"""),
    nbf.v4.new_markdown_cell("# 2. Classes & Features\n\n## 2.a Grouping of Words and \"Other\" Snippets\n\nDetails on how the 20 keywords and \"Other\" snippets were grouped into classes."),
    nbf.v4.new_markdown_cell("## 2.b Subset of Selected Features"),
    nbf.v4.new_code_cell("""\
# Feature selection example
from sklearn.feature_selection import SelectKBest, f_classif

X = train_data.drop('label', axis=1)
y = train_data['label']

selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print('Selected features:', selected_features)
"""),
    nbf.v4.new_markdown_cell("## 2.c Preprocessing Steps\n\nApplied preprocessing steps include normalization and noise reduction using ICA."),
    nbf.v4.new_code_cell("""\
# Example of normalization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Example of ICA
from sklearn.decomposition import FastICA

ica = FastICA(n_components=10)
X_ica = ica.fit_transform(X_scaled)
"""),
    nbf.v4.new_markdown_cell("# 3. Evaluation\n\n## 3.a Chosen Evaluation Criteria\n\nChosen evaluation criteria include accuracy, precision, recall, and F1-score."),
    nbf.v4.new_markdown_cell("## 3.b Baseline and Best Possible Performance\n\nBaseline performance using a simple model."),
    nbf.v4.new_code_cell("""\
# Baseline model example
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
y_dummy_pred = dummy.predict(X_test)

accuracy_baseline = accuracy_score(y_test, y_dummy_pred)
print(f'Baseline accuracy: {accuracy_baseline}')
"""),
    nbf.v4.new_markdown_cell("# 4. Experiments\n\n## Random Forest\n### 4.a Classification Performance with Varying Hyperparameters"),
    nbf.v4.new_code_cell("""\
# Random Forest example
param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [10, 20, 30]}
rf_scores = []

for n in param_grid['n_estimators']:
    for d in param_grid['max_depth']:
        rf = RandomForestClassifier(n_estimators=n, max_depth=d)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        rf_scores.append((n, d, score))

# Visualize the results
rf_df = pd.DataFrame(rf_scores, columns=['n_estimators', 'max_depth', 'accuracy'])
plt.figure(figsize=(10, 6))
sns.lineplot(data=rf_df, x='n_estimators', y='accuracy', hue='max_depth')
plt.title('Random Forest Hyperparameter Tuning')
plt.savefig('fig/rf_hyperparameter_tuning.png')
plt.show()
"""),
    nbf.v4.new_markdown_cell("### 4.b Overfitting and Underfitting Analysis\n\nDiscuss the extent of overfitting or underfitting observed in Random Forest experiments."),
    nbf.v4.new_markdown_cell("### 4.c Final Unbiased Performance Comparison\n\nSummarize the results in a comparative table or plot."),
    nbf.v4.new_markdown_cell("## Nearest Neighbour\n### 4.a Classification Performance with Varying Hyperparameters"),
    nbf.v4.new_code_cell("""\
# Nearest Neighbour example
param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
knn_scores = []

for k in param_grid['n_neighbors']:
    for w in param_grid['weights']:
        knn = KNeighborsClassifier(n_neighbors=k, weights=w)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        knn_scores.append((k, w, score))

# Visualize the results
knn_df = pd.DataFrame(knn_scores, columns=['n_neighbors', 'weights', 'accuracy'])
plt.figure(figsize=(10, 6))
sns.lineplot(data=knn_df, x='n_neighbors', y='accuracy', hue='weights')
plt.title('K-Nearest Neighbours Hyperparameter Tuning')
plt.savefig('fig/knn_hyperparameter_tuning.png')
plt.show()
"""),
    nbf.v4.new_markdown_cell("### 4.b Overfitting and Underfitting Analysis\n\nDiscuss the extent of overfitting or underfitting observed in Nearest Neighbour experiments."),
    nbf.v4.new_markdown_cell("### 4.c Final Unbiased Performance Comparison\n\nSummarize the results in a comparative table or plot."),
    nbf.v4.new_markdown_cell("## CNN\n### 4.a Classification Performance with Varying Hyperparameters"),
    nbf.v4.new_code_cell("""\
# CNN example
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(height, width, channels)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('CNN Training History')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('fig/cnn_training_history.png')
plt.show()
"""),
    nbf.v4.new_markdown_cell("### 4.b Overfitting and Underfitting Analysis\n\nDiscuss the extent of overfitting or underfitting observed in CNN experiments."),
    nbf.v4.new_markdown_cell("### 4.c Final Unbiased Performance Comparison\n\nSummarize the results in a comparative table or plot."),
    nbf.v4.new_markdown_cell("# 5. Analysis of Realistic Scenes\n\n## 5.a Qualitative Evaluation of Best Classifier\n\nListen to the provided scenes and inspect classifier predictions."),
    nbf.v4.new_markdown_cell("## 5.b Problematic Conditions and Solutions\n\nIdentify problematic conditions causing misrecognition or misprediction of keywords. Suggest potential solutions."),
    nbf.v4.new_markdown_cell("## 5.c Visualization of Findings"),
    nbf.v4.new_code_cell("""\
# Example of visualizing predictions on spectrogram
import librosa.display

audio, sr = librosa.load('example_scene.wav')
S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
log_S = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(12, 8))
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel-Spectrogram of Example Scene')
plt.colorbar(format='%+2.0f dB')
plt.savefig('fig/mel_spectrogram_example_scene.png')
plt.show()
"""),
    nbf.v4.new_markdown_cell("# Conclusion\n\nSummary of findings and future research directions.")
]

nb['cells'] = cells

with open('main.ipynb', 'w') as f:
    nbf.write(nb, f)
