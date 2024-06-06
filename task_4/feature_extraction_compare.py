import os
import numpy as np
import pandas as pd

data_dir = '../dataset'
extracted_dir = f'{data_dir}/scenes/extracted_features'
precompiled_dir = f'{data_dir}/scenes/npy'
idx_to_feature_name_file = f'{data_dir}/meta/idx_to_feature_name.csv'
extracted_feature_names_file = os.path.join(extracted_dir, 'idx_to_extracted_feature_names.csv')

# Load feature names
precompiled_feature_names = pd.read_csv(idx_to_feature_name_file)
extracted_feature_names = pd.read_csv(extracted_feature_names_file)

# Ensure the feature names match
if not precompiled_feature_names['feature_name'].equals(extracted_feature_names['feature_name']):
    print("Warning: Feature names do not match exactly.")

# Calculate mean and stddev for the precompiled features
precompiled_files = [f for f in os.listdir(precompiled_dir) if f.endswith('.npy')]
precompiled_features = np.array([np.load(os.path.join(precompiled_dir, f)) for f in precompiled_files])

precompiled_mean = np.mean(precompiled_features, axis=0)
precompiled_std = np.std(precompiled_features, axis=0)

# Load mean and stddev for the extracted features
extracted_mean = np.load(os.path.join(extracted_dir, 'mean.npy'))
extracted_std = np.load(os.path.join(extracted_dir, 'std.npy'))

# Compare the means and stddevs
mean_diff = np.abs(extracted_mean - precompiled_mean)
std_diff = np.abs(extracted_std - precompiled_std)

print(f"Mean difference: {np.mean(mean_diff)}")
print(f"Standard deviation difference: {np.mean(std_diff)}")
