import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_dir = '../dataset'
precompiled_feature_dir = f'{data_dir}/scenes/npy'
extracted_feature_dir = f'{data_dir}/scenes/extracted_features'
meta_dir = f'{data_dir}/meta'


# Load the precompiled features
def load_features(feature_dir):
    feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
    features = []
    max_len = 0
    skipped_files = []

    # First pass to find the maximum length
    for f in tqdm(feature_files, desc="Finding max length"):
        feature = np.load(os.path.join(feature_dir, f))
        if feature.ndim == 2 and feature.shape[1] > max_len:
            max_len = feature.shape[1]

    logging.info(f"Maximum feature length: {max_len}")

    # Second pass to load and pad features
    for f in tqdm(feature_files, desc="Loading and padding features"):
        feature = np.load(os.path.join(feature_dir, f))
        if feature.ndim == 2:
            pad_width = max_len - feature.shape[1]
            if pad_width > 0:
                feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
            features.append(feature)
        else:
            logging.warning(f"Feature file {f} has unexpected shape {feature.shape} and will be skipped.")
            skipped_files.append(f)

    if len(features) == 0:
        raise ValueError("No valid features loaded. Please check the feature files.")

    logging.info(f"Number of valid feature files: {len(features)}")
    logging.info(f"Number of skipped feature files: {len(skipped_files)}")

    return np.array(features), feature_files


logging.info("Loading precompiled features...")
precompiled_features, precompiled_files = load_features(precompiled_feature_dir)

logging.info("Loading extracted features...")
extracted_features, extracted_files = load_features(extracted_feature_dir)

# Ensure features are three-dimensional
if precompiled_features.ndim != 3 or extracted_features.ndim != 3:
    raise ValueError("Loaded features do not have the expected dimensions. Please check the feature loading process.")

# Load feature names
precompiled_feature_names_file = os.path.join(meta_dir, 'idx_to_feature_name.csv')
extracted_feature_names_file = os.path.join(meta_dir, 'idx_to_extracted_feature_names.csv')

precompiled_feature_names = pd.read_csv(precompiled_feature_names_file)['feature_name'].tolist()
extracted_feature_names = pd.read_csv(extracted_feature_names_file)['feature_name'].tolist()

# Find common features
common_features = set(precompiled_feature_names).intersection(set(extracted_feature_names))
common_indices_precompiled = [precompiled_feature_names.index(f) for f in common_features]
common_indices_extracted = [extracted_feature_names.index(f) for f in common_features]

logging.info(f"Number of common features: {len(common_features)}")

# Filter features to only include common features
precompiled_features_common = precompiled_features[:, :, common_indices_precompiled]
extracted_features_common = extracted_features[:, :, common_indices_extracted]

# Calculate means and standard deviations for common features
precompiled_mean = np.mean(precompiled_features_common, axis=0)
precompiled_std = np.std(precompiled_features_common, axis=0)

extracted_mean = np.mean(extracted_features_common, axis=0)
extracted_std = np.std(extracted_features_common, axis=0)

# Compare means and standard deviations
mean_diff = np.abs(precompiled_mean - extracted_mean)
std_diff = np.abs(precompiled_std - extracted_std)

# Set a threshold for significant differences
mean_diff_threshold = 1e-5
std_diff_threshold = 1e-5

significant_mean_diff = mean_diff > mean_diff_threshold
significant_std_diff = std_diff > std_diff_threshold

# Report differences
num_significant_mean_diff = np.sum(significant_mean_diff)
num_significant_std_diff = np.sum(significant_std_diff)

logging.info(f"Number of significant mean differences: {num_significant_mean_diff}")
logging.info(f"Number of significant std differences: {num_significant_std_diff}")

# Save comparison results
comparison_results = {
    'precompiled_mean': precompiled_mean,
    'precompiled_std': precompiled_std,
    'extracted_mean': extracted_mean,
    'extracted_std': extracted_std,
    'mean_diff': mean_diff,
    'std_diff': std_diff,
    'significant_mean_diff': significant_mean_diff,
    'significant_std_diff': significant_std_diff
}

comparison_results_file = os.path.join(meta_dir, 'feature_comparison_results.npz')
np.savez(comparison_results_file, **comparison_results)

logging.info(f"Comparison results saved to {comparison_results_file}")

# Plotting the differences
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(mean_diff.flatten(), label='Mean Difference', color='blue')
plt.axhline(y=mean_diff_threshold, color='red', linestyle='--', label='Threshold')
plt.title('Mean Differences Between Precompiled and Extracted Features')
plt.xlabel('Feature Index')
plt.ylabel('Mean Difference')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(std_diff.flatten(), label='Std Difference', color='green')
plt.axhline(y=std_diff_threshold, color='red', linestyle='--', label='Threshold')
plt.title('Standard Deviation Differences Between Precompiled and Extracted Features')
plt.xlabel('Feature Index')
plt.ylabel('Std Difference')
plt.legend()

plt.tight_layout()
plot_file = os.path.join(meta_dir, 'feature_differences.png')
plt.savefig(plot_file)
plt.show()

logging.info(f"Plots saved to {plot_file}")
