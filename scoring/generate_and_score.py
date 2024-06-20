import pandas as pd
import subprocess

# Define file paths
annotations_file_path = 'development_scene_annotations.csv'
predictions_file_path = 'prf_predictions.csv'
scoring_script_path = 'score_predictions.py'

# Step 1: Load the annotations file
annotations_df = pd.read_csv(annotations_file_path)

# Step 2: Transform annotations to predictions with capitalized commands
def transform_annotations_to_predictions(annotations_df):
    predictions = []
    for index, row in annotations_df.iterrows():
        filename = row['filename']
        command = row['command'].capitalize()  # Capitalize the command
        timestamp = (row['start'] + row['end']) / 2  # Calculate the mean of start and end
        predictions.append({'filename': filename, 'command': command, 'timestamp': timestamp})
    return pd.DataFrame(predictions)

# Transform annotations to predictions with capitalized commands
predictions_df = transform_annotations_to_predictions(annotations_df)

# Save predictions to CSV
predictions_df.to_csv(predictions_file_path, index=False)

# Step 3: Run the scoring script
result = subprocess.run(
    ['python3', scoring_script_path, '--predictions', predictions_file_path, '--annotations', annotations_file_path],
    capture_output=True,
    text=True
)

# Print the output of the scoring script
print(result.stdout)
