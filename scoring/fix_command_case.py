import pandas as pd

# Define file paths
predictions_file_path = 'predictions.csv'
corrected_predictions_file_path = 'chg_predictions.csv'

# Load the predictions file
predictions_df = pd.read_csv(predictions_file_path)

# Function to capitalize the first word of each command
def capitalize_first_word(command):
    words = command.split()
    if len(words) > 0:
        words[0] = words[0].capitalize()
    return ' '.join(words)

# Apply the function to the command column
predictions_df['command'] = predictions_df['command'].apply(capitalize_first_word)

# Save the corrected predictions to CSV
predictions_df.to_csv(corrected_predictions_file_path, index=False)

print(f"Corrected predictions saved to {corrected_predictions_file_path}")
