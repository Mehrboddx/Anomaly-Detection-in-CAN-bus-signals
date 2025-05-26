import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Base path to your dataset
base_path = '../../Datasets/road/signal_extractions'

ambient_folder = os.path.join(base_path, 'ambient')
train_folder = os.path.join(base_path, 'CANet', 'train')
val_folder = os.path.join(base_path, 'CANet', 'val')

# Create folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

for filename in os.listdir(ambient_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(ambient_folder, filename)
        
        # Load the CSV data
        df = pd.read_csv(file_path)
        
        # Determine split index based on 80/20
        split_idx = int(0.8 * len(df))

        # Use first 80% for train, last 20% for validation
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        # Save the splits
        train_df.to_csv(os.path.join(train_folder, filename), index=False)
        val_df.to_csv(os.path.join(val_folder, filename), index=False)
        
        print(f'Processed {filename}: {len(train_df)} train rows, {len(val_df)} val rows')

