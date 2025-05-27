import glob
import gc
from collections import OrderedDict
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Dataset configuration
DATASET_DIR = '../../Datasets/road/signal_extractions/CANet'
CSV_FILE = "../../Datasets/road/signal_extractions/ambient/ambient_dyno_drive_basic_long.csv"

# Load your dataset to analyze structure
print("Loading and analyzing dataset structure...")
df = pd.read_csv(CSV_FILE)

# Fix column names - convert from Signal_X_of_ID to SignalX format
print("Original columns:", df.columns.tolist())
column_mapping = {}
for col in df.columns:
    if col.startswith('Signal_') and col.endswith('_of_ID'):
        # Extract signal number from Signal_X_of_ID
        signal_num = col.split('_')[1]
        new_name = f'Signal{signal_num}'
        column_mapping[col] = new_name

# Apply column renaming
df.rename(columns=column_mapping, inplace=True)
print("Renamed columns:", df.columns.tolist())

# Data file paths
data_files = {
    'train': glob.glob(f'{DATASET_DIR}/train/*.csv') + glob.glob(f'{DATASET_DIR}/train/*.parquet'),
    'valid': glob.glob(f'{DATASET_DIR}/val/*.csv') + glob.glob(f'{DATASET_DIR}/val/*.parquet'),
    'test': glob.glob('../../Datasets/road/signal_extractions/attack/*.csv') + glob.glob('../../Datasets/road/signal_extractions/attack/*.parquet')
}

# List of signal columns (everything after column 2, excluding Label, Time, ID)
signal_columns = [col for col in df.columns if col.startswith('Signal')]
print(f"Found {len(signal_columns)} signal columns: {signal_columns}")

WINDOW_SIZE = 1
TRAIN_START = WINDOW_SIZE + 1

# Initialize dictionaries
ID_NSIG = OrderedDict()
ID_MPS = {}

# Group by CAN ID
print("Analyzing CAN IDs...")
grouped = df.groupby("ID")

for can_id, group in grouped:
    # Count how many rows this CAN ID has → MPS (message per sequence)
    ID_MPS[str(can_id)] = len(group)
    
    # For NSIG: find the max number of non-null signal columns used in any row of this ID
    signals_used = group[signal_columns].notnull().sum(axis=1)
    ID_NSIG[str(can_id)] = int(signals_used.max())

# Calculate frequency
start_time = df["Time"].min()
end_time = df["Time"].max()
duration_sec = end_time - start_time
ID_MPS = {id_: round(count / duration_sec) for id_, count in ID_MPS.items()}
ID_MPS = dict(sorted(ID_MPS.items(), key=lambda x: x[1], reverse=True))

print("ID_FREQ =", ID_MPS)
print("ID_NSIG =", dict(ID_NSIG))

# Filter IDs by frequency threshold
ID_FREQ = {k: v for k, v in ID_MPS.items() if v >= 40}
ID_NSIG = {k: ID_NSIG[k] for k in ID_FREQ.keys()}

print(f"Filtered to {len(ID_FREQ)} IDs with frequency >= 40")
print("Final ID_FREQ =", ID_FREQ)
print("Final ID_NSIG =", dict(ID_NSIG))

def load_arrange_data(file_path, print_option=False):
    """Load and arrange data from CSV or parquet file"""
    file_path = Path(file_path)
    
    # Load data based on file extension
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Step 1: Convert time from microseconds to seconds if needed
    df['Time'] = round(df['Time'], 7)

    # Step 2: Rename 'Label' to 'Session' if Label column exists
    if 'Label' in df.columns:
        df.rename(columns={'Label': 'Session'}, inplace=True)

    # Step 3: Standardize signal column names
    column_mapping = {}
    for col in df.columns:
        if col.startswith('Signal_') and col.endswith('_of_ID'):
            # Extract signal number from Signal_X_of_ID
            signal_num = col.split('_')[1]
            new_name = f'Signal{signal_num}'
            column_mapping[col] = new_name
    
    # Apply column renaming
    if column_mapping:
        df.rename(columns=column_mapping, inplace=True)

    # Step 4: Drop trailing empty signal columns
    df = df.dropna(axis=1, how='all')

    if print_option:
        print(f'# rows: {df.shape[0]:,}')
        if 'Session' in df.columns:
            print(df['Session'].value_counts())
        
    return df

def str_to_list(data_str: str) -> list:
    """Convert string data to list of integers"""
    data_list_str = data_str.split()
    data_list = [int(x) for x in data_list_str]
    if len(data_list) < 8:  # fill with dummy values (0) to 8 bytes
        data_list += [0] * (8 - len(data_list))
    return data_list

def get_repeated_sequences(data, can_id, n_sig, n_step):
    """Get repeated sequences for a specific CAN ID"""
    sig_columns = [f'Signal{i}' for i in range(1, n_sig + 1)]
    
    # Filter data for specific CAN ID
    df_id = data.loc[data['ID'] == can_id, ['Idx'] + sig_columns].copy()
    
    if df_id.empty:
        # Return empty array with correct shape if no data for this ID
        return np.empty((0, n_step, n_sig))
    
    # Fill NaN values with 0
    df_id[sig_columns] = df_id[sig_columns].fillna(0)
    
    np_sig = df_id[sig_columns].to_numpy()
    
    if len(np_sig) < n_step:
        # If we don't have enough data for even one window, pad with zeros
        padded_sig = np.zeros((n_step, n_sig))
        padded_sig[:len(np_sig)] = np_sig
        return padded_sig.reshape(1, n_step, n_sig)
    
    np_seq = np.lib.stride_tricks.sliding_window_view(np_sig, window_shape=n_step, axis=0)
    np_seq = np_seq.swapaxes(1, 2)
    n_seq = np_seq.shape[0]
    
    end_idx = data['Idx'].iloc[-1]
    idx_list = df_id['Idx'].to_list()
    n_repeats = np.diff(idx_list + [end_idx])[-n_seq:]
    
    repeated_seq = np.repeat(np_seq, n_repeats, axis=0)
    return repeated_seq

def prepare_dataset(file_path, time_cutoff, label='last', print_option=True):
    """Prepare dataset for training/validation"""
    assert label in ['last', 'sum', False]
    
    data = load_arrange_data(file_path, print_option=False)
    data = data.reset_index(names='Idx')
    
    time_start = data['Time'].iloc[0]
    cutoff_mask = data['Time'] > time_start + time_cutoff
    n_rows_to_use = cutoff_mask.sum()
    
    if n_rows_to_use == 0:
        print(f"Warning: No data after time cutoff {time_cutoff}")
        return {}, np.array([])
    
    if label == 'last':
        time_and_labels = data.loc[cutoff_mask, ['Time', 'Session']].to_numpy()
    
    data_dict = dict()
    for id_str, nsig in ID_NSIG.items():
        mps = ID_MPS[id_str]  # messages per second (window size)
        seq_data = get_repeated_sequences(data, id_str, nsig, mps)
        
        if len(seq_data) >= n_rows_to_use:
            data_dict[id_str] = seq_data[-n_rows_to_use:].copy()
        else:
            # Pad with zeros if we don't have enough data
            padded_data = np.zeros((n_rows_to_use, mps, nsig))
            if len(seq_data) > 0:
                padded_data[:len(seq_data)] = seq_data
            data_dict[id_str] = padded_data
        
        if print_option:
            print(f"ID {id_str}: {data_dict[id_str].shape}")
    
    if label and 'time_and_labels' in locals():
        return data_dict, time_and_labels
    else:
        return data_dict

def slice_data(file_path: str, n_sliced: int):
    """Slice large data files into smaller chunks"""
    file_path = Path(file_path)
    
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        raise Exception(f'File type {file_path.suffix} not supported')
    
    p = len(df) // n_sliced
    split_indices = list(range(0, len(df) + 1, p))[:-1]
    paths = []
    
    for i in range(n_sliced):
        if i + 1 < n_sliced:
            sliced_df = df.iloc[split_indices[i]:split_indices[i+1]]
        else:  # the last slice
            sliced_df = df.iloc[split_indices[i]:]
        
        cache_dir = file_path.parent / 'cache'
        cache_dir.mkdir(exist_ok=True)
        save_path = cache_dir / f'{file_path.stem}_{i+1}.parquet'
        sliced_df.to_parquet(save_path)
        paths.append(str(save_path))
    
    return paths

def load_inputs(data_path, time_cutoff, label, shuffle=True, seed=0):
    """Load inputs for training/validation"""
    if not label:
        x_dict = prepare_dataset(data_path, time_cutoff=time_cutoff, label=False, print_option=False)
        x_time_label = None
    else:
        x_dict, x_time_label = prepare_dataset(data_path, time_cutoff=time_cutoff, label=label, print_option=False)
    
    if not x_dict:  # Empty dictionary
        return {}, np.array([]), x_time_label
    
    if shuffle:
        np.random.seed(seed)
        n_samples = len(x_dict[list(x_dict.keys())[0]])
        shuffled_idx = np.arange(n_samples)
        np.random.shuffle(shuffled_idx)
        x_dict = {id_str: seqs[shuffled_idx] for id_str, seqs in x_dict.items()}
        if label and x_time_label is not None:
            x_time_label = x_time_label[shuffled_idx]
    
    # Create y by concatenating last timestep of all IDs
    y_parts = []
    for id_str in FIXED_IDS:
        if id_str in x_dict:
            y_parts.append(x_dict[id_str][:, -1, :])
        else:
            # If ID not in data, create zeros
            n_samples = len(x_dict[list(x_dict.keys())[0]]) if x_dict else 0
            y_parts.append(np.zeros((n_samples, ID_NSIG[id_str])))
    
    if y_parts:
        y = np.concatenate(y_parts, axis=1)
    else:
        y = np.array([])
    
    if not label:
        return x_dict, y
    else:
        return x_dict, y, x_time_label

# Calculate constants
N_SIGS = sum([n_sig for n_sig in ID_NSIG.values()])
FIXED_IDS = list(ID_NSIG.keys())
FIXED_IDS.sort()

print(f"Total number of signals: {N_SIGS}")
print(f"Fixed IDs: {FIXED_IDS}")

def create_canet_model(h):
    """Create CANet model architecture"""
    inputs = {id_str: keras.Input(shape=(ID_MPS[id_str], ID_NSIG[id_str]), name=id_str) for id_str in FIXED_IDS}
    lstms = [keras.layers.LSTM(h * ID_NSIG[id_str], name=f'lstm_{id_str}') for id_str in FIXED_IDS]
    x_id = [lstms[i](inputs[id_str]) for i, id_str in enumerate(FIXED_IDS)]
    x = keras.layers.Concatenate()(x_id)
    x = keras.layers.Dense((h * N_SIGS) // 2, activation='elu')(x)
    x = keras.layers.Dense(N_SIGS - 1, activation='elu')(x)
    outputs = keras.layers.Dense(N_SIGS, activation='elu')(x)
    return keras.Model(inputs, outputs)

# Create and compile model
print("Creating model...")
h = 5
lr = 0.0001  # learning rate
model = create_canet_model(h)
optimizer = keras.optimizers.Adam(learning_rate=lr)
model.compile(
    optimizer=optimizer,
    loss='mean_squared_error'
)

print(f"Model created with {model.count_params():,} parameters")

# Training configuration
batch_size = 250
n_epoch = 10

# Create models directory
Path('models/CANET').mkdir(parents=True, exist_ok=True)

starttime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
print(f"Starting training at {starttime}")

for epoch in range(n_epoch):
    print(f'********** Epoch {epoch+1}/{n_epoch} **********')
    
    epoch_train_losses = []
    epoch_val_losses = []
    
    # Process each training file with corresponding validation
    for i, train_file in enumerate(data_files['train']):
        print(f"\n--- Processing file pair {i+1}/{len(data_files['train'])} ---")
        
        # Check if training file exists
        if not Path(train_file).exists():
            print(f"Warning: Training file {train_file} not found, skipping...")
            continue
        
        # Get corresponding validation file
        if i < len(data_files['valid']):
            val_file = data_files['valid'][i]
        else:
            print(f"Warning: No corresponding validation file for {train_file}, skipping validation for this file")
            val_file = None
        
        try:
            # TRAINING PHASE for current file
            print(f"Training with {train_file}...")
            sliced_files = slice_data(train_file, 10)
            
            file_train_losses = []
            for sliced_file in sliced_files:
                x_train_dict, y_train = load_inputs(sliced_file, time_cutoff=TRAIN_START, label=False, shuffle=True, seed=epoch)
                
                if len(y_train) == 0:
                    print(f"No training data in {sliced_file}, skipping...")
                    continue
                
                print(f'Training with slice: {sliced_file} {y_train.shape}')
                
                with tf.device('CPU'):
                    train_ds = tf.data.Dataset.from_tensor_slices((x_train_dict, y_train))
                train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
                
                # Train and capture loss
                history = model.fit(train_ds, verbose=2)
                if hasattr(history.history, 'loss') and history.history['loss']:
                    file_train_losses.append(history.history['loss'][-1])
                
                del x_train_dict, y_train, train_ds
                gc.collect()
            
            # Calculate average training loss for this file
            if file_train_losses:
                avg_file_train_loss = sum(file_train_losses) / len(file_train_losses)
                epoch_train_losses.append(avg_file_train_loss)
                print(f"Average training loss for {train_file}: {avg_file_train_loss:.6f}")
            
        except Exception as e:
            print(f"Error processing training file {train_file}: {e}")
            continue
        
        # VALIDATION PHASE for corresponding file
        if val_file and Path(val_file).exists():
            try:
                print(f"Validating with {val_file}...")
                sliced_val_files = slice_data(val_file, 10)
                
                file_val_losses = []
                for sliced_val_file in sliced_val_files:
                    x_valid_dict, y_valid = load_inputs(sliced_val_file, time_cutoff=TRAIN_START, label=False, shuffle=False)
                    
                    if len(y_valid) == 0:
                        print(f"No validation data in {sliced_val_file}, skipping...")
                        continue
                    
                    print(f'Validating with slice: {sliced_val_file} {y_valid.shape}')
                    
                    with tf.device('CPU'):
                        valid_ds = tf.data.Dataset.from_tensor_slices((x_valid_dict, y_valid))
                    valid_ds = valid_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
                    
                    # Evaluate and capture loss
                    batch_loss = model.evaluate(valid_ds, verbose=2)
                    file_val_losses.append(batch_loss)
                    
                    del x_valid_dict, y_valid, valid_ds
                    gc.collect()
                # Calculate average validation loss for this file
                if file_val_losses:
                    avg_file_val_loss = sum(file_val_losses) / len(file_val_losses)
                    epoch_val_losses.append(avg_file_val_loss)
                    print(f"Average validation loss for {val_file}: {avg_file_val_loss:.6f}")
            except Exception as e:
                print(f"Error processing validation file {val_file}: {e}")
                
        elif val_file:
            print(f"Warning: Validation file {val_file} not found")
        try:
            weight_name = f'models/CANET/road_{starttime}_epoch{epoch+1:02d}'
            model.save_weights(weight_name)
            print(f"✓ Model weights saved to {weight_name}")
        except Exception as e:
            print(f"Error saving model weights: {e}")
    # Print epoch summary
    print(f"\n--- Epoch {epoch+1} Summary ---")
    if epoch_train_losses:
        avg_epoch_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        print(f"Average training loss for epoch: {avg_epoch_train_loss:.6f}")
    else:
        print("No training data processed in this epoch")
    
    if epoch_val_losses:
        avg_epoch_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        print(f"Average validation loss for epoch: {avg_epoch_val_loss:.6f}")
    else:
        print("No validation data processed in this epoch")
    
    # Save model weights after each epoch

print("Training completed!")