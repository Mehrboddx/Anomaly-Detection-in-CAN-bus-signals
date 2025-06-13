import glob
import gc
from collections import OrderedDict
import datetime
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm.auto import tqdm
import h5py

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
    'test': glob.glob('../../Datasets/road/signal_extractions/attacks/*.csv') + glob.glob('../../Datasets/road/signal_extractions/attack/*.parquet')
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
ID_FREQ = {k: v for k, v in ID_MPS.items() if v >= 50}
ID_NSIG = {k: ID_NSIG[k] for k in ID_FREQ.keys()}

print(f"Filtered to {len(ID_FREQ)} IDs with frequency >= 50")
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
    data_list = [float(x) for x in data_list_str]
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
    
    
    return data_dict, time_and_labels

def slice_data(file_path: str):
    """Slice large data files into smaller chunks"""
    file_path = Path(file_path)
    
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        raise Exception(f'File type {file_path.suffix} not supported')
    
    p = 40000
    split_indices = list(range(0, len(df) + 1, p))[:-1]
    paths = []
    n_sliced = len(split_indices)
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

def load_inputs(data_path, time_cutoff, shuffle=True, seed=0):
    x_dict, x_time_label = prepare_dataset(data_path, time_cutoff=time_cutoff)
    if shuffle:
        np.random.seed(seed)
        n_samples = len(x_dict[list(x_dict.keys())[0]])
        shuffled_idx = np.arange(n_samples)
        np.random.shuffle(shuffled_idx)
        x_dict = {id: seqs[shuffled_idx] for id, seqs in x_dict.items()}
        x_time_label = x_time_label[shuffled_idx]
    y = np.concatenate([x_dict[id][:, -1, :] for id in FIXED_IDS], axis=1)
    return x_dict, y, x_time_label


# Calculate constants
N_SIGS = sum([n_sig for n_sig in ID_NSIG.values()])
FIXED_IDS = list(ID_NSIG.keys())
FIXED_IDS.sort()

def create_canet_model(h):
    """Create CANet model - FIX: Handle empty FIXED_IDS"""
    if not FIXED_IDS:
        raise ValueError("No CAN IDs available for model creation")
    
    inputs = {id: keras.Input(shape=(ID_MPS[id], ID_NSIG[id]), name=id) for id in FIXED_IDS}
    lstms = [keras.layers.LSTM(h * ID_NSIG[id], name=f'lstm_{id}') for id in FIXED_IDS]
    x_id = [lstms[i](inputs[id]) for i, id in enumerate(FIXED_IDS)]
    
    if len(x_id) > 1:
        x = keras.layers.Concatenate()(x_id)
    else:
        x = x_id[0]
        
    x = keras.layers.Dense((h * N_SIGS) // 2, activation='elu')(x)
    x = keras.layers.Dense(N_SIGS - 1, activation='elu')(x)
    outputs = keras.layers.Dense(N_SIGS, activation='elu')(x)
    return keras.Model(inputs, outputs)
def get_datasets_from_group(group):
    """
    Recursively get all datasets from an HDF5 group as a flat list.
    """
    datasets = []
    def visitor(name, node):
        if isinstance(node, h5py.Dataset):
            datasets.append(node[()])
    group.visititems(visitor)
    return datasets

def load_weights_by_order(model, weights_path):
    with h5py.File(weights_path, 'r') as f:
        weight_layer_names = list(f.keys())
        trainable_layers = [layer for layer in model.layers if layer.weights]

        if len(trainable_layers) != len(weight_layer_names):
            print(f"Warning: trainable layers ({len(trainable_layers)}) != weight groups ({len(weight_layer_names)})")

        for i, layer in enumerate(trainable_layers):
            if i >= len(weight_layer_names):
                print(f"No weight group at index {i}")
                break

            weight_group = f[weight_layer_names[i]]
            weights = []
            for key in weight_group.keys():
                if isinstance(weight_group[key], h5py.Dataset):
                    weights.append(weight_group[key][()])

            # Print debug info
            print(f"Layer '{layer.name}' expects {len(layer.weights)} weight arrays")
            print(f"Weight group '{weight_layer_names[i]}' provides {len(weights)} arrays")

            try:
                layer.set_weights(weights)
                print(f"Loaded weights into layer '{layer.name}'")
            except Exception as e:
                print(f"Error loading weights into layer '{layer.name}': {e}")
h = 5
dt_experiment = 'models/CANET/road_2025-05-28_16-56-27_epoch01.weights.h5'
model = create_canet_model(h)
model.summary()
print(f'# parameters: {model.count_params():,}')
weight_file = 'models/CANET/road_2025-05-28_16-56-27_epoch01.weights.h5'
load_weights_by_order(model, 'models/CANET/road_2025-05-28_16-56-27_epoch01.weights.h5')
mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
batch_size = 250

with tf.device('/CPU:0'):
    # Load the test dataset
    test_file = data_files['test'][0]
    print(f'Getting predictions of the first test file {Path(test_file).name} with measuring inference speed')
    attack = Path(test_file).stem.split('_')[-1]
    x_test_dict, y_test, time_label_test = load_inputs(test_file, time_cutoff=WINDOW_SIZE + 1, shuffle=False)
    x_time = time_label_test[:, 0]
    x_label = time_label_test[:, 1]
    with tf.device('CPU'):
        test_ds = tf.data.Dataset.from_tensor_slices(x_test_dict)
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        start_time = time.process_time()
        y_pred = model.predict(test_ds, verbose=0)
        mse_values = mse(y_test, y_pred).numpy()
    end_time = time.process_time()

# Measure inference speed
process_time = end_time - start_time
inference_speed = len(y_test) / process_time
print("-----------------------------------------")
print(f'CPU execution time: {process_time:,} seconds')
print(f'Inference speed: {inference_speed:.2f} messages per second')
print("-----------------------------------------")

# Save the results of the first file
results = pd.DataFrame({'Time': x_time, 'MSE': mse_values, 'Session': x_label})
results['Time'] = results['Time'].round(7)
results['Session'] = results['Session'].astype(int)
save_path = f'../../Results/road_CANet_{dt_experiment}_{attack}.parquet'
results.to_parquet(save_path, index=False)
print(f'Predictions are saved at {save_path}')

# Save the results of the rest test files
for test_file in tqdm(data_files['test'][1:]):
    print(f'Getting predictions of {Path(test_file).name}')
    attack = Path(test_file).stem.split('_')[-1]
    x_test_dict, y_test, time_label_test = load_inputs(test_file, time_cutoff=WINDOW_SIZE + 1, shuffle=False)
    x_time = time_label_test[:, 0]
    x_label = time_label_test[:, 1]
    with tf.device('CPU'):
        test_ds = tf.data.Dataset.from_tensor_slices(x_test_dict)
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    y_pred = model.predict(test_ds, verbose=0)
    mse_values = mse(y_test, y_pred).numpy()
    results = pd.DataFrame({'Time': x_time, 'MSE': mse_values, 'Session': x_label})
    results['Time'] = results['Time'].round(7)
    results['Session'] = results['Session'].astype(int)
    save_path = f'../../Results/road_CANet_{dt_experiment}_{attack}.parquet'
    results.to_parquet(save_path, index=False)
    print(f'Predictions are saved at {save_path}')