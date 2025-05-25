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

import pandas as pd
from collections import OrderedDict

DATASET_DIR = '../../Datasets/road/signal_extractions/CANet'
# Load your dataset
df = pd.read_csv("../../Datasets/road/signal_extractions/ambient/ambient_dyno_drive_basic_long.csv")
data_files = {
    'train': glob.glob(f'{DATASET_DIR}/train/*.csv'),
    'valid': glob.glob(f'{DATASET_DIR}/val/*.csv'),
    'test': glob.glob('../../Datasets/road/signal_extractions/attack/*.csv')
}
# List of signal columns (everything after column 2)
signal_columns = df.columns[3:]
WINDOW_SIZE = 1
TRAIN_START = WINDOW_SIZE + 1
# Initialize dictionaries
ID_NSIG = OrderedDict()
ID_MPS = {}

# Group by CAN ID
grouped = df.groupby("ID")

for can_id, group in grouped:
    # Count how many rows this CAN ID has → MPS (message per sequence)
    ID_MPS[str(can_id)] = len(group)
    
    # For NSIG: find the max number of non-null signal columns used in any row of this ID
    signals_used = group[signal_columns].notnull().sum(axis=1)
    ID_NSIG[str(can_id)] = int(signals_used.max())

# Done!
start_time = df["Time"].min()
end_time = df["Time"].max()
duration_sec = end_time - start_time
ID_MPS = {id_: round(count / duration_sec) for id_, count in ID_MPS.items()}
ID_MPS = dict(sorted(ID_MPS.items(), key=lambda x: x[1], reverse=True))
print("ID_FREQ =", ID_MPS)
print("ID_MPS =", ID_MPS)
print("ID_NSIG =", ID_NSIG)

ID_FREQ = {k: v for k, v in ID_MPS.items() if v >= 40}
ID_NSIG = {k: ID_NSIG[k] for k in ID_FREQ.keys()}
def load_arrange_data(file_path,print_option=False):
    # Step 1: Convert time from microseconds to seconds (assumed from sample)
    df = pd.read_parquet(file_path)
    df['Time'] = round(df['Time'], 7)

    # Step 2: Rename 'Label' to 'Session'
    df.rename(columns={'Label': 'Session'}, inplace=True)

    # Step 3: Rename signal columns uniformly to 'Signal1', 'Signal2', ...
    signal_cols = [col for col in df.columns if col.startswith("Signal_")]
    for i, col in enumerate(signal_cols):
        df.rename(columns={col: f'Signal{i+1}'}, inplace=True)

    # Step 4: Drop trailing empty signal columns (e.g., NaNs or blanks)
    signal_cols = [col for col in df.columns if col.startswith("Signal")]
     # Optional: drop empty columns (those that are all empty)
    df = df.dropna(axis=1, how='all')

    if print_option:
        print(f'# rows: {df.shape[0]:,}')
        print(df['Session'].value_counts())
    return df

def str_to_list(data_str: str) -> list:
    data_list_str = data_str.split()
    data_list = [int(x) for x in data_list_str]
    if len(data_list) < 8:  # fill with dummy values (0) to 8 bytes
        data_list += [0] * (8 - len(data_list))
    return data_list
    return df
def get_repeated_sequences(data, can_id, n_sig, n_step):
    sig_columns = [f'Signal{i}' for i in range(1, n_sig + 1)]
    df_id = data.loc[data['ID'] == can_id, ['Idx'] + sig_columns]
    np_sig = df_id[sig_columns].to_numpy()
    np_seq = np.lib.stride_tricks.sliding_window_view(np_sig, window_shape=n_step, axis=0)
    np_seq = np_seq.swapaxes(1, 2)
    n_seq = np_seq.shape[0]
    end_idx = data['Idx'].iloc[-1]
    n_repeats = np.diff(df_id['Idx'].to_list() + [end_idx])[-n_seq:]
    repeated_seq = np.repeat(np_seq, n_repeats, axis=0)
    return repeated_seq

def prepare_dataset(file_path, time_cutoff, label='last', print_option=True):
    assert label in ['last', 'sum', False]
    data = load_arrange_data(file_path, print_option=False)
    data = data.reset_index(names='Idx')
    time_start = data['Time'].iloc[0]
    n_rows_to_use = data.loc[data['Time'] > time_start + time_cutoff, 'Time'].shape[0]
    if label == 'last':
        time_and_labels = data.loc[data['Time'] > time_start + time_cutoff, ['Time', 'Session']].to_numpy()
    data_dict = dict()
    for id, nsig in ID_NSIG.items():
        seq_data = get_repeated_sequences(data, id, nsig, ID_MPS[id])
        data_dict[id] = seq_data[-n_rows_to_use:].copy()
        if print_option:
            print(id, data_dict[id].shape)
    assert all([l == n_rows_to_use for l in map(len, data_dict.values())])
    if label:
        assert len(time_and_labels) == n_rows_to_use
        return data_dict, time_and_labels
    else:
        return data_dict

def get_count_features(data_df: pd.DataFrame, window_size_s: int, start_s=0) -> pd.DataFrame:
    data_df = data_df[['Time', 'ID']].copy()
    data_df['occur'] = 1
    data_df = pd.concat([data_df['Time'], data_df.pivot(columns='ID', values='occur')], axis=1)
    data_df['TimeIndex'] = pd.to_timedelta(data_df['Time'], unit='s')
    data_df.set_index('TimeIndex', inplace=True)
    count_df = data_df.drop(columns=['Time']).rolling(str(window_size_s)+'s').count()
    count_df = count_df.astype('int16')
    assert len(data_df) == len(count_df)
    real_start_s = count_df.index.min().seconds
    real_start_us = count_df.index.min().microseconds
    count_df = count_df.loc[count_df.index > pd.Timedelta(seconds=real_start_s + start_s, microseconds=real_start_us)]
    return count_df

def load_inputs(data_path, time_cutoff, label, shuffle=True, seed=0):
    if not label:
        x_dict = prepare_dataset(data_path, time_cutoff=time_cutoff, label=False, print_option=False)
    else:
        x_dict, x_time_label = prepare_dataset(data_path, time_cutoff=time_cutoff, label=label, print_option=False)
    if shuffle:
        np.random.seed(seed)
        n_samples = len(x_dict[list(x_dict.keys())[0]])
        shuffled_idx = np.arange(n_samples)
        np.random.shuffle(shuffled_idx)
        x_dict = {id: seqs[shuffled_idx] for id, seqs in x_dict.items()}
        if label:
            x_time_label = x_time_label[shuffled_idx]
    y = np.concatenate([x_dict[id][:, -1, :] for id in FIXED_IDS], axis=1)
    if not label:
        return x_dict, y
    else:
        return x_dict, y, x_time_label
N_SIGS = sum([n_sig for n_sig in ID_NSIG.values()])
FIXED_IDS = list(ID_NSIG.keys())
FIXED_IDS.sort()

def create_canet_model(h):
    inputs = {id: keras.Input(shape=(ID_MPS[id], ID_NSIG[id]), name=id) for id in FIXED_IDS}
    lstms = [keras.layers.LSTM(h * ID_NSIG[id], name=f'lstm_{id}') for id in FIXED_IDS]
    x_id = [lstms[i](inputs[id]) for i, id in enumerate(FIXED_IDS)]
    x = keras.layers.Concatenate()(x_id)
    x = keras.layers.Dense((h * N_SIGS) // 2, activation='elu')(x)
    x = keras.layers.Dense(N_SIGS - 1, activation='elu')(x)
    outputs = keras.layers.Dense(N_SIGS, activation='elu')(x)
    return keras.Model(inputs, outputs)
# Create the model
h = 5
lr = 0.0001 # learning rate
model = create_canet_model(h)
optimizer = keras.optimizers.Adam(learning_rate=lr)
model.compile(
    optimizer=optimizer,
    loss='mean_squared_error'
)
def slice_data(file_path: str, n_sliced: int):
    if Path(file_path).suffix == '.csv':
        df = pd.read_csv(file_path)
    elif Path(file_path).suffix == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        raise Exception(f'File type {file_path} not supported')
    p = len(df) // n_sliced
    split_indices = list(range(0, len(df) + 1, p))[:-1]
    paths = []
    for i in range(n_sliced):
        if i + 1 < n_sliced:
            sliced_df = df.iloc[split_indices[i]:split_indices[i+1]]
        else:  # the last slice
            sliced_df = df.iloc[split_indices[i]:]
        cache_dir = Path(file_path).parent / 'cache'
        cache_dir.mkdir(exist_ok=True)
        save_path = str(cache_dir / f'{Path(file_path).stem}_{i+1}.parquet')
        sliced_df.to_parquet(save_path)
        paths.append(save_path)
    return paths

batch_size = 250
n_epoch = 10

starttime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
for epoch in range(n_epoch):
    print(f'********** Epoch {epoch+1} **********')
    # Train the model
    for data_file in data_files['train']:
        sliced_files = slice_data(data_file, 10)  # When the data is too big to make inputs at once
        for sliced_file in sliced_files:
            x_train_dict, y_train = load_inputs(sliced_file, time_cutoff=TRAIN_START, label=False, shuffle=True, seed=epoch)
            print(f'Training with {sliced_file} {y_train.shape}')
            with tf.device('CPU'):
                train_ds = tf.data.Dataset.from_tensor_slices((x_train_dict, y_train))
            train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            model.fit(train_ds, verbose=2)
            del x_train_dict, y_train, train_ds
            gc.collect()

    # Validate the model
    val_loss = 0
    for data_file in data_files['valid']:
        
        sliced_files = slice_data(data_file, 10)  # When the data is too big to make inputs at once
        for sliced_file in sliced_files:
            x_valid_dict, y_valid = load_inputs(sliced_file, time_cutoff=TRAIN_START, label=False, shuffle=False)
            gc.collect()
            print(f'Validating with {sliced_file} {y_valid.shape}')
            with tf.device('CPU'):
                valid_ds = tf.data.Dataset.from_tensor_slices((x_valid_dict, y_valid))
            valid_ds = valid_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            val_loss += model.evaluate(valid_ds, verbose=2)
            del x_valid_dict, y_valid, valid_ds
            gc.collect()

    # save the model weight
    weight_name = f'models/CANET/{"road"}_{starttime}_epoch{epoch+1:02}'
    model.save_weights(weight_name)