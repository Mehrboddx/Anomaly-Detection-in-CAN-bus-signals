import glob
import pandas as pd
import cantools
import os
import tempfile
import shutil
# === 1. Load min/max values from DBC ===

import pandas as pd
import cantools

def get_signal_min_max(signal):
    if signal.minimum is not None and signal.maximum is not None and signal.minimum != signal.maximum:
        return signal.minimum, signal.maximum

    length = signal.length
    is_signed = signal.is_signed
    factor = signal.scale or 1.0
    offset = signal.offset or 0.0

    if is_signed:
        raw_min = -1 * (2 ** (length - 1))
        raw_max = (2 ** (length - 1)) - 1
    else:
        raw_min = 0
        raw_max = (2 ** length) - 1

    min_physical = raw_min * factor + offset
    max_physical = raw_max * factor + offset
    return min_physical, max_physical

def load_signal_ranges(dbc_path):
    db = cantools.database.load_file(dbc_path)
    signal_ranges = {}
    for message in db.messages:
        frame_id = message.frame_id
        for idx, signal in enumerate(message.signals):
            min_val, max_val = get_signal_min_max(signal)
            signal_ranges[f"{frame_id}+{idx}"] = {"min": min_val, "max": max_val}
    return signal_ranges

def normalize_chunk(chunk, signal_ranges):
    for col_idx in range(3, chunk.shape[1]):
        col_name = chunk.columns[col_idx]
        signal_num = col_idx - 3

        # Normalize per frame_id in this chunk
        for frame_id in chunk["ID"].unique():
            key = f"{frame_id}+{signal_num}"
            if key not in signal_ranges:
                continue

            min_val = signal_ranges[key]["min"]
            max_val = signal_ranges[key]["max"]
            if min_val == max_val:
                continue

            mask = chunk["ID"] == frame_id
            chunk[col_name] = chunk[col_name].astype(float)
            chunk.loc[mask, col_name] = (
                pd.to_numeric(chunk.loc[mask, col_name], errors='coerce') - min_val
            ) / (max_val - min_val)
    return chunk

def normalize_csv_chunks(csv_path, dbc_path, chunksize=100_000):
    signal_ranges = load_signal_ranges(dbc_path)

    # Create a temp file in the same directory as the original
    dir_name = os.path.dirname(csv_path)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=dir_name, suffix='.csv') as tmpfile:
        tmp_path = tmpfile.name

    # Normalize in chunks and write to temp file
    first_chunk = True
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk = normalize_chunk(chunk, signal_ranges)

        if first_chunk:
            chunk.to_csv(tmp_path, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(tmp_path, index=False, mode='a', header=False)

    # Replace original file with the temp file
    shutil.move(tmp_path, csv_path)
    print(f"Normalized and overwritten: {csv_path}")

# Example usage:
if __name__ == "__main__":
    files = glob.glob("road/signal_extractions/ambient/*.csv")
    dbc_path = "road/signal_extractions/DBC/anonymized.dbc"

    for file in files:
        output_path = file  # Overwrite the original file
        normalize_csv_chunks(file, dbc_path)
        print(f"Normalized and overwritten: {file}")
    
    files = glob.glob("road/signal_extractions/attacks/*.csv")
    dbc_path = "road/signal_extractions/DBC/anonymized.dbc"

    for file in files:
        output_path = file  # Overwrite the original file
        normalize_csv_chunks(file, dbc_path)
        print(f"Normalized and overwritten: {file}")


