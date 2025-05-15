import csv
import os
import argparse

# Attack intervals for each log file
attack_intervals = {
    "correlated_signal_attack_1_masquerade.csv": (9.191851, 30.050109),
    "correlated_signal_attack_2_masquerade.csv": (6.830477, 28.225908),
    "correlated_signal_attack_3_masquerade.csv": (4.318482, 16.95706),
    "max_engine_coolant_temp_attack_masquerade.csv": (19.979078, 24.170183),
    "max_speedometer_attack_1_masquerade.csv": (42.009204, 66.449011),
    "max_speedometer_attack_2_masquerade.csv": (16.009225, 47.408246),
    "max_speedometer_attack_3_masquerade.csv": (9.516489, 70.587285),
    "reverse_light_off_attack_1_masquerade.csv": (16.627923, 23.347311),
    "reverse_light_off_attack_2_masquerade.csv": (13.168608, 36.87663),
    "reverse_light_off_attack_3_masquerade.csv": (16.524085, 40.862015),
    "reverse_light_on_attack_1_masquerade.csv": (18.929177, 38.836015),
    "reverse_light_on_attack_2_masquerade.csv": (20.407134, 57.297253),
    "reverse_light_on_attack_3_masquerade.csv": (23.070278, 46.580686)
}

def process_log_file(log_path, attack_range):
    """ Process the log file and change the labels based on attack intervals """
    updated_rows = []

    with open(log_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        updated_rows.append(header)

        for row in reader:
            timestamp = float(row[1])
            label = 1 if attack_range[0] <= timestamp <= attack_range[1] else 0
            row[0] = label
            updated_rows.append(row)

    # Save back to the same file
    with open(log_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(updated_rows)
    
    print(f"File updated: {log_path}")

def batch_process_logs(logs_folder):
    """ Process all log files listed in the attack_intervals dictionary """
    for log_file, attack_range in attack_intervals.items():
        log_path = os.path.join(logs_folder, log_file)
        if os.path.exists(log_path):
            print(f"Processing: {log_path}")
            process_log_file(log_path, attack_range)
        else:
            print(f"File not found: {log_path}")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CAN log files and label attacks.")
    parser.add_argument("--logs-folder", type=str, required=True, help="Path to the folder containing log files.")
    args = parser.parse_args()
    batch_process_logs(args.logs_folder)

