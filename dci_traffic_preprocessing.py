import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use backend for saving plots only (no GUI)
import matplotlib.pyplot as plt


def sliding_time_windows_split(df, file_name, output_dir, window_size=10, num_windows=10, start_multiplier=0.2):
    """
    Split DCI traffic data using sliding time windows with different starting offsets.

    Args:
        df (DataFrame): Parsed DCI CSV data.
        file_name (str): Output file name without extension.
        output_dir (str): Path to save the generated .npy file.
        window_size (int): Length of each sliding window (in seconds).
        num_windows (int): Number of sliding windows with different start offsets.
        start_multiplier (float): Time offset multiplier for each sliding window.
    """

    print(f"File {file_name} split start.")
    time_list = []
    tbs_list = []

    for i in range(num_windows):
        start_time = i * start_multiplier  # Start of the i-th window
        end_time = start_time + window_size
        # Slide window with step equal to window size (non-overlapping)
        while end_time <= (300 + num_windows * start_multiplier):
            window = df[(df['time'] >= start_time) & (df['time'] < end_time)].copy()
            window['time'] = window['time'] - window['time'].iloc[0]
            window['time'] = window['time'].round(3)
            window = window[window['rnti'] != 65535].reset_index(drop=True)  # Remove non-c-rnti rows

            # Mark uplink(0) TBS as negative
            window.loc[window['drct'] == 0, 'tbs'] *= -1
            window = window.drop(['rnti', 'drct'], axis=1)

            time_list.append(window['time'].tolist())
            tbs_list.append(window['tbs'].tolist())
            start_time += window_size
            end_time += window_size

    # Convert to numpy
    time = np.array(time_list, dtype=object)
    tbs = np.array(tbs_list, dtype=object)
    data = np.empty(len(time), dtype=[('time', object), ('tbs', object)])
    data['time'] = time
    data['tbs'] = tbs

    output_file_path = os.path.join(output_dir, f'{file_name}')
    np.save(output_file_path, data, allow_pickle=True)

    print(f"File {file_name} split complete.")


def csv_processing(csvfile_path, npy_output_dir):
    """
    Load and preprocess DCI CSV.

    Args:
        csvfile_path (str): Path to input DCI CSV file.
        npy_output_dir (str): Directory to save the resulting .npy file.
    """

    if not os.path.isfile(csvfile_path):
        print(f"Error: File '{csvfile_path}' is invalid.")
        return
    os.makedirs(npy_output_dir, exist_ok=True)

    # Load required columns: timestamp, rnti, direction, tbs
    df = pd.read_csv(csvfile_path, header=None, delimiter='\t', usecols=[0, 3, 4, 7])
    df.columns = ['time', 'rnti', 'drct', 'tbs']

    # Sort by time
    df = df.sort_values(by='time').reset_index(drop=True)
    df['time'] = df['time'].round(3)
    df['time'] = df['time'] - df['time'].iloc[0]

    # Perform time-window splitting
    input_filename = os.path.basename(csvfile_path)
    file_name_without_extension = os.path.splitext(input_filename)[0]
    sliding_time_windows_split(df, file_name_without_extension, npy_output_dir)


def plot_tbs_npy_scatter(npy_path, fig_output_dir, num_fig):
    """
    Plot TBS scatter plots.

    Args:
        npy_path (str): Path to input .npy file.
        fig_output_dir (str): Directory to save TBS scatter images.
        num_fig (int): Index offset for saved image file names.
    """

    if not os.path.isfile(npy_path):
        print(f"File {npy_path} is invalid.")
        return
    os.makedirs(fig_output_dir, exist_ok=True)

    input_filename = os.path.basename(npy_path)
    file_name_without_extension = os.path.splitext(input_filename)[0]
    print(f"Starting plot {file_name_without_extension} scatter.")

    # Load TBS
    data = np.load(npy_path, allow_pickle=True)
    x = data['time']
    y = data['tbs']

    for i in range(len(x)):
        plt.figure(figsize=(6,6))
        plt.scatter(x[i], y[i], c='red', s=10)
        plt.ylim([-63776, 63776])
        plt.xlim([0, 10])
        plt.yticks([])
        plt.xticks([])

        # Save figure with offset index, 300 = 300 / window_size * num_windows
        plt.savefig(os.path.join(fig_output_dir, f'{i+1+300*num_fig}.png'), bbox_inches='tight', pad_inches=0.2)
        plt.close('all')


def main():
    """
    Main entry: process CSV and generate TBS scatter plots.
    """

    csvfile_path = "./dci_data_csv/YouTube.csv"
    npy_output_dir = "./data_npy"
    csv_processing(csvfile_path, npy_output_dir)

    npy_path = "./data_npy/YouTube.npy"
    fig_output_dir = "./datasets/train/YouTube"
    plot_tbs_npy_scatter(npy_path, fig_output_dir, 0)


if __name__ == '__main__':
    main()

