from scipy.signal import butter, filtfilt, welch, find_peaks
import matplotlib.pyplot as plt
import numpy as np


## finding peaks for simple classifier
def find_max_min_pattern(max_peaks, min_peaks):
    # Create a list to store the windows that match the pattern
    pattern_windows = []

    # Iterate through the max peaks and check if the pattern occurs
    for i in range(len(max_peaks) - 2):  # We need at least 5 peaks for a complete pattern
        # Define the candidate window of 5 peaks (max, min, max, min, max)
        candidate_max1 = max_peaks[i]
        candidate_min1 = min_peaks[i] if i < len(min_peaks) else None
        candidate_max2 = max_peaks[i + 1]
        candidate_min2 = min_peaks[i + 1] if i + 1 < len(min_peaks) else None
        # candidate_max3 = max_peaks[i + 2]

        # print((candidate_max1, candidate_min1, candidate_max2, candidate_min2, candidate_max3))

        # Check if we have a valid pattern: max, min, max, min, max
        if (candidate_min1 is not None and candidate_max1 < candidate_min1 and
            candidate_max2 > candidate_min1 and candidate_min2 is not None and
            candidate_max2 < candidate_min2): #and candidate_max3 > candidate_min2
            # If the pattern matches, store the start and end indices of the window
            pattern_windows += [candidate_max1, candidate_min1, candidate_max2, candidate_min2] #candidate_max3

    return pattern_windows

def filter(window):
    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)
    def lowpass_filter(data, cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low')
        return filtfilt(b, a, data, axis=0)
    
    window = bandpass_filter(window, 1, 50, 250)
    window = lowpass_filter(window, 10, 250)
    return window


def model(window, i):
    filtered_eeg_window = filter(window)
    # if filtered_eeg_window.std() > 10:
    signal = filtered_eeg_window
    peaks, _ = find_peaks(-1*signal, height=75, distance=40)
    max_peaks, _ = find_peaks(signal, height=50, distance=40)
    potential_peaks = peaks
    potential_max_peaks =  max_peaks
    ground_points = find_max_min_pattern(max_peaks, peaks)
    print(peaks)
    if len(ground_points) > 1 and signal[potential_max_peaks].std() < 100 and len(peaks) == 2 and (peaks[1] - peaks[0]) < 250:
        # fig, ax = plt.subplots()    
        # filtered_eeg_window.plot(kind='line')
        # plt.scatter(potential_peaks, signal[potential_peaks], color='r', label="Potential Peaks", marker='x')
        # plt.scatter(potential_max_peaks, signal[potential_max_peaks], color='g', label="Potential Peaks", marker='x')
        # fig.savefig(f'blink_graphs/img{i}.png')
        np.save(f'blink_graphs/img{i}.npy', signal)
        return True
        # return potential_peaks, potential_max_peaks

    return False
