from psychopy import visual, core, logging, event
import random

import time

import numpy as np
import pandas as pd

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter

import glob, sys, time, serial
from serial import Serial
from threading import Thread, Event
from queue import Queue
from scipy.signal import find_peaks, butter, filtfilt

experiment_running = True
cyton_in = True
test = False
lsl_out = False
calibrate = True


# Create window
width = 1440
height = 900

win = visual.Window([1440, 900], color="black", units="pix", fullscr= False)

# Create fixation cross and cue
fixation = visual.TextStim(win, text="+", color="white", height=50, pos=(0, 0))  
cue = visual.TextStim(win, text="", color="white", height=30, pos=(0, 0))  

# List of artifacts
artifacts = ['jaw_clench', 'double_blink', 'jaw_clench_blink', 'blink_hard']

artifact_marker = {artifacts[i]: i + 1 for i in range(len(artifacts))}

# Logging
logging.setDefaultClock(core.Clock())

# Calibration
def calibration_phase(duration=15.0):
    instruction = visual.TextStim(win, 
                                  text="Fix your eyes. Watch the screen and blink naturally for 15s.", 
                                  color="white", height=30, pos=(0, 150))  

    start_time = core.getTime()  
    logging.log(level=logging.DATA, msg="Calibration phase started.")

    while core.getTime() - start_time < duration:
        fixation.draw()  
        instruction.draw()  
        win.flip()

    logging.log(level=logging.DATA, msg="Calibration phase ended.")


# Reminder
def trial_reminder(trial_num):
    """
    Displays a reminder before each trial starts.
    """
    reminder = visual.TextStim(win, text=f"Trial {trial_num} is about to start.\nPress any key to continue.", 
                               color="white", height=30, pos=(0, 150))  
    fixation.draw() 
    reminder.draw()
    win.flip()
    logging.log(level=logging.DATA, msg=f'Reminder before Trial {trial_num}')
    event.waitKeys()  

# Cues and logging artifact data
def present_cue(artifact, duration=1.5):
    #fixation.draw()
    cue.setText(f"{artifact}")
    cue.draw()
    win.flip()
    logging.log(level=logging.DATA, msg=f'Artifact cue presented: {artifact}')
    core.wait(duration) 

    # Display fixation and "Begin artifact" message together
    photosensor_dot.color = np.array([1, 1, 1])
    photosensor_dot.draw()
    fixation.draw()
    # board.insert_marker(marker)
    #cue.setText("Begin artifact")
    #cue.draw()
    win.flip()
    photosensor_dot.color = np.array([-1, -1, -1])
    photosensor_dot.draw()
    # board.insert_marker(marker)
    logging.log(level=logging.DATA, msg=f'Action cue presented: {artifact}')
    core.wait(1.5)

# Simon's code for creating photosensor
def create_photosensor_dot(size=2/8*0.7):
    ratio = width/height
    dot_size = size * min(width, height)
    return visual.Rect(win=win, units="pix", width=dot_size, height=dot_size * ratio, 
                       fillColor='white', pos = (width/2 - dot_size/2, -height/2 + dot_size/2) #[1 - size/2, -1 - size/8]
    )

# Randomized
def generate_experiment_trials(num_trials=3, num_samples_per_trial=40):
    """
    - 30 samples per artifact across 3 trials
    - 40 artifact samples per trial
    """
    total_samples = num_trials * num_samples_per_trial  
    artifact_counts = {artifact: 0 for artifact in artifacts}  

    all_samples = []

    while len(all_samples) < total_samples:
        available_artifacts = [a for a in artifacts if artifact_counts[a] < 30]  
        if not available_artifacts:
            break  

        artifact = random.choice(available_artifacts)
        artifact_counts[artifact] += 1
        all_samples.append(artifact)

    
    trials = [all_samples[i * num_samples_per_trial: (i + 1) * num_samples_per_trial] for i in range(num_trials)]
    for trial in trials:
        random.shuffle(trial)  
    
    return trials

def end_experiment():
    experiment_running = False
    win.flip()


def write_data_file(trial_num):
    BoardShim.enable_dev_board_logger()

    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    time.sleep(2)
    data = board.get_board_data()
    # board.stop_stream() -- what is this???
    # board.release_session()  -- what is this???

    # demo how to convert it to pandas DF and plot data
    # if test:
    #     eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value) #CYTON_BOARD_ID.value me thinks
    # else:
    #     eeg_channels = board.get_eeg_channels(CYTON_BOARD_ID) #work in progress
    #     aux_channels = board.get_analog_channels(CYTON_BOARD_ID)
    #     data = np.hstack((eeg_channels, aux_channels))
    # df = pd.DataFrame(np.transpose(data))
    # df.columns = eeg_channels + aux_channels

    # demo for data serialization using brainflow API, we recommend to use it instead pandas.to_csv()
    filename = f"trial_{trial_num}.csv"
    DataFilter.write_file(data, filename, 'a')  # use 'a' for append mode
    # DataFilter.write_file(eeg_channels, f'eeg_data_trial{trial_num}', 'a')
    # DataFilter.write_file(aux_channels, f'aux_data_trial{trial_num}', 'a')
    # board.prepare_session()  -- what is this???
    # board.start_stream()  -- what is this???

# for filtering for the model
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# Simon's code for Brainflow + getting parameters
if cyton_in: 
    CYTON_BOARD_ID = 0 
    BAUD_RATE = 115200
    ANALOGUE_MODE = '/2'

    #Finds the port to which the Cyton Dongle is connected to.
def find_openbci_port(): 
    # Find serial port names per OS
    print(sys.platform)
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        ports = glob.glob('/dev/ttyUSB*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/cu.usbserial*')
        print(ports)
    else:
        raise EnvironmentError('Error finding ports on your operating system')
    openbci_port = ''
    for port in ports:
        try:
            s = serial.Serial(port=port, baudrate=BAUD_RATE, timeout=None)
            s.write(b'v')
            line = ''
            time.sleep(2)
            if s.inWaiting():
                line = ''
                c = ''
                while '$$$' not in line:
                    c = s.read().decode('utf-8', errors='replace')
                    line += c
                if 'OpenBCI' in line:
                    openbci_port = port
            s.close()
        except (OSError, serial.SerialException):
            pass
    if openbci_port == '':
        raise OSError('Cannot find OpenBCI port.')
        exit()
    else:
        return openbci_port
        

#Responsible for collecting EEG data from the OpenBCI Cyton board 
def get_data(queue_in, lsl_out=False):
    while not stop_event.is_set():
        data_in = board.get_board_data()
        timestamp_in = data_in[board.get_timestamp_channel(CYTON_BOARD_ID)]
        eeg_in = data_in[board.get_eeg_channels(CYTON_BOARD_ID)]
        aux_in = data_in[board.get_analog_channels(CYTON_BOARD_ID)]
        if len(timestamp_in) > 0:
            print('queue-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
            queue_in.put((eeg_in, aux_in, timestamp_in))
        time.sleep(0.1)
    # while not queue_in.empty(): # Collect all data from the queue
    #                 eeg_in, aux_in, timestamp_in = queue_in.get()
    #                 print('data-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
    #                 eeg = np.concatenate((eeg, eeg_in), axis=1)
    #                 aux = np.concatenate((aux, aux_in), axis=1)
    #                 timestamp = np.concatenate((timestamp, timestamp_in), axis=0)
    # return eeg, aux, timestamp
    # if os.path.exists(model_file_path):
    #     with open(model_file_path, 'rb') as f:
    #         model = pickle.load(f)
    # else:
    #     model = None

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
        candidate_max3 = max_peaks[i + 2]

        # print((candidate_max1, candidate_min1, candidate_max2, candidate_min2, candidate_max3))

        # Check if we have a valid pattern: max, min, max, min, max
        if (candidate_min1 is not None and candidate_max1 < candidate_min1 and
            candidate_max2 > candidate_min1 and candidate_min2 is not None and
            candidate_max2 < candidate_min2 and candidate_max3 > candidate_min2):
            # If the pattern matches, store the start and end indices of the window
            pattern_windows += [candidate_max1, candidate_min1, candidate_max2, candidate_min2, candidate_max3]

    return pattern_windows

def simple_classifier(file):
    file_name = file.split('.npy')[0]
    fs = 250
    new_data = np.load(file)
    df = pd.DataFrame(new_data)

    df['onsets'] = (df[9].shift(1) < 30) & (df[9] >= 30)

    all_onsets = df[['onsets']].index
    eeg_data = df[["EEG_1", "EEG_2", "EEG_3", "EEG_4"]]
    filtered_eeg = eeg_data.apply(lambda x: bandpass_filter(x, 1, 50, fs))

    for i, onset_idx in enumerate(all_onsets):
        offset_idx = onset_idx + int(1 * fs)

        offset_idx = min(offset_idx, len(df) - 1)

        signal = filtered_eeg['EEG_3'].loc[onset_idx: offset_idx]
        peaks, _ = find_peaks(-1*signal, height=40, distance=60)
        max_peaks, _ = find_peaks(signal, height=20, distance=60)
        potential_peaks = signal.index[0] + peaks
        potential_max_peaks = signal.index[0] + max_peaks

        ground_points = find_max_min_pattern(max_peaks, peaks)

        df.at[onset_idx, 'potential_peaks'] = potential_peaks.tolist()  
        df.at[onset_idx, 'potential_max_peaks'] = potential_max_peaks.tolist() 
        df.at[onset_idx, 'ground_points'] = ground_points.tolist()
    
    np.save(f'{file_name}_processed.npy', df.to_numpy())
    


# Performing the Experiment
params = BrainFlowInputParams()  #this sets up the connection Parameters

if test:
    board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
    stop_event = Event()
# board.prepare_session()
    board.start_stream() #what is 45000????
if not test and cyton_in: 
    BAUD_RATE = 115200
    ANALOGUE_MODE = '/2'
    if CYTON_BOARD_ID != 6:
        params.serial_port = find_openbci_port()
    elif CYTON_BOARD_ID == 6:
        params.ip_port = 9000
    board = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    res_query = board.config_board('/0')
    print(res_query)
    res_query = board.config_board('//')
    print(res_query)
    res_query = board.config_board(ANALOGUE_MODE)
    print(res_query)
    # board.prepare_session()
    board.start_stream(450000)
    stop_event = Event()

    queue_in = Queue()
    cyton_thread = Thread(target=get_data, args=(queue_in, lsl_out))
    cyton_thread.daemon = True
    cyton_thread.start()


if calibrate:
    calibration_phase()
np.random.seed(0)  
trials = generate_experiment_trials()
np.save('trials.npy', trials)
photosensor_dot = create_photosensor_dot()
photosensor_dot.color = np.array([-1, -1, -1])
photosensor_dot.draw()

for trial_num, trial_artifacts in enumerate(trials, start=1):
    trial_reminder(trial_num)  

    #define variables we will write into
    eeg = np.zeros((8, 0))
    aux = np.zeros((3, 0))
    timestamp = np.zeros((0))
        
    logging.log(level=logging.DATA, msg=f'Starting trial {trial_num}')
    for artifact in trial_artifacts:
        logging.log(level=logging.DATA, msg=f'Starting artifact: {artifact}')
        # marker = artifact_marker[artifact]
        present_cue(artifact)
        if test:
            write_data_file(trial_num)
        if not test and cyton_in:
            # get_data(queue_in, lsl_out)
            pass
    
    while not queue_in.empty(): # Collect all data from the queue
        eeg_in, aux_in, timestamp_in = queue_in.get()
        print('data-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
        eeg = np.concatenate((eeg, eeg_in), axis=1)
        aux = np.concatenate((aux, aux_in), axis=1)
        timestamp = np.concatenate((timestamp, timestamp_in), axis=0)
    print('total: ',eeg.shape, aux.shape, timestamp.shape)
    # Pre processing data -- fun stuff
    # photo_trigger = (aux[1] > 20).astype(int)
    # trial_starts = np.where(np.diff(photo_trigger) == 1)[0] 
    # trial_ends = np.where(np.diff(photo_trigger) == -1)[0]
    filename = f"trial_{trial_num}.npy"
    
    data = np.vstack((eeg, aux, timestamp)).T
    np.save(filename, data)
    # DataFilter.write_file(data, filename, 'a')

        # Display rest break
    simple_classifier(filename)
    rest_break = visual.TextStim(win, text="End of Trial. Rest Break", color="white", height=30, pos=(0, 150))
    fixation.draw()  
    rest_break.draw()
    win.flip()
    core.wait(10.0)

stop_event.set()   
board.stop_stream()
board.release_session()

end_experiment()
win.close()