from psychopy import visual, core, logging, event
import random
import serial

import time

import numpy as np
import pandas as pd

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
import sys
import glob

experiment_running = True
cyton_in = True 

# Create window
win = visual.Window([800, 600], color="black", units="pix")

# Create fixation cross and cue
fixation = visual.TextStim(win, text="+", color="white", height=50, pos=(0, 0))  
cue = visual.TextStim(win, text="", color="white", height=30, pos=(0, 0))  

# List of artifacts
artifacts = ['jaw_clench', 'double_blink', 'jaw_clench_blink', 'blink_hard']

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
def present_cue(artifact, duration=2.0):
    #fixation.draw()
    cue.setText(f"{artifact}")
    cue.draw()
    win.flip()
    logging.log(level=logging.DATA, msg=f'Artifact cue presented: {artifact}')
    core.wait(duration) 

    # Display fixation and "Begin artifact" message together
    fixation.draw()
    #cue.setText("Begin artifact")
    #cue.draw()
    win.flip()
    logging.log(level=logging.DATA, msg=f'Action cue presented: {artifact}')
    core.wait(1.5)

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


def write_data_file():
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = find_openbci_port()
    board = BoardShim(CYTON_BOARD_ID, params)

    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    time.sleep(10)
    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    # demo how to convert it to pandas DF and plot data
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
    df = pd.DataFrame(np.transpose(data))

    # demo for data serialization using brainflow API, we recommend to use it instead pandas.to_csv()
    filename = f"trial_{trial_num}_{artifact}.csv"
    DataFilter.write_file(data, filename, 'w')  # use 'a' for append mode


# Simon's code for Brainflow + getting parameters
if cyton_in: 
    CYTON_BOARD_ID = 0 
    BAUD_RATE = 115200

    #Finds the port to which the Cyton Dongle is connected to.
def find_openbci_port(): 
    # Find serial port names per OS
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        ports = glob.glob('/dev/ttyUSB*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/cu.usbserial*')
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
        

# #Responsible for collecting EEG data from the OpenBCI Cyton board 
# def get_data(queue_in, lsl_out=False):
#     while not stop_event.is_set():
#         data_in = board.get_board_data()
#         timestamp_in = data_in[board.get_timestamp_channel(CYTON_BOARD_ID)]
#         eeg_in = data_in[board.get_eeg_channels(CYTON_BOARD_ID)]
#         aux_in = data_in[board.get_analog_channels(CYTON_BOARD_ID)]
#         if len(timestamp_in) > 0:
#             print('queue-in: ', eeg_in.shape, aux_in.shape, timestamp_in.shape)
#             queue_in.put((eeg_in, aux_in, timestamp_in))
#         time.sleep(0.1)
    
#     queue_in = Queue()
#     cyton_thread = Thread(target=get_data, args=(queue_in, lsl_out))
#     cyton_thread.daemon = True
#     cyton_thread.start()

#     if os.path.exists(model_file_path):
#         with open(model_file_path, 'rb') as f:
#             model = pickle.load(f)
#     else:
#         model = None


# Performing the Experiment
params = BrainFlowInputParams()  #this sets up the connection Parameters
    
if CYTON_BOARD_ID != 6:
    params.serial_port = find_openbci_port()
elif CYTON_BOARD_ID == 6:
    params.ip_port = 9000
board = BoardShim(CYTON_BOARD_ID, params)
board.prepare_session()
res_query = board.config_board('/0')
res_query = board.config_board('//')
res_query = board.config_board(ANALOGUE_MODE)
board.start_stream(45000)

calibration_phase()  
trials = generate_experiment_trials()

for trial_num, trial_artifacts in enumerate(trials, start=1):
    trial_reminder(trial_num)  
        
    logging.log(level=logging.DATA, msg=f'Starting trial {trial_num}')
    for artifact in trial_artifacts:
        logging.log(level=logging.DATA, msg=f'Starting artifact: {artifact}')
        present_cue(artifact)
        write_data_file()  

        # Display rest break
    rest_break = visual.TextStim(win, text="End of Trial. Rest Break", color="white", height=30, pos=(0, 150))
    fixation.draw()  
    rest_break.draw()
    win.flip()
    core.wait(10.0)

board.stop_stream()
board.release_session()

end_experiment()
win.close()