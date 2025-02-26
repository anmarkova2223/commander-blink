from psychopy import visual, core, logging, event
import random

import time

import numpy as np
import pandas as pd

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter

experiment_running = True 

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

def main():
    BoardShim.enable_dev_board_logger()

    # use synthetic board for demo
    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
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
    DataFilter.write_file(data, 'test.csv', 'w')  # use 'a' for append mode
    restored_data = DataFilter.read_file('test.csv')
    restored_df = pd.DataFrame(np.transpose(restored_data))


if __name__ == "__main__":
    main()

# Run
def run_experiment():
    calibration_phase()  
    trials = generate_experiment_trials()

    for trial_num, trial_artifacts in enumerate(trials, start=1):
        trial_reminder(trial_num)  
        
        logging.log(level=logging.DATA, msg=f'Starting trial {trial_num}')
        for artifact in trial_artifacts:
            logging.log(level=logging.DATA, msg=f'Starting artifact: {artifact}')
            present_cue(artifact)  

        # Display rest break
        rest_break = visual.TextStim(win, text="End of Trial. Rest Break", color="white", height=30, pos=(0, 150))
        fixation.draw()  
        rest_break.draw()
        win.flip()
        core.wait(10.0)

    end_experiment()
    win.close()


run_experiment()
