from psychopy import visual, core, logging, event
import random

experiment_running = True 

# Create window
win = visual.Window([800, 600], color="black", units="pix")

# Create fixation cross and cue
fixation = visual.TextStim(win, text="+", color="white", height=50)
cue = visual.TextStim(win, text="Blink your eyes", color="white", height=30)


# List of artifacts
artifacts = ['jaw_clench', 'double_blink', 'jaw_clench_blink', 'blink_hard']

# Logging
logging.setDefaultClock(core.Clock())

# Calibration
def calibration_phase(duration=15.0):
    instruction = visual.TextStim(win, 
                                  text="Fix your eyes. Watch the screen and blink naturally for 15s.", 
                                  color="white", height=30, pos=(0, 100))  

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
                               color="white", height=30)
    reminder.draw()
    win.flip()
    logging.log(level=logging.DATA, msg=f'Reminder before Trial {trial_num}')
    event.waitKeys()  #

# Fixation cross, present as soon as calibration ends
def present_fixation(duration):
    fixation.draw()
    win.flip()
    core.wait(duration)

# ends experiment, removes +
def end_experiment():
    experiment_running = False
    win.flip()

# Cues and logging artifact data
def present_cue(artifact, duration=2.0):
    cue.setText(f"{artifact}")
    cue.draw()
    win.flip()
    logging.log(level=logging.DATA, msg=f'Artifact presented: {artifact}')
    core.wait(duration) 

    cue.setText("Begin artifact")
    cue.draw()
    win.flip()
    logging.log(level=logging.DATA, msg=f'Action cue presented: {artifact}')
    core.wait(3.0)

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

# Run
def run_experiment():
    calibration_phase()  
    
    trials = generate_experiment_trials()

    for trial_num, trial_artifacts in enumerate(trials, start=1):
        trial_reminder(trial_num)  
        
        logging.log(level=logging.DATA, msg=f'Starting trial {trial_num}')
        present_fixation(2.0)
        for artifact in trial_artifacts:
            logging.log(level=logging.DATA, msg=f'Starting artifact: {artifact}')
            present_cue(artifact)

        rest_break = visual.TextStim(win, text="End of Trial. Rest Break", color="white", height=30)
        rest_break.draw()
        win.flip()
        core.wait(10.0)

    end_experiment()

    win.close()

run_experiment()