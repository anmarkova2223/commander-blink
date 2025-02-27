from psychopy import visual, core, logging
import random

# Create window
win = visual.Window([800, 600], color="black", units="pix")

# Create fixation cross and cue
fixation = visual.TextStim(win, text="+", color="white", height=50)
cue = visual.TextStim(win, text="Blink your eyes", color="white", height=30)

# List of artifacts
artifacts = ['eye_blink', 'jaw_clench', 'double_blink', 'jaw_clench_blink', 'blink_hard'] # Added a hard blink to artifacts

# Logging
logging.setDefaultClock(core.Clock())

# Fixation cross and cue
def present_fixation(duration):
    fixation.draw()
    win.flip()
    core.wait(duration)

# Cues and logging artifact data
def present_cue(artifact, duration=2.0):
    cue.setText(f"{artifact}")
    cue.draw()
    win.flip()
    logging.log(level=logging.DATA, msg=f'Artifact presented: {artifact}')
    core.wait(duration) 

# Running the experiment: 3 trials
for trial in range(3):
    logging.log(level=logging.DATA, msg=f'Starting trial {trial + 1}')
    
    # Randomize the order of artifacts, 6 times per artifact (total of 30 samples per trial)
    trial_artifacts = artifacts * 6 
    random.shuffle(trial_artifacts) 

    # Artifact in randomized order
    for artifact in trial_artifacts:
        logging.log(level=logging.DATA, msg=f'Starting artifact: {artifact}')
        
        # Fixation cross, then cue for the artifact
        present_fixation(1.0) 
        present_cue(artifact) 
            
    # Rest break between trials
    rest_break = visual.TextStim(win, text="End of Trial. Rest Break", color="white", height=30)
    rest_break.draw()
    win.flip()
    core.wait(10.0)

# Close window after trials
win.close()