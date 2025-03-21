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
import json
from controller import control
from model import model 
import webbrowser


cyton_in = True
test = False
lsl_out = False
fs = 250

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

    #add something to queue


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


def window_model(queue_in, lsl_out=False):
    print('model thread')
    eeg = np.zeros((8, 0))
    i = 0
    while not stop_event.is_set():
        if not queue_in.empty() and eeg.shape[1] <= int(fs*1.5):
            eeg_in, aux_in, timestamp_in = queue_in.get()
            eeg = np.concatenate((eeg, eeg_in), axis=1)
            print('eeg: ', eeg.shape)
        if eeg.shape[1] >= int(fs*1.5):
            res = model(eeg[3], i)
            i += 1
            print(res)
            if res == True:
                control('play_pause')
                eeg = np.zeros((8, 0))
            else:
                eeg = eeg[:, int(fs*0.5):]
    return None

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
        # params.serial_port = ''
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
    model_thread = Thread(target=window_model, args=(queue_in, lsl_out))
    cyton_thread.daemon = True
    model_thread.daemon = True
    cyton_thread.start()
    model_thread.start()

#open youtube
webbrowser.open('https://www.youtube.com/watch?v=9yjZpBq1XBE&t=0s')
control('yt_fullscreen')


while not stop_event.is_set():
    try:
        stop_event.is_set()
    except KeyboardInterrupt:
        stop_event.set() #this stops the data collection

        # Wait for the thread to finish
        cyton_thread.join()
        model_thread.join()

        board.stop_stream()
        board.release_session()