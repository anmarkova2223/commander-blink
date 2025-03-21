from pynput.keyboard import Key, Controller
keyboard = Controller()


# key/key-combinations for each button ID in youtube remote mode
youtube_commands = {
    'shortcuts': {
        'play_pause': 'k',
        'yt_forward': 'l',
        'yt_rewind': 'j',
        'yt_captions': 'c',
        'yt_fullscreen': 'f'
    },
    'shift': {
        'playlist_previous': 'p',
        'playlist_next': 'n',
    },
    'media': {
        'volume_down': Key.media_volume_down,
        'volume_up': Key.media_volume_up,
    }
}

def control(key):

    """
    args:
    mode: 'YOUTUBE' or 'NORMAL'
    key: button_id which maps to its corresponding pynput Key
    """

    if key in youtube_commands['shortcuts'].keys():
        keyboard.type(youtube_commands['shortcuts'][key])

    elif key in youtube_commands['shift'].keys():
        with keyboard.pressed(Key.shift):
            keyboard.type(youtube_commands['shift'][key])

    elif key in youtube_commands['media'].keys():
        keyboard.press(youtube_commands['media'][key])
        keyboard.release(youtube_commands['media'][key])