import time
from controller import control 

def main():
    print("Go to youtube")
    time.sleep(3)
    print("Pressing k")
    control('play_pause')
    print('All done')

if __name__ == '__main__':
    main()