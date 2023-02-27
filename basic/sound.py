import winsound
import time
import vlc
import threading

MAX_FREQ = 32767
MIN_FREQ = 32

def __playBeep(freq = 1000, duration = 0.5):
    '''
    DEPRECTED, USE playBeep() instead that allows to decide if continue code execution while playing sound
    
    Plays a sound at the given frequency for duration s

    Parameters
    ----------
    freq : TYPE, optional
        DESCRIPTION. The default is 1000.
    duration : float, optional
        duration in seconds of beep execution. The default is 0.5.

    Returns
    -------
    None.

    '''
    # frequency values must be between 37 and 32767
    freq = max(min(freq,MAX_FREQ),MIN_FREQ)
    freq = int(freq)
    winsound.Beep(freq, int(duration*1000))
    time.sleep(0.01)
    
def playBeep(freq = 1000, duration = 0.5, blockExec = False):
    if blockExec: # execute the function
        __playBeep(freq, duration)
    else: # create a thread and execute the function
        thread = threading.Thread(target=__playBeep, args = (freq,duration,))
        thread.start()
    

def __playFreq(startFreq = 5000, endFreq = 0, stepFreq = -500, duration = 0.5):
    '''
    DEPRECATED, USE playFreq() instead that allows to decide if continue code execution while playing sound
    
    Plays a sound in the range of frequency from startFreq to endFreq with step
    stepFreq, each one reproduced for duration s.
    
    NB: frequency values must be between 37 and 32767

    Parameters
    ----------
    startFreq : int, optional
        frequency of the first freq played. The default is 5000.
    endFreq : int, optional
        frequency of the last freq played. The default is 0.
    stepFreq : int, optional
        step of frequencies in the loop. The default is -500.
    duration : int, optional
        duration of each freq in s. The default is 0.5.

    Returns
    -------
    None.

    '''
    # frequency values must be between 37 and 32767
    startFreq = max(min(startFreq,MAX_FREQ),MIN_FREQ)
    endFreq = max(min(endFreq,MAX_FREQ),MIN_FREQ)
    startFreq = int(startFreq)
    endFreq = int(endFreq)
    
    for freq in range(startFreq, endFreq, stepFreq):
        playBeep(freq, duration, blockExec = True)
        time.sleep(0.01)

def playFreq(startFreq = 5000, endFreq = 0, stepFreq = -500, duration = 0.5, blockExec = False):
    '''
    Plays a sound in the range of frequency from startFreq to endFreq with step
    stepFreq, each one reproduced for duration s.

    use blockExec = True to pause the code until the sound is played, otherwise code execution will continue while the sound is played
    
    NB: frequency values will be put in the range between 37 and 32767

    Parameters
    ----------
    startFreq : int, optional
        frequency of the first freq played. The default is 5000.
    endFreq : int, optional
        frequency of the last freq played. The default is 0.
    stepFreq : int, optional
        step of frequencies in the loop. The default is -500.
    duration : int, optional
        duration of each freq in s. The default is 0.5.
    blockExec : bool, optional
        if True, blocks the execution of the program waiting for the sound to 
        be played. The default is False.

    Returns
    -------
    None.

    '''
    if blockExec: # execute the function
        __playFreq(startFreq, endFreq, stepFreq, duration)
    else: # create a thread and execute the function
        thread = threading.Thread(target=__playFreq, args = (startFreq,endFreq,stepFreq,duration,))
        thread.start()

def __playFile(source, duration = -1):
    '''
    DEPRECATED, USE playFile() instead that allows to decide if continue code execution while playing sound
    Plays the media in the given path and pauses the execution till 
        - the end of playing if duration is not specified
        - duration [s] if it is specified

    Parameters
    ----------
    source : string
        path to the media.
    duration : float, optional
        for how many seconds should the media be played before being stopped? 
        The default is -1, which plays the media for the whole duration. 

    Returns
    -------
    None.

    '''
    # creating a vlc instance
    vlc_instance = vlc.Instance()
    # creating a media player
    player = vlc_instance.media_player_new()
    # creating a media
    media = vlc_instance.media_new(source)
    # setting media to the player
    player.set_media(media)
    # play the video
    player.play()
    time.sleep(0.1) # necessary to give time to the player to start
    if duration == -1:
        # getting the duration of the audio in s
        duration = player.get_length() / 1000
    # wait time
    time.sleep(duration)
    # stop the player
    player.stop()

def playFile(source, duration = -1, blockExec = False):
    '''
    Plays the audio of the file in the given path until
        - the end of playing if duration is not specified
        - duration [s] if it is specified

    use blockExec = True to pause the code until the sound is played, otherwise code execution will continue while the sound is played
    
    Parameters
    ----------
    source : string
        path to the media.
    duration : float, optional
        for how many seconds should the media be played before being stopped 
        and continuing the execution? 
        The default is -1, which plays the media for the whole duration.
    blockExec : bool, optional
        if True, blocks the execution of the program waiting for the sound to 
        be played. The default is False.

    Returns
    -------
    None.

    '''
    if blockExec: # execute the function
        duration = __playFile(source, duration)
    else: # create a thread and execute the function
        thread = threading.Thread(target=__playFile, args = (source, duration, ))
        thread.start()

if __name__ == '__main__':

    filePath = r'C:\Users\eferlius\Downloads\tiStaShort.mp3'
    filePath = r'C:\Users\eferlius\Downloads\hoFinito.m4a'

    playFile(filePath, duration = -1)
    for i in range(10):
        time.sleep(1)
        print('no block ' + str(i))

    print('now the sound')
    # with blocking
    playFile(filePath, duration = 0.7, blockExec = True)
    for i in range(10):
        time.sleep(0.01)
        print('block ' + str(i))

    time.sleep(2)
    for i in range(20):
        time.sleep(0.01)
        print('exec ' + str(i))
        playFile(filePath, duration = 0.7, blockExec = False)
    
    playFile(filePath, duration = -1)
