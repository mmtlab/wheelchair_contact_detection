# -*- coding: utf-8 -*-

import time
import numpy as np
from . import sound

class Countdown:
    
    def __init__(self, seconds=10, outputUpdate=1, beepFreq=1000, beepDuration=0.5, 
                 incrDurationFlag=True, printOutFlag=True, audioOutFlag=True, start=True):
        
        self.seconds = seconds
        self.outputUpdate = outputUpdate 
        self.beepFreq = beepFreq
        self.beepDuration = beepDuration
        self.incrDurationFlag = incrDurationFlag
        self.printOutFlag = printOutFlag
        self.audioOutFlag = audioOutFlag
        
        if start:
            self.start()
    
    def start(self):
        if self.incrDurationFlag:
            beepDurations = np.linspace(0.1, 0.9, int(np.ceil(self.seconds/self.outputUpdate)))
        else:
            beepDurations = [self.beepDuration]*int(np.ceil(self.seconds/self.outputUpdate))
                                        
        for i, bd in zip(range(0, self.seconds, self.outputUpdate), beepDurations):
            if self.printOutFlag:
                print(self.seconds - i)
            if self.audioOutFlag:
                sound.playBeep(self.beepFreq, bd, blockExec = False)
            time.sleep(min(self.outputUpdate, self.seconds-i))
        
        if self.printOutFlag:
            print(0)
        if self.audioOutFlag:
            sound.playBeep(self.beepFreq/2, max(1,self.beepDuration*2), blockExec = False)
            
        
        
    