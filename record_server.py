
import data_collector
import pygame
import numpy as np
from enum import Enum
class State(Enum):
    RECORDING=1
    READY=2


class RecordServer:

    def __init__(self,simulator):

        self.simulator = simulator
        self.state  = State.READY
        self.collector = data_collector.DataCollector('record_data',100,10,50,debug=False)
        self.waiting  = False
        simulator.client.start_recorder("recording.log")
        self.start_time = pygame.time.get_ticks()
        self.enabled = True
    def start_recording(self):
        curr =pygame.time.get_ticks()
        if self.state==State.READY and not self.waiting and self.enabled and (curr-self.start_time)<1000000:
            self.state = State.RECORDING
            self.start_time = np.int64(min(0,pygame.time.get_ticks()-2000)/1000)

    def stop_recording(self,type_):

        if self.state==State.RECORDING:
            self.state = State.READY
            stop_time = np.int64(min(0,pygame.time.get_ticks()+2000)/1000)
            self.collector.save_data([self.start_time,stop_time],type_)
            self.wait_start = pygame.time.get_ticks()
            self.waiting  =True

    def update(self):

        if self.waiting:
            curr =pygame.time.get_ticks()

            if (curr-self.wait_start)>6000:
                self.waiting = False
            
    
    


