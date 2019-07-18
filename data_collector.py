

import numpy as np
import os
import pygame
import scipy.misc

class DataCollector:

    def __init__(self,simulator,folder='record_data',max_saves=2000,buffer_max=50,save_size=500):

        self.prev = 0
        self.prev_time = pygame.time.get_ticks()

        self.save_cnt = 0
        self.data_buffer = []
        self.target_buffer = []
        self.save_cnt =0
        self.curr_file = 0
        
        self.folder=folder
        self.max_saves = max_saves
        self.buffer_max = buffer_max
        self.save_size = save_size
        self.load()
        self.max_reached = False

    def save_data(self,data,target,as_image=False):
        

        if self.save_cnt<self.max_saves:
        

            self.data_buffer.append(data)
            self.target_buffer.append(target)

            
        
            if len(self.data_buffer)==self.buffer_max:
                if not self.save_cnt%self.save_size and self.save_cnt!=0:
                    self.curr_file+=1
                files = os.listdir(f'{self.folder}')
                
                if f'data{self.curr_file}.npy' in files:
                    prev_data = np.load(f'{self.folder}/data{self.curr_file}.npy')
                    data = np.r_[prev_data,self.data_buffer]
                else:
                    data = np.array(self.data_buffer)

                if 'targets.npy' in files:
                    prev_targets = np.load(f'{self.folder}/targets.npy')
                    targets = np.r_[prev_targets,self.target_buffer]
                else:
                    targets = np.array(self.target_buffer)

                np.save(f'{self.folder}/data{self.curr_file}',data)
                np.save(f'{self.folder}/targets',targets)
                print(f"{self.folder}: Saved in data{self.curr_file} ,SaveCnt: {self.save_cnt}")
                f = open(f'{self.folder}/data.conf','w')
                f.write(f'{self.curr_file} {self.save_cnt}\n')
                f.close()
                self.data_buffer =[]
                self.target_buffer =[]
                self.save_cnt+=1
            else:
                print(f"{self.folder}: Recived Data: {self.buffer_max-len(self.data_buffer) } more calls to save!")
            
            


            


    def load(self):
        files = os.listdir(f'{self.folder}')
        if 'data.conf' in files:
            f = open(f'{self.folder}/data.conf')
            self.curr_file,self.save_cnt = [int(s) for s in f.read()[:-1].split()]
            f.close() 
            print("Found Conf, Curr_File:",self.curr_file,"Save_Cnt:",self.save_cnt)
            self.save_cnt+=1