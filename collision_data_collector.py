

import numpy as np
import os
import pygame

class DataCollector:

    def __init__(self,simulator):
        self.vehicle_variables = simulator.vehicle_variables
        self.map = simulator.map
        self.prev = 0
        self.prev_time = pygame.time.get_ticks()
        self.game_manager = simulator.game_manager
        self.save_cnt = 0
        self.data_buffer = []
        self.target_buffer = []
        self.save_cnt =0
        self.curr_file = 0
        self.load()
    


    def save_image(self,data,target,as_image=False):
        
        

        if as_image:
            # if t==0:
            #     f_name = f'slow{self.id_slow}.png'
            #     self.id_slow+=1
            # elif t==1:
            #     f_name = f'go{self.id_go}.png'
            #     self.id_go+=1
            # elif t==2:
            #     f_name = f'stop{self.id_stop}.png'
        
            # self.id_stop+=1
            
            # try:
            #     pygame.image.save(self.game_manager.surface2, os.path.join('images',f_name))
            # except Exception as e:
            #     print(e)
                pass
        
        else:
            print("here")
            if self.save_cnt<2000:

                self.data_buffer.append(data)
                self.target_buffer.append(target)

                
            
                if len(self.data_buffer)==50:
                    if not self.save_cnt%500 and self.save_cnt!=0:
                        self.curr_file+=1
                    files = os.listdir('collision_data')
                    
                    if f'data{self.curr_file}.npy' in files:
                        prev_data = np.load(f'collision_data/data{self.curr_file}.npy')
                        data = np.r_[prev_data,self.data_buffer]
                    else:
                        data = np.array(self.data_buffer)

                    if 'targets.npy' in files:
                        prev_targets = np.load('collision_data/targets.npy')
                        targets = np.r_[prev_targets,self.target_buffer]
                    else:
                        targets = np.array(self.target_buffer)

                    np.save(f'collision_data/data{self.curr_file}',data)
                    np.save('collision_data/targets',targets)
                    print(f"Saved in data{self.curr_file} ,SaveCnt: {self.save_cnt}")
                    f = open('collision_data/data.conf','w')
                    f.write(f'{self.curr_file} {self.save_cnt}\n')
                    f.close()
                    self.data_buffer =[]
                    self.target_buffer =[]
                    self.save_cnt+=1
                else:
                    print(f"Recived Data: {50-len(self.data_buffer) } more calls to save!")

    def load(self):
        files = os.listdir('images')
        if 'data.conf' in files:
            f = open('images/data.conf')
            self.curr_file,self.save_cnt = [int(s) for s in f.read()[:-1].split()]
            f.close() 
            print("Found Conf, Curr_File:",self.curr_file,"Save_Cnt:",self.save_cnt)
            self.save_cnt+=1