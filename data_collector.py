import carla 
import numpy as np
import navigation_system
import pygame
import game_manager
import vehicle_controller
import control_manager
import sensor_manager
import reward_system
import drawing_library
import math
from enum import Enum
import weakref
import  random
import lane_ai
import traffic_controller
from agents.navigation import basic_agent
import os

class DataCollector:

    def __init__(self,simulator):
        self.vehicle_variables = simulator.vehicle_variables
        self.map = simulator.map
        self.prev = 0
        self.no_signal_get = 0
        self.intersection_get = 0
        self.on_intersection = False
        self.prev_time = pygame.time.get_ticks()
        self.game_manager = simulator.game_manager
        self.id_slow = 0
        self.id_go  =0
        self.id_stop =0
        self.save_cnt = 0
        self.pixel_buffer = []
        self.class_buffer = []
        self.save_cnt =0
        self.curr_file = 0
        self.load()
    
    def get_deviations(self,a):
        s = np.std(a)
        # print(s)
        if s>10:
            return True
        else:
            return False


    def update(self):
        d =0
        curr_waypoint = self.vehicle_variables.vehicle_waypoint
        
        if not curr_waypoint.is_intersection:
            self.on_intersection = False
            found  = False
            prev_data = [navigation_system.NavigationSystem.transform_angle(curr_waypoint.transform.rotation.yaw%360)]
            while d<13:
                waypoints = curr_waypoint.next(1)
                
                if waypoints:
                    curr_waypoint = waypoints[0]
                    prev_data.append( navigation_system.NavigationSystem.transform_angle(curr_waypoint.transform.rotation.yaw%360) )
                    d+=1
                    if curr_waypoint.is_intersection:  
                        self.junction_found(d)
                        found = True
                        break
                else:
                    break
            if not found:
                self.no_signal()

        else:
            pass
            # self.check_intersection()

    def check_intersection(self):
        
        if not self.on_intersection:
            self.on_intersection = True
            print("Take Intersection Image")
            self.save_image(1)

    def no_signal(self):
        curr_time = pygame.time.get_ticks()
        if (curr_time-self.prev_time)>1300:
            if self.no_signal_get>1:
                print( "Image Taken",self.no_signal_get)
                print("Take Normal Image")
                self.save_image(1)
                self.no_signal_get-=1

            self.prev_time =curr_time

    def junction_found(self,distance):
        
       if abs(self.prev-distance)>1:

            # if 3<distance<6:
            #     self.no_signal_get+=1
            #     print(f"Stop Signal: {distance}m\n")
            #     self.save_image(2)
                
            if 11>distance>7:
                self.no_signal_get+=1
                print(f"Slow Signal: {distance}m")
                self.save_image(0)

            self.prev = distance
    
    def save_image(self,t,as_image=False):
       
        if self.game_manager.started:
            if as_image:
                if t==0:
                    f_name = f'slow{self.id_slow}.png'
                    self.id_slow+=1
                elif t==1:
                    f_name = f'go{self.id_go}.png'
                    self.id_go+=1
                elif t==2:
                    f_name = f'stop{self.id_stop}.png'
            
                self.id_stop+=1
                
                try:
                    pygame.image.save(self.game_manager.surface2, os.path.join('images',f_name))
                except Exception as e:
                    print(e)
                    pass
            
            else:
                if self.save_cnt<400:
                    
                    pixels = self.game_manager.saver_pixels.reshape(-1)
                    self.pixel_buffer.append(pixels)
                    self.class_buffer.append(t)

                    
                    
                    if len(self.pixel_buffer)==50:
                        if not self.save_cnt%25 and self.save_cnt!=0:
                            self.curr_file+=1
                        files = os.listdir('images')
                        
                        if f'pixels{self.curr_file}.npy' in files:
                            prev_pixels = np.load(f'images/pixels{self.curr_file}.npy')
                            pixels = np.r_[prev_pixels,self.pixel_buffer]
                        else:
                            pixels = np.array(self.pixel_buffer)

                        if 'classes.npy' in files:
                            prev_classes = np.load('images/classes.npy')
                            classes = np.r_[prev_classes,self.class_buffer]
                        else:
                            classes = np.array(self.class_buffer)

                        np.save(f'images/pixels{self.curr_file}',pixels)
                        np.save('images/classes',classes)
                        print(f"Saved in pixels{self.curr_file} ,SaveCnt: {self.save_cnt}")
                        f = open('images/data.conf','w')
                        f.write(f'{self.curr_file} {self.save_cnt}\n')
                        f.close()
                        self.pixel_buffer =[]
                        self.class_buffer =[]
                        self.save_cnt+=1
                    

    def load(self):
        files = os.listdir('images')
        if 'data.conf' in files:
            f = open('images/data.conf')
            self.curr_file,self.save_cnt = [int(s) for s in f.read()[:-1].split()]
            f.close() 
            print("Found Conf, Curr_File:",self.curr_file,"Save_Cnt:",self.save_cnt)
            self.save_cnt+=1

class CollisionCollector:

    def __init__(self,simulator):
        self.simulator = simulator
        self.game_manager = simulator.game_manager
        self.normal_cnt = 0
        self.path = 'images/collision'
        self.id_avoid = 0
        self.id_go =0
        self.save_cnt = 0
        self.pixel_buffer = []
        self.class_buffer = []
        self.normal_cnt = 0
        self.prev = pygame.time.get_ticks()
        self.call_cnt = 0
        self.curr_file = 0
        self.last_normal =False
        self.buffer_size = 10
        self.cool_down = False
        self.cool_down_prev = pygame.time.get_ticks()

    def collect_normal(self,t):
        if self.normal_cnt and self.game_manager.started:
            # if t==0:
                #  print("Normal Away")
            # else:
                #  print("Normal Range")
            # print(f"Normal Call, PixelBuffer:{len(self.pixel_buffer)}, ClassBuffer:{len(self.class_buffer)}")
            pixels = self.game_manager.saver_pixels
            self.save_images(0,pixels)
            self.normal_cnt-=1
            

    def collect_pixels(self):
        pixels = []
        self.game_manager.pixel_buffer.get_pixels()
        for i in range(4):
            avail,pixel = self.game_manager.pixel_buffer.get_pixels()
            if avail:
                self.normal_cnt+=1
                pixels.append(pixel)
            else:
                break
        
        return pixels
        
    def save_images(self,t,pixels,as_image=False):
       
        if self.game_manager.started:
            if as_image:
                if t==0:
                    f_name = f'avoid{self.id_avoid}.png'
                    self.id_avoid+=1
                elif t==1:
                    f_name = f'go{self.id_go}.png'
                    self.id_go+=1
                
                
                try:
                    pygame.image.save(self.game_manager.surface2, os.path.join(self.path,f_name))
                except Exception as e:
                    print(e)
            
            else:
                if self.save_cnt<400:
                    extras = False
                    if t==1:
                        self.cool_down = True
                        self.cool_down_prev = pygame.time.get_ticks()
                        self.pixel_buffer+=pixels
                        self.class_buffer+= [t]*len(pixels)
                        
                        if len(self.pixel_buffer)>self.buffer_size:
                            self.pixel_buffer,extra_pixels = self.pixel_buffer[:self.buffer_size],self.pixel_buffer[self.buffer_size:]
                            self.classes,extra_classes = self.class_buffer[:self.buffer_size],self.class_buffer[self.buffer_size:]
                            extras = True
                        
                    else:
                        self.pixel_buffer.append(pixels)
                        self.class_buffer.append(t)
                
                    if len(self.pixel_buffer)==self.buffer_size:
                        if not self.save_cnt%25 and self.save_cnt!=0:
                            self.curr_file+=1
                        files = os.listdir(self.path)
                    
                        if f'pixels{self.curr_file}.npy' in files:
                            prev_pixels = np.load( os.path.join(self.path,f'pixels{self.curr_file}.npy'))
                            pixels = np.r_[prev_pixels,self.pixel_buffer]
                        else:
                            pixels = np.array(self.pixel_buffer)

                        if 'classes.npy' in files:
                            prev_classes = np.load( os.path.join(self.path,'classes.npy'))
                            classes = np.r_[prev_classes,self.class_buffer]
                        else:
                            classes = np.array(self.class_buffer)

                        np.save( os.path.join(self.path,f'pixels{self.curr_file}'),pixels)
                        np.save(os.path.join(self.path,'classes'),classes)
                        print(f"Saved in pixels{self.curr_file} ,SaveCnt: {self.save_cnt}")

                        f = open( os.path.join(self.path,'data.conf'),'w')
                        f.write(f'{self.curr_file} {self.save_cnt}\n')
                        f.close()

                        if extras:
                            self.pixel_buffer = extra_pixels
                            self.class_buffer = extra_classes
                        else:
                            self.pixel_buffer =[]
                            self.class_buffer =[]

                        self.save_cnt+=1
                
    def on_collide(self):

        pixels = self.collect_pixels()
        self.save_images(1,pixels)
        self.simulator.collide_callback()
                    
    def update(self):
        curr = pygame.time.get_ticks()
        vehicle_loc = self.simulator.vehicle_variables.vehicle_location
        locs = self.simulator.traffic_controller.curr_locations
        min_d = 10000
        for l in locs:
            d = navigation_system.NavigationSystem.get_distance(l,vehicle_loc,res=1)
            if d<min_d:
                min_d = d

        if not self.cool_down:
            if (curr-self.prev)>1000:
                if self.last_normal:
                    if min_d<60:
                        self.last_normal = False
                        # print("Normal Away")
                        self.collect_normal(0)
                
                else:
                   
                    self.last_normal = True
                    self.collect_normal(1)
                    # print("Normal Range")
                self.prev = curr
        else:
            if (curr-self.cool_down_prev)>1500:
                self.cool_down = False

            
    def load(self):
        files = os.listdir('images')
        if 'data.conf' in files:
            f = open('images/data.conf')
            self.curr_file,self.save_cnt = [int(s) for s in f.read()[:-1].split()]
            f.close() 
            print("Found Conf, Curr_File:",self.curr_file,"Save_Cnt:",self.save_cnt)
            self.save_cnt+=1


