#pylint: disable=no-member
import carla 
import pygame
import enum
import numpy as np
import drawing_library
import Simulator
import random
from queue import Queue

class GameManager:

    def __init__(self,simulator,resolution=(640,480)):
        self.initialize_pygame(resolution)
        self.simulator = simulator
        self.new_frame = False
        self.new_frame2 =False
        self.surface = None
        self.surface2 =None
        self.prev = pygame.time.get_ticks()
        self.draw_periodic = False
        # self.color_density = Density(simulator)
        self.curr_interval = 100
        self.started =False
        self.draw_prev = self.prev
        self.pixel_buffer  =PixelBuffer(simulator)
        self.saver_pixels = None
        self.started = False
        
    def initialize_pygame(self,resolution):
        pygame.init()
        self.display = pygame.display.set_mode(resolution,pygame.HWSURFACE | pygame.DOUBLEBUF)
    
    def render(self):
        if self.new_frame:
            self.display.blit(self.surface, (0, 0))
            self.new_frame =False 
        if self.new_frame2:
            self.display.blit(self.surface2, (0, 0))
            self.new_frame2 =False 
        pygame.display.flip()
    
    def update(self):
        self.keys = pygame.key.get_pressed()
        self.handle_events()
        curr = pygame.time.get_ticks()
        if (curr-self.prev)>self.curr_interval:
            if self.started:
                self.pixel_buffer.add_pixels(self.saver_pixels)
            self.prev = curr
            # if random.random()<0.5:
            #     self.simulator.lane_ai.request_new_lane(prefer_left=True)
            # else:
            #     self.simulator.lane_ai.request_new_lane(prefer_left=False)
            # self.prev = curr
            # self.curr_interval = self.curr_interval==20000 and 30000 or 20000
        self.draw_green_line()
        # self.get_density()

    
    def draw_green_line(self):
        curr = pygame.time.get_ticks()
        if (curr-self.draw_prev)>1000:
            if self.draw_periodic:
                drawing_library.draw_arrows(self.simulator.world.debug,[i.location for i in self.simulator.navigation_system.local_route],color=carla.Color(0,255,0),life_time=0.5)
            self.draw_prev=  curr

    def handle_events(self):
        for event in pygame.event.get():

            if event.type==pygame.QUIT:
                self.simulator.running =False

            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_o:
                    self.simulator.switch_input()

                if event.key==pygame.K_p:
                    self.draw_periodic = not self.draw_periodic
                    print("curr_pos is %d"%(self.simulator.navigation_system.curr_pos))
                
                if event.key==pygame.K_c:
                    self.simulator.camera_switch()


                if event.key ==pygame.K_q:
                    self.simulator.reward_system.status = Simulator.Status.COMPLETED
                
                if event.key==pygame.K_l:
                    self.simulator.lane_ai.lane_changer.check_new_lane(force=True)
                    

                # if event.key==pygame.K_UP:

                #     self.simulator.traffic_controller.env.ai.change_action_state(0)
                # if event.key==pygame.K_DOWN:
                #     self.simulator.traffic_controller.env.ai.change_action_state(1)
                #    self.simulator.lane_ai.lane_changer.check_new_lane(min_angle=150)
                   
            
    def print_waypoint(self,waypoint):
        print("transform :",waypoint.transform)
        print("lane width :",waypoint.lane_width)
        print("roadid :",waypoint.road_id)
        print("section id :",waypoint.section_id)
        print("lane id :",waypoint.lane_id)
        print("lane change :",waypoint.lane_change) 
        print("lane type :",waypoint.lane_type) 
        print("right mark :",waypoint.right_lane_marking) 
        print("left mark :",waypoint.left_lane_marking)

    def camera_callback(self,image):
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.new_frame =True
    
    def semantic_callback(self,image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        array = array.reshape(-1,3)
        self.array =array

        array = self.transform_array2(array)
        array  = np.reshape(array, (image.height//2, image.width, 3))
    
        self.surface2 = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        self.new_frame2 =True
        # self.color_density.add_density(self.array)
        self.started = True
        # print(self.color_density.get_offset())
        # print(self.color_density.buffer)

    def transform_array(self,array):
        array = np.c_[  np.ceil(np.mean(array,-1)) ]
        # array = array[200*67:200*167]
        self.started = True
        self.saver_pixels = array
        # print(array)
        array =np.repeat(array,3,1)
        return array

    def transform_array2(self,array):
        # t = np.where( array==[128, 64, 128], [255,255,255], array)
        # temp = np.where( t==[157, 234, 50], [255,255,255], t)
        array = array[200*67:200*167]
        temp =  np.where( array!=[0, 0, 142], [0,0,0], array)
        array = temp
        #  array[np.where(array==[128, 64, 128]) ] =  [0,0,0]
        # array[np.where(array==[157, 234, 50]) ] =  [0,0,0]
        
        # array[np.where( (array!=[128, 64, 128]) + (array!=[157, 234, 50]), [0,0,0], array)
        
        # array = array[64*256:,:]
        # print(array.shape)
        # print(array)
        array = self.transform_array(array)
        return array
        
    def get_density(self):
        if self.started:
            density = sum(np.all(self.array==[0,0,142],axis=1))
            
            if density>600:
                self.simulator.collision_vehicle = True
            else:
                self.simulator.collision_vehicle =False


class PixelBuffer:

    def __init__(self,simulator):
        self.simulator = simulator
        self.queue = np.zeros((10,100*200,1))
        self.curr_len = 0
        self.max_len =10

    def add_pixels(self,array):
        # print(self.curr_len)
        if self.curr_len==self.max_len:
            self.queue =  np.r_[ self.queue[1:] , [array]]
        else:
            self.queue[self.curr_len] = array
            self.curr_len+=1

    def get_pixels(self):
        if self.curr_len>0:
            index = self.curr_len-1
            self.curr_len-=1
            return True,self.queue[index]

        else:
            return False,None




    def make_density(self,array):
        density_road = sum(np.all(array==[128,64,128],axis=1)) +sum(np.all(array==[157, 234, 50],axis=1))
        density_car = sum(np.all(array==[0,0,142],axis=1))
        return density_road,density_car
    
    def get_offset(self):
        # print(len(self.buffer))
        if len(self.buffer)==self.max_len:
            return (self.buffer[59][0]-self.buffer[0][0]),(self.buffer[59][1]-self.buffer[0][1])
        else:
            return 0,0

