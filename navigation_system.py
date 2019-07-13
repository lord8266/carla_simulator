# pylint: disable=no-member
# pylint: disable=import-error

import carla 
from agents.navigation import global_route_planner,global_route_planner_dao
from agents.tools import misc
import math
import pygame
import Simulator
import numpy as np
from sklearn.preprocessing import StandardScaler
import drawing_library
import lane_ai

class NavigationSystem:
    def __init__(self,simulator):
        self.simulator = simulator
        self.spawn_points = self.simulator.map.get_spawn_points()
        
        self.drawn_point =False
        self.prev = pygame.time.get_ticks()
        self.dynamic_path  = None
        self.event_buffer = []
        self.re_route_ = False

    def add_event(self,prev_waypoints, next_waypoint):
        self.event_buffer.append( ( prev_waypoints,next_waypoint))
        self.make_parallel( prev_waypoints, next_waypoint)

    def make_map_data(self,res=3):
        self.map_data = global_route_planner_dao.GlobalRoutePlannerDAO(self.simulator.map,res)
        self.route_planner = global_route_planner.GlobalRoutePlanner(self.map_data)
        self.route_planner.setup()
    
    def make_ideal_route(self,start_index,destination_index):
        self.start = self.spawn_points[start_index]
        self.destination = self.spawn_points[destination_index]
        self.ideal_route_waypoints = [ w[0] for w in self.route_planner.trace_route(self.start.location, self.destination.location)]
        self.ideal_route = [w.transform for w in self.ideal_route_waypoints]
        self.clean_route()
        self.fill_gaps()
        self.clean_back()
        self.curr_pos = 0
        self.prev_pos = None
        self.destination_index = len(self.ideal_route)-1
        print(len(self.ideal_route))
        self.write_data()
    
    def make_ideal_route_r(self,start,end):
        self.start = start
        self.destination = end
        self.ideal_route_waypoints = [ w[0] for w in self.route_planner.trace_route(self.start.location, self.destination.location)]
        self.ideal_route = [w.transform for w in self.ideal_route_waypoints]
        self.clean_route()
        self.fill_gaps()
        self.clean_back()
        self.curr_pos = 0
        self.prev_pos = None
        self.destination_index = len(self.ideal_route)-1
        # print(len(self.ideal_route))
        self.write_data()
    
    def make_parallel(self,prev_waypoints,start_waypoint,max_lane=100,width=2.5):
        # print("make parallel")
        parallel_lane =  prev_waypoints+[start_waypoint]
        road_id,lane_id = start_waypoint.road_id,start_waypoint.lane_id

        while 1:
            n = parallel_lane[-1].next(width)
            if n:
                wp = n[0]
                if wp.road_id==road_id and wp.lane_id==lane_id:
                    parallel_lane.append(wp)
                else:
                    break
            else:
                break

        start_waypoint_current = self.ideal_route_waypoints[self.curr_pos]
        road_id,lane_id = start_waypoint_current.road_id,start_waypoint_current.lane_id
        curr_pos = self.curr_pos
        while curr_pos<(len(self.ideal_route_waypoints)-1):
            
            # print(curr_pos)
            start_waypoint_current  = self.ideal_route_waypoints[curr_pos]
            if start_waypoint_current.road_id!=road_id and start_waypoint_current.lane_id!=lane_id:
                break
            curr_pos+=1
        self.curr_pos = min(len(self.ideal_route_waypoints)-1,curr_pos) 
        self.start = parallel_lane[0].transform
        self.ideal_route_waypoints = parallel_lane+ self.ideal_route_waypoints[self.curr_pos:]
        self.ideal_route = [w.transform for w in parallel_lane] + self.ideal_route[self.curr_pos:]
        # print("made it here")
        drawing_library.draw_arrows(self.simulator.world.debug,[i.location for i in self.ideal_route][:15],life_time=3)
        self.clean_route()
        self.fill_gaps()
        self.clean_back()
        self.curr_pos = 0
        self.prev_pos = None
        self.destination_index = len(self.ideal_route)-1
        self.simulator.reward_system.prev_pos = 0
        self.make_local_route()


    def write_data(self):

        f1=open('data_loc.txt','w')
        f2 =open('data_rot.txt','w')
        for i in self.ideal_route:
            f1.write(f'{i.location.x} {i.location.y}\n')
            f2.write(f'{i.rotation.yaw%360}\n')
        
    def make_local_route(self):
        if self.simulator.vehicle_variables.wait_for_lag:
            self.curr_pos =0
        else:
            i = self.curr_pos
            vehicle_location = self.simulator.vehicle_variables.vehicle_location
            vehicle_transform = self.simulator.vehicle_variables.vehicle_transform
            prev_len = None
            prints = []
            while i<len(self.ideal_route):
                loc = self.ideal_route[i].location
                prints.append(i)
                curr_len = NavigationSystem.get_distance(loc,vehicle_location)
            
                if prev_len!=None and curr_len>prev_len:
                    break
                prev_len = curr_len
                i+=1
            i-=1
        
            if i<(len(self.ideal_route) -1):
                i = NavigationSystem.check_behind_i(vehicle_transform,self.ideal_route[i],self.ideal_route[i+1],i)

            # if i<(len(self.ideal_route) -1):
            #     i = NavigationSystem.check_angle(self.ideal_route,self.simulator.vehicle_variables,i)
            self.curr_pos = min(i,len(self.ideal_route)-1)
        # print("Choosing ",i)
        self.local_route = [self.simulator.vehicle_variables.vehicle_transform]+self.ideal_route[self.curr_pos:self.curr_pos+4]
        self.local_route_waypoints = [self.simulator.vehicle_variables.vehicle_waypoint]+self.ideal_route_waypoints[self.curr_pos:self.curr_pos+4]

        if len(self.local_route)<5:
            add = 5-len(self.local_route)
            self.local_route = self.local_route + [self.local_route[-1]]*add
            self.local_route_waypoints = self.local_route_waypoints + [self.local_route_waypoints[-1]]*add
        # print("choosing %d\n"%(self.curr_pos))
        # self.fill_local_route_gaps()

    def fill_local_route_gaps(self):

        for i in range(len(self.local_route)-1):
            p1 = self.local_route[i].location
            p2  = self.local_route[i+1].location
            if NavigationSystem.get_distance(p1,p2,res=1)>20:
                self.simulator.lane_ai.lane_changer.state = lane_ai.State.RUNNING
                self.re_route()

    def re_route(self):
        loc_start = self.simulator.vehicle_variables.vehicle_waypoint.transform
        loc_end = self.destination
        self.make_ideal_route_r(loc_start,loc_end)
        self.re_route_ = True
        self.simulator.reward_system.prev_pos = 0


    @staticmethod
    def check_angle(ideal_route,variables,i):
        p1 = ideal_route[i].location
        
        vp = variables.vehicle_location
        vy = NavigationSystem.transform_angle(variables.vehicle_yaw)
        y1 = p1.y-vp.y + np.finfo(float).eps 
        x1 = p1.x-vp.x +np.finfo(float).eps 
        angle1 =  NavigationSystem.transform_angle(math.degrees(np.arctan2(y1,x1)))
        offset1 = abs(NavigationSystem.transform_angle(angle1-vy))
        distance1 = NavigationSystem.get_distance(p1,vp,res=1)
        start = i+1
        if offset1>60 or distance1<0.35:
            while start<(i+4):
                p2 = ideal_route[start].location
                y2 = p2.y-vp.y + np.finfo(float).eps 
                x2 = p2.x-vp.x +np.finfo(float).eps 
                angle2 =  NavigationSystem.transform_angle(math.degrees(np.arctan2(y2,x2)))
                offset2 = abs(NavigationSystem.transform_angle( (angle2-vy)%360 ))
                distance2 = NavigationSystem.get_distance(p2,vp,res=1)
                if offset2<60 or distance2<0.5:
                    break
                else:
                    start+=1
        return start
    
    def get_rot_offset(self): # needs to change
        vehicle_yaw = NavigationSystem.transform_angle(self.simulator.vehicle_variables.vehicle_yaw)
        # vehicle_yaw = np.tan( math.radians(self.simulator.vehicle_variables.vehicle_yaw))

        rot_offsets =[]
        # print(len(self.local_route))
        for i in range(len(self.local_route)-1):
            p1 = self.local_route[i].location
            p2 = self.local_route[i+1].location
            # vec1 = misc.vector(p1,p2)
            # vec2 = [math.cos(vehicle_yaw*180/np.pi),math.sin(vehicle_yaw*180/np.pi),0]
            # angle_ =  np.arccos(vec1.dot(vec2))*180/np.pi
            y_ = p2.y-p1.y + np.finfo(float).eps 
            x_ = p2.x-p1.x +np.finfo(float).eps 
            angle = NavigationSystem.transform_angle(math.degrees(np.arctan2(y_,x_))%360)
            rot_offsets.append( NavigationSystem.transform_angle( (angle-vehicle_yaw)%360 ) )
            # print(i,angle,vehicle_yaw)
        return rot_offsets

    def get_offset_distance(self):
        p1 = self.local_route[0].location
        p2 = self.local_route[1].location
        y_ = p2.y-p1.y + np.finfo(float).eps 
        x_ = p2.x-p1.x +np.finfo(float).eps
        angle =  math.degrees(np.arctan2(y_,x_))
        distance = self.get_distance(self.simulator.vehicle_variables.vehicle_location,p2,res=1)
        waypoint_angle = NavigationSystem.transform_angle(self.local_route[1].rotation.yaw)
        offset = NavigationSystem.transform_angle(angle-waypoint_angle)
        return math.sin( math.radians(offset) )*distance,abs(math.cos( math.radians(offset) )*distance)

    @staticmethod
    def transform_angle(angle):
        if (angle<180):
            return angle
        else:
            return angle-360
         
        
    def clean_route(self):
        temp_route = []
        i = 0
        while i<(len(self.ideal_route)-1):
            t1 = self.ideal_route[i]
            t2 = self.ideal_route[i+1]
            temp_route.append(t1)
            if NavigationSystem.get_distance(t1.location,t2.location,res=1)<=5:
                i+=1
            i+=1
        temp_route.append(self.ideal_route[-1])
        self.ideal_route = temp_route    
       

    def clean_back(self):
        first = True
        back_cnt =0
        n_iter =0 
        while (first or back_cnt!=0) and n_iter<=5:
            i=1
            temp_route = []
            back_cnt=0
            first =False
            temp_route.append(self.ideal_route[0])
            while i<(len(self.ideal_route)-1):
                t0 = self.ideal_route[i-1]
                t1 = self.ideal_route[i]
                t2 = self.ideal_route[i+1]
                
                behind = NavigationSystem.check_behind(t0,t1,t2)
            
                temp_route.append(self.ideal_route[i])
                if behind:
                    back_cnt+=1
                    i+=1
                i+=1
            temp_route.append(self.ideal_route[-1])
            self.ideal_route = temp_route 
            n_iter+=1  

    def reset(self):
        # print("calling reset")
        self.curr_pos = 0
        # print("curent pos is %d"%(self.curr_pos))
    
    @staticmethod 
    def check_error(t0,t1,t2):
        b=None
        if t0!=None:
            b = NavigationSystem.check_behind(t0,t1,t2)
        p1 = t1.location
        p2 = t2.location
        l = NavigationSystem.get_distance(p1,p2)
        if l>=7 and (t0==None or  b[0]==False):
            return False
        else:
            return True

    @staticmethod
    def check_behind_i(t0,t1,t2,i):
       
        unit_vec = NavigationSystem.get_loc(misc.vector(t0.location,t1.location))
        r_vec = NavigationSystem.get_loc(misc.vector(t1.location,t2.location))

        dot = r_vec.x*unit_vec.x + r_vec.y*unit_vec.y
        angle = math.degrees(np.arccos(dot))
        # print(angle)
        if angle>45:
            return i+1 # True# (True,unit_vec,r_vec,dot)
        else:
            return i # (False,unit_vec,r_vec,dot)
    
    @staticmethod
    def check_behind(t0,t1,t2):
       
        unit_vec = NavigationSystem.get_loc(misc.vector(t0.location,t1.location))
        r_vec = NavigationSystem.get_loc(misc.vector(t1.location,t2.location))

        dot = r_vec.x*unit_vec.x + r_vec.y*unit_vec.y
        angle = math.degrees(np.arccos(dot))
        # print(angle)
        if dot<0:
            return  True# (True,unit_vec,r_vec,dot)
        else:
            return  False #,unit_vec,r_vec,dot)

    @staticmethod
    def get_loc(p):
        return carla.Location(p[0],p[1],p[2])

    @staticmethod
    def loc_str(l,z=0):
        if z:
            return f'x: {l.x}, y: {l.y}, z:{l.z}'
        else:
            return f'x: {l.x}, y: {l.y}'
       
    def fill_gaps(self):
        temp_route = []
        
        p1,p2=None,None
        first = True
        done = True
        cnt =0
        i=0
        for j in range(2):
            i=0
            temp_route =[]
            while i<(len(self.ideal_route)-1):
                p1 = self.ideal_route[i].location
                p2 = self.ideal_route[i+1].location
                distance =NavigationSystem.get_distance(p1,p2,res=1)
                temp_route.append(self.ideal_route[i])
                if distance>=7:
                    done=False
                    # angle = math.radians(self.ideal_route[i].rotation.yaw) #need to change
                    p_t =carla.Location(x = (p1.x+p2.x)/2,y = (p1.y+p2.y)/2,z=p1.z)
                    w = self.simulator.map.get_waypoint(p_t)
                    temp_route.append(w.transform)
                i+=1
            cnt+=1
            temp_route.append(self.ideal_route[-1])
            self.ideal_route = temp_route

    @staticmethod
    def get_distance(p1,p2,res=0): 
        l =(p1.x-p2.x)**2 + (p1.y-p2.y)**2
        if res:
            return l**0.5
        else:
            return l
    

    
            