import drawing_library
import carla
import navigation_system
import math
import numpy as np
from agents.tools import misc
import pygame
from enum import Enum
import collision_control
class LaneAI:
    def __init__(self,simulator):
            self.count = 1
            self.simulator =simulator
            self.previous_lane_id = None
            self.prev = pygame.time.get_ticks()
            self.obstacles  ={}
            self.cnt = 0
            self.print_prev = self.prev
            self.add_buffer = []
            self.obstacle_table = {}
            self.update_ = False
            self.lane_changer = LaneChanger(self)
            self.lane_prev = self.prev
            self.lane_closest = {}
            # self.collision_control = collision_control.CollisionControl(self)

    def get_closest_obstacles(self):
        vehicle  = self.simulator.vehicle_controller.vehicle
        vehicle_waypoint = self.simulator.map.get_waypoint(self.simulator.vehicle_variables.vehicle_location)
        vehicle_lane_id = vehicle_waypoint.lane_id
        vehicle_road_id = vehicle_waypoint.road_id

        lane_closest = {}
        lane_sign = vehicle_lane_id>0

        if vehicle_road_id in self.obstacle_table:

            for lane in self.obstacle_table[vehicle_road_id]:
                if (lane>0)==lane_sign:
                    in_front = [ w for w in self.obstacle_table[vehicle_road_id][lane] if w.angle<120] 
                    # print(in_front)
                    if in_front:
                        closest_obstacle = min(in_front ,key=lambda f:f.distance)
                        lane_closest[lane] = closest_obstacle
                    
        
        self.lane_closest = lane_closest

    def get_obstacle_status(self,event):
        
        obstacle = event.other_actor

        if str(obstacle.type_id).find("vehicle")!=-1:
            # print(obstacle,self.cnt)
            self.cnt+=1
            obstacle_waypoint = self.simulator.map.get_waypoint(obstacle.get_location())
            obstacle_lane_id = obstacle_waypoint.lane_id
            obstacle_road_id = obstacle_waypoint.road_id

            vehicle  = self.simulator.vehicle_controller.vehicle
            vehicle_waypoint = self.simulator.map.get_waypoint(self.simulator.vehicle_variables.vehicle_location)
            vehicle_lane_id = vehicle_waypoint.lane_id
            vehicle_road_id = vehicle_waypoint.road_id

            # print(f"Found obstacle on lane:{obstacle_lane_id}, road: {obstacle_road_id}, vehicle on lane:{vehicle_lane_id}, road: {vehicle_road_id} ")
            # print(f"Obstacle ID:{obstacle}, Vehicle ID: {vehicle},  Distance:",event.distance)
            # print()
            self.add_buffer.append((obstacle,obstacle_road_id,obstacle_lane_id))

    def add_obstacle(self,obstacle,distance):
            obstacle_waypoint = self.simulator.map.get_waypoint(obstacle.get_location())
            obstacle_lane_id = obstacle_waypoint.lane_id
            obstacle_road_id = obstacle_waypoint.road_id

            # if distance<6:
            #     obstacle_lane_dir = obstacle_lane_id>0
            #     vehicle = self.simulator.vehicle_variables.vehicle_waypoint
            #     road_id = vehicle.road_id
            #     lane_dir = vehicle.lane_id>0
            #     if road_id==obstacle_road_id:
            #         if lane_dir==obstacle_lane_dir:
            #             self.force_stop()

            self.add_buffer.append((obstacle,obstacle_road_id,obstacle_lane_id))

   
    
    def start_queue(self):
        if len(self.add_buffer)>50:
            r = self.add_buffer[:50]
            self.add_buffer = self.add_buffer[50:]
            for o,r,l in r:
                self.add_to_set(o,r,l)
        else:
            r = self.add_buffer
            self.add_buffer = []
            for o,r,l in r:
                self.add_to_set(o,r,l)
         


    def request_new_lane(self,prefer_left=False):
        curr =pygame.time.get_ticks()
        if (curr-self.lane_prev)>2000:
            vehicle =self.simulator.vehicle_variables.vehicle_location
            waypoint = self.simulator.map.get_waypoint(vehicle)
            next_waypoint =None
            
            if prefer_left:
                if str(waypoint.lane_change)=='Left' or str(waypoint.lane_change)=='Both':
                    next_waypoint = waypoint.get_left_lane()
                    # print("Change Left")
                elif str(waypoint.lane_change)=='Right' :
                    next_waypoint = waypoint.get_right_lane()
                    # print("Change Right")
                else:
                    pass
                    # print("Not Possible")
            else:
                if str(waypoint.lane_change)=='Right' or str(waypoint.lane_change)=='Both':
                    next_waypoint = waypoint.get_right_lane()
                    # print("Change Right")
                elif str(waypoint.lane_change)=='Left':
                    next_waypoint = waypoint.get_left_lane()
                    pass
                    # print("Change Left")
                else:
                    pass
                    # print("Not Possible")
                
            self.lane_prev = curr
                
            
            return self.check_waypoint_integrity(next_waypoint)
        
        return False,None,None

    def check_waypoint_integrity(self,next_waypoint):

        if next_waypoint:
            curr_waypoint = self.simulator.vehicle_variables.vehicle_waypoint
            start_ = curr_waypoint
            road_id,lane_id = curr_waypoint.road_id,curr_waypoint.lane_id
            passed = True
            additional = []
            for i in range(3):
                if passed: 
                    curr_waypoint = curr_waypoint.next(2.1)
                    if curr_waypoint:
                        curr_waypoint = curr_waypoint[0]
                        if curr_waypoint.road_id==road_id and curr_waypoint.lane_id==lane_id:
                            additional.append(curr_waypoint)
                            # print(additional)
                        else:
                            passed = False
                    else:
                        passed = False
            road_id,lane_id = next_waypoint.road_id,next_waypoint.lane_id
            if passed:
                for i in range(3):
                    if passed: 
                        next_waypoint = next_waypoint.next(2.5)
                        if next_waypoint:
                            next_waypoint = next_waypoint[0]
                            if next_waypoint.road_id==road_id and next_waypoint.lane_id==lane_id:
                                pass
                            else:
                                passed = False
                        else:
                            passed = False
            
            if passed:
                return True,[start_]+additional,next_waypoint
            else:
                return False,None,None
        else:
            return False,None,None


    def check_waypoint_angle(self,next_waypoint,vehicle_transform,turn_angle=150):
       
        vp = vehicle_transform
        p2 = next_waypoint.transform.location
        u2 =misc.vector(p2, vp.location)
        u1 = np.array(misc.vector(vp.location,self.simulator.navigation_system.ideal_route[self.simulator.navigation_system.curr_pos].location))
        angle = math.degrees(np.arccos(u1.dot(u2) ))
        cnt =0 
        # print(angle)
        while angle<turn_angle and cnt<10:
            n = next_waypoint.next(0.6)
            if n:
                next_waypoint = n[0]
            else:
                break
            
            p2 = next_waypoint.transform.location
            u2 =misc.vector(p2, vp.location)
            u1 = np.array(misc.vector(vp.location,self.simulator.navigation_system.ideal_route[self.simulator.navigation_system.curr_pos].location))
            angle = math.degrees(np.arccos(u1.dot(u2) ))
            # print(angle)
            cnt+=1
        return next_waypoint


        # cnt =0
        # while int(abs(y2-x2))==1:
        #     next_waypoint = next_waypoint.next(0.5)[0]
        #     cnt+=1
        #     vp = vehicle_transform.location
        #     p2 = next_waypoint.transform.location
        #     y2 = p2.y-vp.y + np.finfo(float).eps 
        #     x2 = p2.x-vp.x +np.finfo(float).eps 
            
        #     print(slope)


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
    
    def add_to_set(self,obstacle,road_id,lane_id):
        vehicle_road_id = self.simulator.vehicle_variables.vehicle_waypoint.road_id
        vehicle_lane_id = self.simulator.vehicle_variables.vehicle_waypoint.lane_id

        if road_id==vehicle_road_id:
            if obstacle.id not in self.obstacles:
                self.obstacles[obstacle.id] =Obstacle(self.simulator,obstacle,road_id,lane_id)
                self.update_ = True
            else:
                self.obstacles[obstacle.id].last_updated = pygame.time.get_ticks()

    def update_set(self):
        curr_road_id =self.simulator.vehicle_variables.vehicle_waypoint.road_id
        rem_buffer = []
        curr = pygame.time.get_ticks()
        for _,o in self.obstacles.items():
            if (curr-o.last_updated)>2000 or o.road_id!=curr_road_id:
                rem_buffer.append(_)

        for r in rem_buffer:
            if self.obstacles[r].ai_follower!=None:
                self.obstacles[r].ai_follower.disable_AI()
            self.update_= True
            self.obstacles.pop(r)
             
    def update(self):

        curr =pygame.time.get_ticks()
        self.update_set()
        self.start_queue()
        self.update_table()
        if (curr-self.prev)>20:
            self.prev =curr
            for _,obstacle in self.obstacles.items():
                obstacle.update()

        if (curr-self.print_prev)>=400:
            # self.print_table()
            self.print_prev = curr
        
        self.check_collision()
    
    def check_collision(self):
        self.collision_control.update()
           
                    

    def update_table(self):
        if self.update_:
            self.update_ = False
            self.obstacle_table = {}

            for obs_id,obs in self.obstacles.items():

                if obs.road_id in self.obstacle_table:
                    if obs.lane_id in self.obstacle_table[obs.road_id]:
                        self.obstacle_table[obs.road_id][obs.lane_id].add(obs)
                    else:
                        self.obstacle_table[obs.road_id][obs.lane_id] = {obs}
                else:
                    self.obstacle_table[obs.road_id] = {obs.lane_id:{obs}}
        self.get_closest_obstacles()

    def print_table(self):
        curr =pygame.time.get_ticks()
        road_id = self.simulator.vehicle_variables.vehicle_waypoint.road_id
        lane_id = self.simulator.vehicle_variables.vehicle_waypoint.lane_id
        # print("Current Table")
        print(f"Vehicle: LaneType: { str(self.simulator.vehicle_variables.vehicle_waypoint.lane_type)}, LaneWidth:{str(self.simulator.vehicle_variables.vehicle_waypoint.lane_width)} Road:{road_id} Lane:{lane_id}")
        # for road in self.obstacle_table:
        #     print("Road",road)
        #     for lane in self.obstacle_table[road]:
        #         print("  Lane:",lane)

        #         if road_id==road and lane_id==lane:
        #             print("   Vehicle Here")
        #         for obs in self.obstacle_table[road][lane]:
        #             print("   Name:",obs.vehicle, "Last Updated:",curr-obs.last_updated,"Angle:",obs.angle,"Distance:",obs.distance,"Delta:",obs.delta_d)

        print("Closest:")
        for i,obs in self.lane_closest.items():
            print(f'Lane:',i,)
            print("   Name:",obs.vehicle, "Last Updated:",curr-obs.last_updated,"Angle:",obs.angle,"Distance:",obs.distance,"Delta:",obs.delta_d)
        print()

class Obstacle:

    def __init__(self,simulator,vehicle,distance):
        self.simulator = simulator
        self.vehicle = vehicle
        self.last_updated = pygame.time.get_ticks()
        self.prev_distance = 0
        self.update_cnt = 0
        self.update(distance)
        self.ai_follower = None
        self.extent = self.vehicle.bounding_box.extent
        
    def update(self,distance):
        # self.last_updated = pygame.time.get_ticks()
        self.location = self.vehicle.get_location()
        self.distance = navigation_system.NavigationSystem.get_distance(self.location,self.simulator.vehicle_variables.vehicle_location,res=1)
        self.waypoint = self.simulator.map.get_waypoint(self.location)
        self.road_id = self.waypoint.road_id
        self.lane_id = self.waypoint.lane_id
        self.bounding_box = self.vehicle.bounding_box
        self.get_direction()
        self.get_distance(distance)
        self.delta_d = self.distance-self.prev_distance
        if not self.update_cnt%7:
            self.prev_distance = self.distance
        self.update_cnt+=1
        if self.update_cnt>10000:
            self.update_cnt =0

    def __str__(self):
        
        return f'{self.vehicle}, Distance:{self.distance}, Delta_D:{self.delta_d}, Angle:{self.angle}'

    def get_distance(self,distance):
        e1 = self.simulator.vehicle_controller.vehicle.bounding_box.extent
        extent1 = max([e1.x,e1.y,e1.z])
        e2 = self.bounding_box.extent
        extent2 = max([e2.x,e2.y,e2.z])
        distance -=(extent1+extent2)*abs( math.cos(math.radians(self.angle)))
        self.distance = max(0,min(distance,70))
    
    
    def get_direction(self):
        
        vehicle_pos = self.simulator.vehicle_variables.vehicle_location
        rot = self.simulator.vehicle_variables.vehicle_waypoint.transform.rotation
        forward_vector = rot.get_forward_vector()
        forward_vector = [forward_vector.x,forward_vector.y,forward_vector.z]

        obstacle_vector = np.array(misc.vector(vehicle_pos,self.location))

        dot = obstacle_vector.dot(forward_vector)

        arc_cos = np.degrees(np.arccos(dot))
       
        self.angle  =  arc_cos

class State(Enum):
    RUNNING=1,
    LANE_CHANGE=2,
    FOLLOW=3
    STOP=4,

class LaneChanger:

    def __init__(self,lane_ai):
        self.lane_ai = lane_ai
        self.prev = pygame.time.get_ticks()
        self.state = State.RUNNING
        self.current_obstacle = None
        self.stop = False
        self.prev_stop = self.stop
    
    def update(self):
        curr =pygame.time.get_ticks()

           
    
    def check_new_lane(self,min_angle=150,force=False):

        done =False
        # if self.state==State.RUNNING:
        next_waypoint = self.lane_ai.request_new_lane()

        if next_waypoint:
            wp = self.lane_ai.simulator.vehicle_variables.vehicle_waypoint
            next_waypoint_lane_id = next_waypoint.lane_id

            if force:
                done = True
            else:
                closest_data = self.lane_ai.lane_closest
                distance_to_same_lane_obstacle = closest_data[wp.lane_id].distance

                if next_waypoint_lane_id in closest_data:
                    distance_other_lane_obstacle = closest_data[next_waypoint_lane_id].distance

                    if distance_to_same_lane_obstacle<distance_other_lane_obstacle:
                        print("Same",distance_to_same_lane_obstacle,", other",distance_other_lane_obstacle)
                        done  =True
                else:
                    done = True

            if done:
                next_waypoint= self.lane_ai.check_waypoint_angle(next_waypoint,self.lane_ai.simulator.vehicle_variables.vehicle_transform,min_angle)
                self.lane_ai.simulator.navigation_system.add_event(next_waypoint)
                self.target_lane_id = next_waypoint_lane_id
                # if not force:
                #     self.state = State.LANE_CHANGE

           
       

        return done
       

    def update_waypoint(self):
        wp_lane_id = self.lane_ai.simulator.vehicle_variables.vehicle_waypoint.lane_id
        if wp_lane_id==self.target_lane_id:
            self.state = State.RUNNING
            print("Change Lane Success")




