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
from agents.tools import misc
import lane_ai
import collision_control 

class TrafficController:

    def __init__(self,simulator,vehicle_count):

        self.simulator = simulator
        self.prev = pygame.time.get_ticks()
        self.batch_running =False
        self.count = 0
        self.max =vehicle_count
        self.applied_stop = False
        self.control = self.simulator.vehicle_controller.control
        self.obstacles = {}
        self.env = collision_control.SpeedControlEnvironment(self)
        self.ai_enabled = False
        self.curr_locations = []
        self.lane_obstacles = {}
        self.collision_control = collision_control.CollisionControl(self)
        self.override = False

    def add_vehicles(self):
        blueprints = self.simulator.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        spawn_points = self.simulator.navigation_system.spawn_points

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        batch = []
        actor_list =[]
        for n, transform in enumerate(spawn_points):
            if n >= self.max:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

        for response in self.simulator.client.apply_batch_sync(batch):
            # print(response)
            actor_list.append(response.actor_id)
        self.get_actors(actor_list)

    def get_actors(self,actor_list):
        vehicles = list(self.simulator.world.get_actors(actor_list))
        self.vehicles = vehicles

    def update_distances(self):
        curr = pygame.time.get_ticks()
        self.curr_locations = []
        # print("called")
        p1 = self.simulator.vehicle_variables.vehicle_location
        found_ids = []  
       
        for v in self.vehicles:
            
            p2 = v.get_location()
            self.curr_locations.append(p2)
            d = navigation_system.NavigationSystem.get_distance(p1,p2,res=1)
            
            if d<30:
                passed =False
                if v.id in self.obstacles:
                    passed=True
                    self.obstacles[v.id].update(d)
                    # print("Update")
                else:
                    this_obs = lane_ai.Obstacle(self.simulator,v,d)
                    passed = True
                    self.obstacles[v.id] = this_obs
                    # print("Add")
                
                if passed:
                    found_ids.append(v.id)
        self.update_lane_obstacles()         
        self.rem_obstacles(found_ids)
    
    def update_lane_obstacles(self):

        self.lane_obstacles = {}
        road_id,lane_id = self.simulator.vehicle_variables.road_id,self.simulator.vehicle_variables.lane_id
        lane_side = lane_id>0
        for _,o in self.obstacles.items():

            if o.angle<110:
                if o.road_id==road_id :
                    this_lane_side = o.lane_id>0

                    if this_lane_side==lane_side:

                        if  o.lane_id in self.lane_obstacles:
                            self.lane_obstacles[o.lane_id].append(o)
                        else:
                            self.lane_obstacles[o.lane_id] = [o]
                
        if lane_id not in self.lane_obstacles:
            self.lane_obstacles[lane_id] = []
        
        for _,o_list in self.lane_obstacles.items():
            o_list.sort(key=lambda f:f.distance)

    def get_far_away(self,distance=15):
        spawn_points = self.simulator.navigation_system.spawn_points
        max_spawn_distance = 0
        for s in spawn_points:
            distances = []
            for s2 in self.simulator.traffic_controller.curr_locations:
                d = navigation_system.NavigationSystem.get_distance(s.location,s2,res=1)
                distances.append(d)

            max_spawn = min(distances)
            if max_spawn>distance:
                spawn_point = s 
                break
            elif max_spawn>max_spawn_distance:
                spawn_point = s
        
        return spawn_point

    def get_closest_in_waypoint(self,waypoint,forward=False):

        road_id = waypoint.road_id 
        lane_id = waypoint.lane_id

        lane_obstacles = []

        for _,obstacle in self.obstacles.items():
            if obstacle.road_id==road_id and obstacle.lane_id==lane_id:
                if forward:
                    if obstacle.angle<130:
                        lane_obstacles.append(obstacle)
                else:
                    lane_obstacles.append(obstacle)

        if lane_obstacles:    
            closest = min(lane_obstacles,key=lambda f:f.distance)
            return closest
        else:
            return None

    def enableAI(self):
        pass
        # if self.ai_enabled:
        #     self.env.run()
            
        # else:
        #     print("Enable AI")
        #     self.ai_enabled =True
        #     self.env.start()

    def disableAI(self,failed=False):
        if self.ai_enabled:
            print("Disable AI")
            self.env.stop(failed)
            self.ai_enabled = False
        
    def print_obstacles(self):
        curr = pygame.time.get_ticks()
        lane_id = self.simulator.vehicle_variables.lane_id
        # data = list(self.obstacles.items())
        # data.sort(key=lambda f:abs(self.obstacles[f[0]].angle) )
        if (curr-self.prev)>1000:
            # print("\n".join( [str(i[1]) for i in data] ))
            # print(self.ai_observation)
            # print(self.surrounding_data)
            print(self.collision_control.state)
            for _,a in self.lane_obstacles.items():
                if _==lane_id:
                    print("Lane:",_,"(Vehicle)")
                else:
                    print("Lane:",_)
                for u in a:
                    print("    ",str(u))
            print()
            
            self.prev = curr

    def compare_waypoint_lanes(self,w1,w2):
        if w1.road_id==w2.road_id and w1.lane_id==w2.lane_id:
            # print(w1.road_id,w1.lane_id,w2.road_id,w2.lane_id)
            return False
        else:
            return True
 
    def predict_future(self):
        nav = self.simulator.navigation_system
        start = self.simulator.vehicle_variables.vehicle_waypoint
        future_vehicles = []
        last = min(len(nav.ideal_route_waypoints)-nav.curr_pos,5)
        for i in range(0,last):
            next_ = nav.ideal_route_waypoints[nav.curr_pos+i]
            if self.compare_waypoint_lanes(start,next_):
                data_future = self.get_closest_in_waypoint(next_,forward=True)
                if data_future:
                    future_vehicles.append(data_future)
                
            else:
                data_future =None
        
        if future_vehicles:
            dat = [str(i) for i in future_vehicles]
            # print('\n'.join(dat),end='\n\n')
            data_future = min(future_vehicles,key=lambda f:f.distance)
        else:
            data_future = None
    
        same_lane = self.get_closest_in_waypoint(self.simulator.vehicle_variables.vehicle_waypoint,forward=True)
        
        s = ""
        if same_lane:
            data_same = same_lane.distance,same_lane.delta_d
            s+=f'SameLane: {str(same_lane)}\n'
        else:
            data_same = 100,0
            s+='SameLane: None\n'
        if data_future:
            data_next = data_future.distance,data_future.delta_d
            s+=f'DataFuture: {str(data_future)}\n'
        else:
            data_next = 100,0
            s+='DataFuture: None\n'

        self.ai_observation = data_same+(self.simulator.vehicle_variables.vehicle_velocity_magnitude*10,)
        # if same_lane:
        #     if same_lane.distance<20 and self.ai_observation[0]<self.ai_observation[2]:
        #         self.simulator.lane_ai.lane_changer.check_new_lane(force=True)
        
        self.check_ai(same_lane,data_future)


        return s

    def check_ai(self,same_lane,data_future):
        
        if same_lane or data_future:
            self.enableAI()
        else:
            self.disableAI()


    def rem_obstacles(self,found_list):
        rem =  []
        for k in self.obstacles:
            if k not in found_list:
                rem.append(k)
        
        for i in rem:
          
            self.obstacles.pop(i)


    def stop_vehicle(self):
        self.control.throttle = 0
        self.control.brake =1.0
                

    def check_lane_road(self,vehicle,vehicle_loc):
        waypoint = self.simulator.vehicle_variables.vehicle_waypoint
        road_id = waypoint.road_id
        lane_id = waypoint.lane_id

        waypoint2 = self.simulator.map.get_waypoint(vehicle_loc)
        road_id2 = waypoint2.road_id
        lane_id2 = waypoint2.lane_id

        if road_id==road_id2 and lane_id==lane_id2:
            return True
        else:
            return False

    
    def update(self):
        # print("call update")
        # curr =pygame.time.get_ticks()
        self.update_distances()
        self.surrounding_data = self.predict_future()
        self.collision_control.update()
        # self.print_obstacles()

        


