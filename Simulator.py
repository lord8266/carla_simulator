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

# import ai_model
class Type(Enum):
    Automatic =1
    Manual =2

class Status(Enum):
    COMPLETED=1,
    FAILED=2,
    RUNNING=3,
    RESTART=4,

class CameraType(Enum):
    RGB=1,
    Semantic=2, # start_point, end_point = np.random.randint(0,len(self.navigation_system.spawn_points),size=2) #temporary
        # self.navigation_system.make_ideal_route(start_point,end_point)

class VehicleVariables:

    def __init__(self,simulator):
        self.simulator = simulator
        self.wait_for_lag = False
        self.future_transform = None
        self.vehicle_velocity_magnitude =0
        self.prev_velocity_magnitude = 0
        self.update()
    def update(self):
        self.vehicle_transform = self.simulator.vehicle_controller.vehicle.get_transform()
        if self.wait_for_lag==True:
            f = self.cmp_transform(self.vehicle_transform.location,self.future_transform.location)
            if f:
                self.wait_for_lag= False
            else:
                self.vehicle_transform = self.future_transform

        self.vehicle_location = self.vehicle_transform.location
        self.vehicle_yaw = self.vehicle_transform.rotation.yaw%360
        self.vehicle_waypoint = self.simulator.map.get_waypoint(self.vehicle_location)
        self.vehicle_velocity = self.simulator.vehicle_controller.vehicle.get_velocity()
        self.vehicle_velocity_magnitude = (self.vehicle_velocity.x**2 + self.vehicle_velocity.y**2)**0.5
        self.road_id = self.vehicle_waypoint.road_id
        self.lane_id = self.vehicle_waypoint.lane_id

        # self.print_velocity()
        # print(self.vehicle_velocity_magnitude)
        
    def update_npc(self):
        pass

    def add_npc(self,actor_list):
        self.actor_list = self.simulator.world.get_actors(actor_list)
    
    def print_velocity(self):
        if abs(self.vehicle_velocity_magnitude-self.prev_velocity_magnitude)>=0.3:
            self.prev_velocity_magnitude = self.vehicle_velocity_magnitude
            print("Veloctity",self.vehicle_velocity_magnitude)

    

    def cmp_transform(self,p1,p2):
        # print(p1,p2)
        if abs(p1.x-p2.x)<4:
            if abs(p1.y-p2.y)<4:
                return True
        return False
    
        # self.prev =curr
    def start_wait(self,transform):
        self.wait_for_lag =True
        self.future_transform = transform


def transform_observation(obs):
    obs_ =obs[1:3]
    obs_[0] = obs_[0]/10
    obs_[1] = obs_[1]/180 +0.5
    return obs_


class Simulator:

    def __init__(self,carla_server='127.0.0.1',port=2000):
        pygame.init()
        self.intitalize_carla(carla_server,port)
        self.initialize_navigation()
        self.initialize_vehicle()
        self.initialize_game_manager()
        self.initialize_sensor_manager()
        self.initialize_control_manager()
        self.initialize_reward_system()
        self.initialize_variables()
        self.type = Type.Automatic
        self.running = True
        self.rendering =True
        self.respawn_pos_times = 0
        self.key_control = False
        self.collision_vehicle =False
        self.lane_ai = lane_ai.LaneAI(self)
        self.traffic_controller = traffic_controller.TrafficController(self,50,100)
        self.traffic_controller.add_vehicles()
        self.traffic_controller.add_pedestrians()
        #need to change from here
        self.navigation_system.make_local_route()
        self.agent = basic_agent.BasicAgent(self.vehicle_controller.vehicle)
        # drawing_library.draw_arrows(self.world.debug,[i.location for i in self.navigation_system.ideal_route])
        # drawing_library.print_locations(self.world.debug,[i.location for i in self.navigation_system.ideal_route])
        # self.add_npc()
        self.world.tick()
        self.world.wait_for_tick()
        # self.data_collector = data_collector.DataCollector(self)
        # self.collision_collector = data_collector.CollisionCollector(self)
        # self.free_road = ai_model.FreeRoad(self.vehicle_controller.control)
        self.last_stop = pygame.time.get_ticks()
        self.cnt = 0
        self.collide_cnt = 0
        
    def temp(self):
        self.vehicle_controller.vehicle.set_transform(self.navigation_system.start)

    def intitalize_carla(self,carla_server,port):
        self.client = carla.Client(carla_server,port)
        self.client.set_timeout(12.0)
        self.world = self.client.load_world('Town05')#self.client.get_world()
        # self.world = self.client.get_world()
        settings = self.world.get_settings() 
        # settings.synchronous_mode = True # 21 22 247 248
        # settings.no_rendering_mode = True
        self.world.apply_settings(settings)
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

    def add_npc(self):
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        spawn_points = self.navigation_system.spawn_points

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        batch = []
        actor_list =[]
        for n, transform in enumerate(spawn_points):
            if n >= 5:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = blueprint.get_attribute('color').recommended_values
                print(color)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

        for response in self.client.apply_batch_sync(batch):
            print(response)
            actor_list.append(response)

        self.vehicle_variables.add_npc(actor_list)
        
    def initialize_navigation(self):
        self.navigation_system = navigation_system.NavigationSystem(self)
        self.navigation_system.make_map_data(res=4)
        self.start_point, self.end_point =np.random.randint(0,len(self.navigation_system.spawn_points),size=2)
        self.start_point,self.end_point = self.navigation_system.spawn_points[self.start_point],self.navigation_system.spawn_points[self.end_point]

        # self.start_point, self.end_point =22,40
        self.navigation_system.make_ideal_route_r(self.start_point,self.end_point)
        self.base_start =self.navigation_system.ideal_route[0]
        self.base_end = self.navigation_system.ideal_route[-1]
         # temporary
    
    def new_path(self):
        self.navigation_system.make_map_data(res=4)
        self.start_point = self.end_point
        self.end_point = self.navigation_system.spawn_points[ np.random.randint(0,len(self.navigation_system.spawn_points)) ]
        d = navigation_system.NavigationSystem.get_distance(self.start_point.location,self.end_point.location,res=1)
        while d<15:
            self.end_point =  self.navigation_system.spawn_points[np.random.randint(0,len(self.navigation_system.spawn_points))]
            d = navigation_system.NavigationSystem.get_distance(self.start_point.location,self.end_point.location,res=1)
        
        self.navigation_system.make_ideal_route_r(self.start_point,self.end_point)
            
        self.base_start =self.navigation_system.ideal_route[0]
        self.base_end = self.navigation_system.ideal_route[-1]
        self.navigation_system.make_local_route()

    def initialize_vehicle(self):
        self.vehicle_controller = vehicle_controller.VehicleController(self,AI=True)

    def initialize_sensor_manager(self):
       self.sensor_manager = sensor_manager.SensorManager(self)
       self.sensor_manager.initialize_rgb_camera()
       self.camera_type = CameraType.RGB
       self.sensor_manager.camera.listen(lambda image: self.game_manager.camera_callback(image))
    #    self.sensor_manager.initialize_semantic_camera()
    #    self.sensor_manager.initialize_obstacle_sensor()
    #    self.sensor_manager.semantic_camera.listen(lambda image: self.game_manager.semantic_callback(image))
       self.sensor_manager.initialize_collision_sensor()
    #    self.sensor_manager.initialize_lane_invasion_sensor()
       
       
    def initialize_game_manager(self):
        self.game_manager = game_manager.GameManager(self)
        self.game_manager.update()

    def initialize_control_manager(self):
        self.control_manager = control_manager.ControlManager(self)
    
    def initialize_reward_system(self):
        self.reward_system = reward_system.RewardSystem(self)

    def initialize_variables(self):
        self.vehicle_variables = VehicleVariables(self)
        self.vehicle_variables.start_wait(self.navigation_system.start)
 
    def step(self,action,model=1):
        # self.world.tick()
        # ts = self.world.wait_for_tick()

        self.vehicle_variables.update()
        self.game_manager.update()
        self.navigation_system.make_local_route()

        self.observation = self.get_observation()
        reward,status = self.reward_system.update_rewards()
        # self.data_collector.update()
        
        if self.type == Type.Manual:
            self.vehicle_controller.control_by_input()
        elif model==1:
            self.vehicle_controller.copy_control(self.control_manager.controls[action])
        elif model==2:
            self.vehicle_controller.change_control(action)



        
        self.render()
        self.traffic_controller.update()

        curr = pygame.time.get_ticks()
        vel = self.vehicle_variables.vehicle_velocity_magnitude
        traffic_light = self.sensor_manager.traffic_light_sensor()
        if ((vel>0.05) or traffic_light==0) or self.traffic_controller.ai_enabled==True:
            self.last_stop = curr
            
       
        
        if (curr-self.last_stop)>5000:
            self.re_level()
            self.last_stop = curr
        # if (traffic_light == 0) and not self.vehicle_variables.vehicle_waypoint.is_intersection:
        #     control = self.vehicle_controller.control
        #     control.throttle = 0.0
        #     control.brake = 1.0
        # self.vehicle_controller.control_by_input(passive=True)

        w = self.vehicle_variables.vehicle_waypoint
        
        # if w.is_intersection and self.vehicle_variables.vehicle_velocity_magnitude > 5 :
        #     self.vehicle_controller.control.throttle *= 0.6
        # if w.is_intersection and self.vehicle_controller.control.steer > 0:
        #     self.vehicle_controller.control.throttle *= 1.3
        self.traffic_controller.update_local_front()

        self.vehicle_controller.apply_control()
        
        
        if self.collide_cnt>0:
            self.collide_cnt-=1
            self.collide_callback()
        # light == 0) and not self.vehicle_variables.vehicle_waypoint.is_intersection:
        #     control = self.vehicle_controller.control
        #     control.throttle = 0.0
        #     control.brake = 1.0
        return self.observation,reward,status!=Status.RUNNING,{}

   
    def switch_render(self):
        if self.rendering==True:
            self.sensor_manager.camera.stop()
            self.sensor_manager.semantic_camera.stop()
            self.rendering =False
        else:
            self.sensor_manager.camera.listen(lambda image: self.game_manager.camera_callback(image))
            self.sensor_manager.semantic_camera.listen(lambda image: self.game_manager.semantic_callback(image))
            self.rendering =True

    def get_observation(self):

        rot_offsets = self.navigation_system.get_rot_offset() # temporary
        distance_to_destination_sin, distance_to_destination_cos= self.navigation_system.get_offset_distance()
       
        return [distance_to_destination_sin,distance_to_destination_cos]+ list(np.clip(rot_offsets[:4],-70,70))
        

    def reset(self):
        status =self.reward_system.status
        if status==Status.COMPLETED:
           return self.on_completion()
        if status==Status.RESTART:
            return self.on_failure()
        else:
            return self.on_failure()
    
    def on_completion(self):

        self.reward_system.reset(t=0)
       
        return self.get_observation()
    
    def on_failure(self):
        fail_point =self.navigation_system.curr_pos
        self.navigation_system.reset()
        if self.respawn_pos_times<3:
            self.navigation_system.curr_pos =fail_point
            self.respawn_pos_times +=1
        else:
            self.respawn_pos_times =0
            fail_point = 0
        self.vehicle_variables.start_wait(self.navigation_system.ideal_route[fail_point])
        self.reward_system.reset(t=1)
        self.vehicle_controller.reset(self.navigation_system.ideal_route[fail_point])
        
        return self.get_observation()

    def on_restart(self):
        self.respawn_pos_times =0
        self.navigation_system.reset()
        self.reward_system.reset(t=1)
        self.vehicle_variables.start_wait(self.navigation_system.ideal_route[0])
        
        self.vehicle_controller.reset(self.navigation_system.ideal_route[0])
        
        # print("Car rest at pos",self.navigation_system.start,self.navigation_system.curr_pos)
        return self.get_observation()

    def re_level(self,random_spawn=False):
            
        self.start_point, self.end_point =np.random.randint(0,len(self.navigation_system.spawn_points),size=2)
        self.start_point, self.end_point = self.navigation_system.spawn_points[self.start_point],self.navigation_system.spawn_points[self.end_point]
        # self.start_point, self.end_point =22,40
        self.navigation_system.make_ideal_route_r(self.start_point,self.end_point)

      
          
                
        self.on_restart()
    
    def collide_callback(self):
        
        self.traffic_controller.disableAI(failed=True)
        self.start_point = self.traffic_controller.get_far_away()
        self.end_point = self.navigation_system.spawn_points[ np.random.randint(0,len(self.navigation_system.spawn_points)) ]
        d = navigation_system.NavigationSystem.get_distance(self.start_point.location,self.end_point.location,res=1)
        while d<15:
            self.end_point =  self.navigation_system.spawn_points[np.random.randint(0,len(self.navigation_system.spawn_points))]
            d = navigation_system.NavigationSystem.get_distance(self.start_point.location,self.end_point.location,res=1)
        
        self.navigation_system.make_ideal_route_r(self.start_point,self.end_point)
        self.on_restart()


    def render(self):
        self.game_manager.render()
    
    def stop(self):
        self.vehicle_controller.vehicle.destroy()
        self.sensor_manager.stop_camera()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
    
    def switch_input(self):
        
        if self.type==Type.Manual:
            self.type = Type.Automatic
        else:
            self.type = Type.Manual

    def camera_switch(self): #temporary
        if (self.camera_type==CameraType.RGB):
            self.camera_type = CameraType.Semantic
            self.sensor_manager.semantic_camera.listen(lambda image: self.game_manager.semantic_callback(image))
            self.sensor_manager.camera.stop()

        else:
            self.sensor_manager.camera.listen(lambda image: self.game_manager.camera_callback(image))
            self.sensor_manager.semantic_camera.stop()
            self.camera_type = CameraType.RGB

def collision_with(event):
    
    print("collision")



       
