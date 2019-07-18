
import drawing_library
import carla
import navigation_system
import math
import numpy as np
from agents.tools import misc
import pygame
import lane_ai
from enum import Enum
from collections import deque
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import sgd,Adam
import os
import random
import reward_system
HIDDEN1_UNITS = 50
HIDDEN2_UNITS = 40
import os
import traffic_controller
import record_server
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import data_collector
import scipy.misc

class ControlState(Enum):
    AI=1,
    MANUAL=2

class LaneState(Enum):
    SAME_LANE=1,
    LANE_CHANGE=2,

class CollisionControl:

    def __init__(self,trc):
        self.traffic_controller = trc
        self.control = trc.simulator.vehicle_controller.control
        self.state = LaneState.SAME_LANE
        self.curr_lane = self.traffic_controller.simulator.vehicle_variables.lane_id
        self.record_server = record_server.RecordServer(trc.simulator)
        self.record_server.enabled=False
        self.check_completion=False
        self.last_state = self.state
        self.data_collector = data_collector.DataCollector('collision_data',500,100,100)
        self.image_collector = data_collector.DataCollector('collision_images',500,10,10)
        self.prev_time = pygame.time.get_ticks()
        self.image_size = (88, 200, 3)
        self.prev_state = [20,0.1]

    def update(self):
        if self.state==LaneState.SAME_LANE:
            self.update_same_lane()
        else:
            self.update_target_lane()

        self.update_lane_change()
        # self.try_lane_change()
        self.check_lane_change_completion()
        self.record_server.update()

        if self.state!=self.last_state:
            print(self.state)
            self.last_state = self.state

    def update_same_lane(self,type_=1):

        lane_obstacles = self.traffic_controller.lane_obstacles
        # print(lane_obstacles)
        lane_id = self.curr_lane
        if lane_id in lane_obstacles:
            if lane_obstacles[lane_id]:
                front = lane_obstacles[lane_id][0]
                self.record_server.start_recording()
                distance = front.distance

                if type_==1:
                    if 6.5<distance<10.5 and front.delta_d<0:
                        if self.traffic_controller.simulator.vehicle_variables.vehicle_velocity_magnitude>1:
                            self.modify_control(2,distance,front.delta_d)

                    if 4.5<distance<6.5 and front.delta_d<0:
                        self.modify_control(1,distance,front.delta_d)
                            

                    if distance<4.5:
                        self.modify_control(0,distance,front.delta_d)
                        self.record_server.stop_recording(2)
                else:
                     if distance<1:
                            self.modify_control(0,distance,front.delta_d)
                            self.record_server.stop_recording(2)

    def stack_images(self):
        avail,dat = self.traffic_controller.simulator.game_manager.pixel_buffer.get_pixels()
        data=[]
        for i in range(4):
            avail,dat = self.traffic_controller.simulator.game_manager.pixel_buffer.get_pixels()
            if avail==False:
                return False,None
            else:
                dat = scipy.misc.imresize(dat, [self.image_size[0],self.image_size[1]])
                data.append(dat)
        
        return True, np.c_[data].reshape( (self.image_size[0]*4,self.image_size[1],3) )

    def collect_data(self,state,target):
        state = round(state[0],3),state[1]
        print(state)
        if abs(state[0]-self.prev_state[0])>0.05:
            curr =pygame.time.get_ticks()

            if (curr-self.prev_time)>1500:
                
                avail,image = self.stack_images()
                if avail:
                    scipy.misc.imsave('data.png',image)
                    self.image_collector.save_data(image,target)
                self.prev_time = curr

            self.data_collector.save_data(state,target)
            self.prev_state = state
            
    
    def modify_control(self,action,distance,delta_d):
        
        if action==0:
            self.control.throttle = 0.0
            self.control.brake = 1.0
        elif action==1:
            self.control.throttle =0.3
        
        elif action==2:
            self.control.throttle/=2

        self.collect_data([distance,delta_d],action)
    def update_target_lane(self):
        lane_obstacles = self.traffic_controller.lane_obstacles
        lane_id = self.target_lane
        if lane_id in lane_obstacles:
            if lane_obstacles[lane_id]:
                front = lane_obstacles[lane_id][0]

                distance = front.distance

                if 3<distance<5 and front.delta_d<0.01:
                    self.modify_control(1,distance,front.delta_d)
                
                if distance<3:
                    self.modify_control(0,distance,front.delta_d)

                    
    def update_lane_change(self):
        lc = self.traffic_controller.simulator.navigation_system.local_route_waypoints
        road_id,lane_id = lc[0].road_id,lc[0].lane_id
        self.curr_lane = lane_id

        found_change = False
        for a in lc[:2]:
            a_road_id,a_lane_id = a.road_id,a.lane_id

            if a_road_id==road_id:
            
                if a_lane_id!=lane_id:
                    self.state  =LaneState.LANE_CHANGE
                    # print("New lane:",a_lane_id)
                    self.target_lane = a_lane_id
                    found_change = True
                    return

                    
        
        if found_change:
            self.state = LaneState.LANE_CHANGE
        else:
            self.state = LaneState.SAME_LANE

    def check_lane_change_completion(self):
        
        if self.check_completion:
            curr = self.traffic_controller.simulator.vehicle_variables.vehicle_waypoint
            target =self.target_lane

            if curr.lane_id==target:
                self.record_server.stop_recording(0)
                self.check_completion = False

                   
    def try_lane_change(self,force=False):

        lane_id = self.traffic_controller.simulator.vehicle_variables.lane_id
        lane_obstacles = self.traffic_controller.lane_obstacles
        passed = False
        if force:
            passed,prev_,next_ = self.change_lane()
            if passed:
                self.traffic_controller.simulator.navigation_system.add_event(prev_,next_)
            return
            
        if lane_id in lane_obstacles:
            if lane_obstacles[lane_id]:
                obs_same = lane_obstacles[lane_id][0]

                if 6<obs_same.distance<15:
                    passed,prev_,next_ = self.change_lane()
                    if passed:
                        lane_id =next_.lane_id
                        if lane_id in lane_obstacles:
                            if lane_obstacles[lane_id]:
                                obs_next = lane_obstacles[lane_id][0]
                                if (obs_next.distance-obs_same.distance)<2:
                                    passed = False
                        
        if passed:
            self.record_server.start_recording()
            self.check_completion = True
            self.traffic_controller.simulator.navigation_system.add_event(prev_,next_)
            self.target = next_.lane_id



    def change_lane(self):
        return self.traffic_controller.simulator.lane_ai.request_new_lane()


class SpeedControlEnvironment:

    def __init__(self,traffic_controller):
        self.traffic_controller = traffic_controller
        # pass distance and delta_d
        # reward negative distance
        # self.actions = [30,20,-20,-40,-60,-120,-140]
        # self.ai = SpeedControlAI(self,input_size=4,action_size=7)
        self.actions = [50,30,-90,-130,-160,-190]
        self.ai = SpeedControlAI(self,input_size=3,action_size=6)

    def start(self):
        self.control = self.traffic_controller.simulator.vehicle_controller.control
        self.ai.reset()

    def stop(self,failed):
        if failed:
            self.ai.run_epoch(True,True)
        else:
            self.ai.run_epoch(True,False)
    
    def run(self):
        self.ai.run_epoch(False)

    def get_observation(self):
      
        return self.traffic_controller.ai_observation

    
    def modify_control(self,action):
        mod = self.actions[action]
        if mod<-100:
            mod = abs(mod+100)/100
            self.control.brake = mod 
            self.control.throttle = 0
        elif mod <0:
            mod = abs(mod)/100
            self.control.throttle*=mod
        else:
            mod+=100
            mod = mod/100
            self.control.throttle*=mod
        s_obs = self.get_observation()
        # print("episode:", self.ai.episode, "  prev_episode:", self.ai.prev_episode, "  epsilon:", self.ai.epsilon)
        if self.control.throttle == 0:
            print("brake :", self.control.brake, "obs :", s_obs)
        else:
            print("throttle :", self.control.throttle, "obs :", s_obs)
        s_obs = self.get_observation()

        # print(f"Action: {action}, Observation: {obs}")

        # if obs[0]!=100:
        #     if 8<obs[0]<12:
        #         return [self.get_observation(),-abs(obs[1]*50)-obs[0]]
        #     else:
        #         return [self.get_observation(),-obs[0]*3]

        # if obs[2]!=100:
        #     if 8<obs[2]<12:
        #         return [self.get_observation(),obs[3]*50]

        curr_reward = 0
        car_distance = abs(s_obs[0])
        car_delta = s_obs[1]
        car_distance = round(car_distance,2)
        car_delta = round(car_delta,2)
        # other_distance = abs(s_obs[2])
        # other_delta = s_obs[3]
        # if 6<=car_distance<11:
        #     curr_reward += (15-car_distance)*4
        # # elif car_distance<6 and car_delta = 0.0:    
        # #     curr_reward +=30
        # elif car_distance<6:    
        #     curr_reward -=50
        # else:
        #     if car_delta<=0:
        #         curr_reward += 5
        #     else:
        #         curr_reward -= car_distance*4

        # if -0.1<car_delta<0.1 and 6<=car_distance<11:
        #     curr_reward += 30
        # elif 0<=car_delta<0.1 and 6>car_distance:
        #     curr_reward += 50
        # elif 0>car_delta and 6>car_distance:
        #     curr_reward -= 30
        # elif -0.1<car_delta<0.1:
        #     curr_reward += 3
        # # if -0.15<car_delta<=0 and 8<=car_distance<11:
        # #     curr_reward += (1+car_delta)*20
        # elif 0.1<car_delta:
        #     curr_reward -= car_delta*10
        # else:
        #     curr_reward += car_delta*50
        
        # if other_distance<10 and other_delta<0:
            # curr_reward -= other_distance*5


        if car_distance>=11:
            if 0<=car_delta:
                curr_reward -= 50
            elif 0>car_delta:
                curr_reward += 50

        elif 8<car_distance<11:
            if -0.3<=car_delta<=0:
                curr_reward += 45
            elif -0.3>car_delta:
                curr_reward += 15
            elif 0.0<car_delta:
                curr_reward -= 50

        elif 6<=car_distance<=8:
            if -0.1<=car_delta<=0:
                curr_reward += 50
            elif 0.02>car_delta>0:
                curr_reward += 10
            elif 0.1>car_delta>0:
                curr_reward -= 10
            elif 0.1<car_delta:
                curr_reward -= 30
            elif -0.1>car_delta:
                curr_reward -= 60
        
        elif 6>car_distance:
            if 0<=car_delta:
                curr_reward += 30
            elif 0>car_delta:
                curr_reward -= 50
            

        print("reward :",curr_reward)
        return self.get_observation(),curr_reward
            

class SpeedControlAI:

    def __init__(self,environment,input_size=3,action_size=6,save_file='save/model'):

        self.state_size = input_size
        self.action_size = action_size
        self.memory = deque(maxlen=32*30)
        self.gamma = 0.95    # discount rate
        self.learning_rate=0.1
        self.running = True
        self.epsilon = 0.8 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self.build_model()

        self.target_model = self.build_model()

        self.reward_tracker = reward_system.RewardTracker(self,50,70000,prefix='traffic_system')
        self.start =0
        self.load()

        self.update_target_model()

        self.save_file = save_file
        self.environment = environment
        self.prev_state = None
        self.batch_size =32
        self.step =0
        self.start_episode=1

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
        
    def build_model(self):

        model = Sequential()
        model.add(Dense(HIDDEN1_UNITS, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(HIDDEN2_UNITS, activation='tanh'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss = 'mse',optimizer = Adam(lr = self.learning_rate))
        print("built double deep q model for collision control")
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            n = random.randint(0,1)
            return n
        if keys[pygame.K_DOWN]:
            n = random.randint(3,5)
            return n
        if random.random() <= self.epsilon:  
            return random.randrange(self.action_size) 
        # if self.action_choice != -1:
        #     x = self.action_choice
        #     self.action_choice = -1
        #     return x
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) 
        # print(state)
        # act_values = self.model.predict(state)
        # return np.argmax(act_values[0])  # returns index value of o/p action

    def predict(self,state):
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
        
    # def replay(self, batch_size):

    #     minibatch = random.sample(self.memory, batch_size)
    #     for state, action, reward, next_state, done in minibatch:
    #         target = reward
    #         if not done:
    #             target = (reward + self.gamma *
    #                       np.amax(self.model.predict(next_state)[0]))
    #         target_f = self.model.predict(state)
    #         target_f[0][action] = target
    #         self.model.fit(state, target_f, epochs=1, verbose=0)
    #     if self.episode % 30 == 0 and self.start_episode:
    #         self.start_episode = 0
    #         if self.epsilon > self.epsilon_min:
    #             self.epsilon *= self.epsilon_decay
    #         else:
    #             self.epsilon = 0.35
    #     elif self.episode % 30 != 0 :
    #         self.start_episode = 1

    def replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)

        if self.episode % 30 == 0 and self.start_episode:
            self.start_episode = 0
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = 0.35
        elif self.episode % 30 != 0 :
            self.start_episode = 1


            
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay


    def load(self):
        self.episode =0
        last_model,episode,epsilon =self.reward_tracker.get_previous()
        if last_model:
            self.model.load_weights(os.path.join('traffic_system', 'save','models',last_model))
            print("Loaded",last_model)
            self.epsilon = epsilon
            self.episode = episode
            print(self.episode)
            print("Last completed episode : ",self.start)


    def reset(self):
        self.step = 0
        self.total_rewards = 0
        self.prev_state =np.reshape(self.environment.get_observation(),[1,self.state_size]) 

    def run_epoch(self,done=False,failed=False):
        batch_size = self.batch_size
        prev_state = self.prev_state
        action = self.act(prev_state)
        state,reward = self.environment.modify_control(action)
        if failed:
            reward-=1200
        # print("State:"+str(state),"Reward:" + str(reward),sep='\n',end='\n\n')
        state = np.reshape(state, [1, self.state_size])
        self.remember(prev_state, action, reward, state, done)

        if len(self.memory) > batch_size:
            self.replay(batch_size)

        self.prev_state = state
        self.total_rewards+=reward
        if not self.step%20:
            print(f"Step:{self.step}, Rewards: {self.total_rewards}")

        if done:
            self.update_target_model()
            self.reward_tracker.end_episode(self.total_rewards/(self.step+1))
            print(f"\n\n\nComplete Episode {self.episode}, Total Rewards: {self.total_rewards/(self.step+1)}, Epsilon: {self.epsilon}")
            self.episode+=1
        self.step+=1

        return action