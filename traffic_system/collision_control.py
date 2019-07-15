
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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class ControlState(Enum):
    AI=1,
    MANUAL=2

class CollisionControl:

    def __init__(self,lane_ai):
        self.lane_ai = lane_ai
        self.lane_changer = lane_ai.lane_changer
        self.simulator = lane_ai.simulator
        self.state = ControlState.MANUAL
        self.environment = SpeedControlEnvironment(self)
        self.halt_time = pygame.time.get_ticks()

    def update(self):
        self.closest_obstacles = self.lane_ai.lane_closest
        self.apply_condtions()
    
    def apply_condtions(self):
        
        if self.lane_changer.state==lane_ai.State.LANE_CHANGE:
            self.disable_AI()
            self.lane_changing_halt()
            self.lane_changer.update_waypoint()

        elif self.lane_changer.state==lane_ai.State.RUNNING:
            if self.state==ControlState.AI:
                self.modify_by_AI()
            self.same_lane_halt()
            self.check_new_lane()

           
    
    def lane_changing_halt(self):
        target_lane_id = self.lane_changer.target_lane_id

        if target_lane_id in self.closest_obstacles:
            closest_obstacle = self.closest_obstacles[target_lane_id]
            if closest_obstacle.distance<7.5 and closest_obstacle.delta_d<0.005:
                self.halt()
            else:
                self.halt_time = pygame.time.get_ticks()
        self.same_lane_halt(distance=4)

    def same_lane_halt(self,distance=7.5):
        vehicle_lane_id = self.simulator.vehicle_variables.vehicle_waypoint.lane_id

        if vehicle_lane_id in self.closest_obstacles:
            closest_obstacle = self.closest_obstacles[vehicle_lane_id]
            if closest_obstacle.distance<distance and closest_obstacle.delta_d<0.005:
                self.halt()
            else:
                self.halt_time = pygame.time.get_ticks()
    
    def halt(self):
        curr =pygame.time.get_ticks()
        if (curr-self.halt_time)>300000:
            self.simulator.re_level()
            self.halt_time = curr
        else:
            self.disable_AI(failed=2)
            vel = self.simulator.vehicle_variables.vehicle_velocity_magnitude
            if vel>0.05:
                self.simulator.vehicle_controller.destroy_movement()
            control = self.simulator.vehicle_controller.control
            control.throttle = 0
            control.brake = 0.95
    
    def check_new_lane(self):
        vehicle_lane_id = self.simulator.vehicle_variables.vehicle_waypoint.lane_id

        if vehicle_lane_id in self.closest_obstacles:
            closest_obstacle = self.closest_obstacles[vehicle_lane_id]

            if 7.5<closest_obstacle.distance<30 and closest_obstacle.delta_d<0.005 :
                change_lane = self.lane_changer.check_new_lane(min_angle=150)
                if not change_lane:
                    self.enable_AI(closest_obstacle)

    
    def enable_AI(self,closest_obstacle):
        if self.state!=ControlState.AI:
            self.state = ControlState.AI
            self.environment.start(closest_obstacle)
            closest_obstacle.ai_follower = self
            print("Enabled AI")
    
    def disable_AI(self,failed=0):
        if self.state==ControlState.AI:
            self.state = ControlState.MANUAL
            self.environment.stop(failed)
            print("Disabled AI")

    def modify_by_AI(self):
        obstacle = self.environment.obstacle
        if obstacle.road_id!=self.environment.current_road:
            self.disable_AI()
        self.environment.ai.run_epoch()


class SpeedControlEnvironment:

    def __init__(self,traffic_controller):
        self.traffic_controller = traffic_controller
        # pass distance and delta_d
        # reward negative distance
        # self.actions = [30,20,-20,-40,-60,-120,-140]
        # self.ai = SpeedControlAI(self,input_size=4,action_size=7)
        self.actions = [60,30,-90,-150,-160,-190]
        self.ai = SpeedControlAI(self,input_size=5,action_size=6)

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
        other_distance = abs(s_obs[2])
        other_delta = s_obs[3]
        if 6<=car_distance<11:
            curr_reward += (15-car_distance)*4
        # elif car_distance<6 and car_delta = 0.0:    
        #     curr_reward +=30
        elif car_distance<6:    
            curr_reward -=50
        else:
            curr_reward -= car_distance*4

        if -0.1<car_delta<0.1 and 6<=car_distance<11:
            curr_reward += 30
        elif 0<=car_delta<0.1 and 6>car_distance:
            curr_reward += 10
        elif 0>car_delta and 6>car_distance:
            curr_reward -= 3
        elif -0.1<car_delta<0.1:
            curr_reward += 3
        # if -0.15<car_delta<=0 and 8<=car_distance<11:
        #     curr_reward += (1+car_delta)*20
        elif 0.1<car_delta:
            curr_reward -= car_delta*10
        else:
            curr_reward += car_delta*50
        
        # if other_distance<10 and other_delta<0:
            # curr_reward -= other_distance*5



        print("reward :",curr_reward)
        return self.get_observation(),curr_reward
            

class SpeedControlAI:

    def __init__(self,environment,input_size=5,action_size=6,save_file='save/model'):

        self.state_size = input_size
        self.action_size = action_size
        self.memory = deque(maxlen=32*30)
        self.gamma = 0.95    # discount rate
        self.learning_rate=0.1
        self.running = True
        self.epsilon = 0.7 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self.build_model()
        self.reward_tracker = reward_system.RewardTracker(self,50,70000,prefix='traffic_system')
        self.start =0
        self.load()
        self.save_file = save_file
        self.environment = environment
        self.prev_state = None
        self.batch_size =32
        self.step =0
        self.start_episode=1
        self.action_choice = -1
    def build_model(self):

        model = Sequential()
        model.add(Dense(HIDDEN1_UNITS, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(HIDDEN2_UNITS, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss = 'mse',optimizer = Adam(lr = self.learning_rate))
        print("built")
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:  
            return random.randrange(self.action_size) 
        keys = pygame.key.get_pressed()
        brake = 0
        if keys[pygame.K_UP]:
            brake= -1
        if keys[pygame.K_DOWN]:
            brake= 1
        if brake!=0:
            n = random.randint(0,2)
            if brake == 1:
                n += 3
            return n
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
        
    def replay(self, batch_size):

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.episode % 30 == 0 and self.start_episode:
            self.start_episode = 0
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = 0.1
        elif self.episode % 30 != 0 :
            self.start_episode = 1


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
            reward-=5500
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
            self.reward_tracker.end_episode(self.total_rewards/(self.step+1))
            print(f"\n\n\nComplete Episode {self.episode}, Total Rewards: {self.total_rewards/(self.step+1)}, Epsilon: {self.epsilon}")
            self.episode+=1
        self.step+=1

        return action

        
