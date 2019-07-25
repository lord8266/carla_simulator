
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
HIDDEN1_UNITS = 26
HIDDEN2_UNITS = 20
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class LaneState(Enum):
    SAME_LANE=1,
    LANE_CHANGE=2,
class ControlState(Enum):
    AI=1,
    MANUAL=2

class CollisionControl:

    def __init__(self,trc):
        self.traffic_controller = trc
        self.control = trc.simulator.vehicle_controller.control
        self.state = LaneState.SAME_LANE
        self.curr_lane = self.traffic_controller.simulator.vehicle_variables.lane_id
        self.check_completion=False
        self.last_state = self.state
        self.prev_time = pygame.time.get_ticks()
      

    def update(self):
        # if self.state==LaneState.SAME_LANE:
        #     self.update_same_lane()
        # else:
        #     self.update_target_lane()

        self.update_lane_change()
        self.try_lane_change()
        self.check_lane_change_completion()
     




    
    def modify_control(self,action,distance,delta_d):
        return
        self.traffic_controller.simulator.override = True
        if action==0:
            self.control.throttle = 0.0
            self.control.brake = 1.0
        elif action==1:
            self.control.throttle =0.3
        
        elif action==2:
            self.control.throttle/=2

        # self.collect_data([distance,delta_d],action)

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
            target =self.target

            if curr.lane_id==target:
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

                if 10<obs_same.distance<30:
                    passed,prev_,next_ = self.change_lane()
                    if passed:
                        lane_id =next_.lane_id
                        if lane_id in lane_obstacles:
                            if lane_obstacles[lane_id]:
                                obs_next = lane_obstacles[lane_id][0]
                                if (obs_next.distance-obs_same.distance)<2:
                                    passed = False
                        
        if passed:
            self.check_completion = True
            self.traffic_controller.simulator.navigation_system.add_event(prev_,next_)
            self.target = next_.lane_id
            return True,self.target

        return False,None

    def try_lane_change2(self,obs_same):

        lane_id = self.traffic_controller.simulator.vehicle_variables.lane_id
        lane_obstacles = self.traffic_controller.lane_obstacles
        passed = False
        passed,prev_,next_ = self.change_lane()
        if passed:
            lane_id =next_.lane_id
            if lane_id in lane_obstacles:
                if lane_obstacles[lane_id]:
                    obs_next = lane_obstacles[lane_id][0]
                    if (obs_next.distance-obs_same.distance)<2:
                        passed = False
                        
        if passed:
            self.check_completion = True
            self.traffic_controller.simulator.navigation_system.add_event(prev_,next_)
            self.target = next_.lane_id
            return True,self.target

        return False,None

    def change_lane(self):
        return self.traffic_controller.simulator.lane_ai.request_new_lane()


class SpeedControlEnvironment:

    def __init__(self,traffic_controller):
        self.traffic_controller = traffic_controller
        self.actions = [50,10,-85,-130,-165,-199]
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

    
    def modify_control(self,action,step_observation):
        mod = self.actions[action]
        if mod<-100:
            mod = abs(mod+100)/100
            self.control.brake = mod 
            self.control.throttle = 0.0
        elif mod <0:
            mod = abs(mod)/100
            self.control.throttle*=mod
            self.control.brake = 0.0
        else:
            mod+=100
            mod = mod/100
            self.control.throttle*=mod
            self.control.brake = 0.0
        s_obs = step_observation[0]
        # print("episode:", self.ai.episode, "  prev_episode:", self.ai.prev_episode, "  epsilon:", self.ai.epsilon)
        # if self.control.throttle == 0:
        #     print("brake :", self.control.brake, "obs :", s_obs)
        # else:
        #     print("throttle :", self.control.throttle, "obs :", s_obs)


        curr_reward = 0
        car_distance = abs(s_obs[0])*3
        car_delta = s_obs[1]/10
        if True:
            pass
        elif car_distance>50:
            self.control.throttle = 0.99
            self.control.brake = 0.0

        elif car_distance>15:
            if self.control.throttle >= 0.5:
                curr_reward += 5
            elif self.control.throttle > 0.0:
                curr_reward += 3
            else:
                curr_reward -= 5

        elif 15>=car_distance>=11:
            # if self.control.throttle >= 0.5 and car_delta<-0.4:
            #     curr_reward += 1.25
            if car_delta<-0.5:
                curr_reward += 2
            elif -0.5<=car_delta<-0.2:
                curr_reward += 5
            elif -0.2<=car_delta<-0.01:
                curr_reward += 3.5
            elif -0.01<=car_delta<0.2:
                curr_reward -= 4
            elif 0.2<=car_delta:
                curr_reward -= 5 

        elif 8<car_distance<11:
            # if self.control.throttle >= 0.3 and -0.4<=car_delta<0.2:
            #     curr_reward += 1.25
            if car_delta<-0.5 and self.control.throttle > 0.5:
                curr_reward -= 6
            elif car_delta<-0.5:
                curr_reward += 2 
            elif -0.5<=car_delta<-0.2 and self.control.throttle > 0.5:
                curr_reward -= 4
            elif -0.5<=car_delta<-0.2:
                curr_reward += 1.5
            elif -0.2<=car_delta<-0.09:
                curr_reward += 3
            elif -0.09<=car_delta<0.09 and s_obs[2]>3.75:
                curr_reward += 2
            elif 0.09<=car_delta<0.2:
                curr_reward -= 2
            elif 0.2<=car_delta:
                curr_reward -= 4

        elif 6.3<=car_distance<=8:
            if car_delta<-0.5 and self.control.throttle > 0.0:
                curr_reward -= 10
            elif car_delta<-0.5:
                curr_reward += 1
            elif -0.5<=car_delta<-0.2 and self.control.throttle > 0.0:
                curr_reward -= 10
            elif -0.5<=car_delta<-0.2:
                curr_reward += 2
            elif -0.2<=car_delta<-0.08 and self.control.throttle > 0.5:
                curr_reward -= 4
            elif -0.2<=car_delta<-0.08:
                curr_reward += 2
            elif -0.08<=car_delta<0.0 and self.control.throttle > 0.0:
                curr_reward -= 20
            elif 0.0<=car_delta<0.08 and self.control.throttle > 0.0:
                curr_reward += 10
            elif -0.08<=car_delta<0.08:
                curr_reward += 3
            elif 0.08<=car_delta<0.2:
                curr_reward -= 3
            elif 0.1<=car_delta:
                curr_reward -= 4
        
        elif 6.3>car_distance>4.8:
            if self.control.brake > 0.0:
                curr_reward += 20
            else:
                curr_reward -= 20
            # if self.control.brake > 0.0 and car_delta<0.0:
            #     curr_reward += 1.25
            # if car_delta<-0.5:
            #     curr_reward -= 10 
            # elif -0.5<=car_delta<-0.2 and self.control.throttle > 0.5:
            #     curr_reward -= 8
            # elif -0.5<=car_delta<-0.2:
            #     curr_reward += 1
            # elif -0.2<=car_delta<-0.04 and self.control.throttle > 0.5:
            #     curr_reward -= 1
            # elif -0.2<=car_delta<-0.02:
            #     curr_reward += 3
            # elif -0.02<=car_delta<0 and self.control.throttle > 0.5:
            #     curr_reward -= 3
            # elif -0.02<=car_delta<0.04:
            #     curr_reward += 6
            # elif 0.04<=car_delta<0.2:
            #     curr_reward -= 0.5
            # elif 0.1<=car_delta:
            #     curr_reward -= 2

        elif 4.8>=car_distance:
            if self.control.brake > 0.0:
                curr_reward += 25
            else:
                curr_reward -= 25
                
        # print("\t\tobs :"," %5.2f"%(car_distance),"  %5.2f"%(car_delta),"  %5.2f"%(s_obs[2]), end="")
        return self.get_observation(),curr_reward
            

class SpeedControlAI:

    def __init__(self,environment,input_size = 3,action_size=6,save_file='save/model'):

        self.state_size = input_size
        self.action_size = action_size
        self.memory = deque(maxlen=32*4)
        self.gamma = 0.95    # discount rate
        self.learning_rate=0.1
        self.running = True
        self.epsilon = 0.5 # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.model = self.build_model()

        self.target_model = self.build_model()

        self.reward_tracker = reward_system.RewardTracker(self,25,100000,prefix='traffic_system')
        self.start =0
        self.load()

        self.update_target_model()

        self.save_file = save_file
        self.environment = environment
        self.prev_state = None
        self.batch_size = 32*4
        self.step =0
        self.start_episode=1
        self.random = random.random()
        self.randomcounter = 0
        # self.collector = collision_data_collector.DataCollector(environment.traffic_controller.simulator,'collision_data',500,100,100)
        # self.image_collector = collision_data_collector.DataCollector(environment.traffic_controller.simulator,'collision_images',600,25,100)
        self.prev_data = 20

        self.prev_time = pygame.time.get_ticks()

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
        # print("\nTarget model updated")
        
    def build_model(self):

        model = Sequential()
        model.add(Dense(HIDDEN1_UNITS, input_dim=self.state_size, activation='relu'))
        model.add(Dense(HIDDEN2_UNITS, activation='relu'))
        model.add(Dense(self.action_size, activation='tanh'))
        model.compile(loss = 'mse',optimizer = Adam(lr = self.learning_rate))
        print("built double deep q model for collision control")
        return model


    def remember(self, state, action, reward, next_state, done):
        # print("memory: ",(state, action, reward, next_state, done))
        self.memory.append((state, action, reward, next_state, done))

    def collect_data(self,state,target):

        curr =pygame.time.get_ticks()

        if (curr-self.prev_time)>300:
            image = self.environment.traffic_controller.simulator.game_manager.array
            self.image_collector.save_image(image,target)
            self.prev_time = curr

        if state[0][0]!=100 and state[0][0]!=self.prev_data:
            self.prev_data = state[0][0]
            image = self.environment.traffic_controller.simulator.game_manager.array

            self.collector.save_image(state,target)
            
        else:
            print("Not Saving")

    def act(self, state):
        # if self.randomcounter == 0:
        #     self.random = random.random()
        # self.randomcounter += 1
        # self.randomcounter %= 5

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            # print("control :",end = " ")
            if keys[pygame.K_1]:
                return 2
            elif keys[pygame.K_2]:
                return 1
            elif keys[pygame.K_3]:
                return 0
            else:
                return random.randint(0,2)
        if keys[pygame.K_DOWN]:
            # print("control :",end = " ")
            if keys[pygame.K_1]:
                return 3
            elif keys[pygame.K_2]:
                return 4
            elif keys[pygame.K_3]:
                return 5
            else:
                return random.randint(3,5)


        if True:#random.random()<self.epsilon:
            # print("action  :",end = " ")
            #return random.randint(0,5)
            s_obs = state[0]
            car_distance = abs(s_obs[0])*3
            car_delta = s_obs[1]/10
            car_speed = s_obs[2]

            if 20<=car_distance and 9<car_speed:
                return 1

            elif 11<=car_distance:
                if car_speed > 9:
                    return 3
                if car_delta<-0.5:
                    return 2
                elif -0.5<=car_delta<-0.2:
                    return 1
                else:
                    return 0
                # elif -0.2<=car_delta<0.05:
                #     return 0
                # elif 0.05<=car_delta<0.5:
                #     return 0
                # elif 0.5<=car_delta:
                #     return 0
            
            elif 8<car_distance<11:
                if car_speed > 6 and car_delta<-0.2:
                    return 4
                elif car_speed > 6 and -0.2<=car_delta<0.05:
                    return 3
                if car_delta<-0.5:
                    return 4
                elif -0.5<=car_delta<-0.3:
                    return 3
                elif -0.3<=car_delta<0.05:
                    return 2
                elif 0.05<=car_delta<0.5:
                    return 1
                elif 0.5<=car_delta:
                    return 1

            elif 6<=car_distance<=8:
                if car_speed > 6:
                    return 5
                if car_delta<-0.2:
                    return 5
                elif -0.2<=car_delta<-0.05:
                    return 5
                elif -0.05<=car_delta<=-0.02:
                    return 4
                elif -0.02<car_delta<0.02:
                    return 2
                elif 0.02<=car_delta<0.1:
                    return 1
                elif 0.1<=car_delta:
                    return 1
            
            elif 4.8<=car_distance<6:
                if car_speed > 3:
                    return 5
                if car_delta<-0.5:
                    return 5
                elif -0.5<=car_delta<-0.2:
                    return 5
                elif -0.2<=car_delta<0.01:
                    return 5
                elif 0.01<=car_delta<0.1:
                    return 2
                elif 0.1<=car_delta:
                    return 1

            elif car_distance<4.8:
                return 5
        # print("predict: ",end=" ")
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) 

    def predict(self,state):
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
        
    # def replay(self, batch_size):

    #     minibatch = self.memory
    #     # minibatch = random.sample(self.memory, batch_size)
    #     # minibatch=np.array(minibatch)
    #     # rewards=minibatch[:,2]
    #     # mean=np.mean(rewards)
    #     # std=np.std(rewards)
    #     # minibatch[:,2]=(rewards-mean)/(std+1e-10)

    #     for state, action, reward, next_state, done in minibatch:
    #         target = reward
    #         if not done:
    #             target = (reward + self.gamma *
    #                       np.amax(self.model.predict(next_state)[0]))
    #         target_f = self.model.predict(state)
    #         target_f[0][action] = target
    #         self.model.fit(state, target_f, epochs=1, verbose=0)
    #     if self.episode % 20 == 0 and self.start_episode:
    #         self.start_episode = 0
    #         if self.epsilon > self.epsilon_min:
    #             self.epsilon *= self.epsilon_decay
    #         else:
    #             self.epsilon = 0.4
    #     elif self.episode % 20 != 0 :
    #         self.start_episode = 1

    def replay(self, batch_size):
        minibatch = self.memory
        # minibatch = random.sample(self.memory, batch_size)
        # minibatch=np.array(minibatch)
        # rewards=minibatch[:,2]
        # print("rewards: ",rewards[2:15])
        # mean=np.mean(rewards)
        # std=np.std(rewards)
        # minibatch[:,2]=(rewards-abs(mean))/(std+1e-10)
        # print("normalised rewards: ",(minibatch[:,2])[2:15])
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            # print("\ntarget",reward,t)
            # print("\nexpected: ",target)
            self.model.fit(state, target, epochs=1, verbose=0)

        if self.episode % 20 == 0 and self.start_episode:
            self.start_episode = 0
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon = 0.8
        elif self.episode % 20 != 0 :
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
        prev_state = self.prev_state
        action = self.act(prev_state)
        # print(action, end="   ")
        # self.collect_data(prev_state,action)
        state,reward = self.environment.modify_control(action,prev_state)
        state = np.reshape(state, [1, self.state_size])
        # print()

        # if failed and action<3 and prev_state[0][0]<50:
        #     print("\nhit")
        #     reward -= 30
        
        # self.remember(prev_state, action, reward, state, done)
        # if len(self.memory) >= self.batch_size:
        #     print("replay function called")
        #     self.replay(self.batch_size)

        self.prev_state = state

        # self.total_rewards+=reward
        # if done:
        #     # self.update_target_model()
        #     self.reward_tracker.end_episode(self.total_rewards/(self.step+1))
        #     print(f"\n\n\nComplete Episode {self.episode}, Total Rewards: {self.total_rewards/(self.step+1)}, Epsilon: {self.epsilon}")
        #     self.episode+=1
        # self.step+=1

        return action