import numpy as np
from collections import deque
from keras import Sequential
from keras.layers import Dense,Conv2D,Flatten
from keras.optimizers import sgd,Adam
import os
import random
import reward_system
import pygame
HIDDEN1_UNITS = 50
HIDDEN2_UNITS = 40

class Model:

    def __init__(self,simulator,state_size=3,action_size=6,save_file='save/model'):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=32*30)
        self.gamma = 0.95    # discount rate
        self.learning_rate=0.0025
        self.running = True
        self.epsilon = 0.4  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.simulator =simulator
        self.model = self.build_model()
        self.reward_tracker = reward_system.RewardTracker(self,200,70000)
        self.start =0
        self.load()
        self.save_file = save_file
        self.simulator.ai_model = self
        self.prev = pygame.time.get_ticks()
    def build_model(self):

        model = Sequential()
        model.add(Dense(HIDDEN1_UNITS, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(HIDDEN2_UNITS, activation='tanh'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss = 'mse',optimizer = Adam(lr = self.learning_rate))
        # self.load('./save/Carla-dqn.h5')
        print("built")
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # if random.random() <= self.epsilon:  
        #     return random.randrange(self.action_size) 
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns index value of o/p action
        # key_state,action = self.simulator.vehicle_controller.check_key_state()
        # self.simulator.key_control =key_state
        # if key_state:
        #     # print("Imitate Activated")
        #     return action
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
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay


    def load(self):
        last_model,episode,epsilon =self.reward_tracker.get_previous()
        if last_model:
            self.model.load_weights(os.path.join('save','models',last_model))
            print("Loaded",last_model)
            self.epsilon = epsilon
            self.start = episode
            print("Last completed episode : ",self.start)

    def save(self, name):
        self.model.save_weights(os.path.join('save','models',name))


    def train_model(self):

        done = False
        batch_size = 32
        EPISODES = 700000
        prev_rewards =0
        for e in range(self.start,EPISODES):
            state = self.simulator.reset() #change to initial state
            state = np.reshape(state, [1, self.state_size])
            self.total_rewards = 0
            
            for time in range(300):
                if not time%50:
                    pass
                    # print(f"Step {time}, Rewards: {self.total_rewards}")
                # env.render()
                action = self.act(state) # self.act(state)
                # next_state, reward, done, _ = env.step(action)
                next_state,reward,done,_ = self.simulator.step(action) #check
                self.total_rewards += reward

                # reward = reward if not done else -10 #check
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
                if len(self.memory) > batch_size:
                    self.replay(batch_size)
                    
                if self.simulator.running==False:
                    self.running =False
                    break
                
            self.reward_tracker.end_episode(self.total_rewards)
            # print(f"Complete Episode {e} , Epsilon: {self.epsilon}, Total Rewards: {self.total_rewards},Position: {self.simulator.navigation_system.curr_pos} / {len(self.simulator.navigation_system.ideal_route)} ")
            
            if self.running==False:
                break
            if e%30==0:
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
            prev_rewards =self.total_rewards
    

class FreeRoad:

    def __init__(self,control):
        self.control = control
        model = Sequential()
        model.add(Conv2D(64, kernel_size=4, activation='relu', input_shape=(100,200,1)))
        model.add(Conv2D(32, kernel_size=4, activation='relu'))
        model.add(Flatten())
        model.add(Dense(40, activation='tanh'))
        model.add(Dense(30, activation='tanh'))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        self.model.load_weights('images/pixels.data')
        self.prev = pygame.time.get_ticks()
    def predict(self,pixels):
        curr =pygame.time.get_ticks()

        if (curr-self.prev)>700:
            self.prev = curr
            pixels = pixels.reshape( (100,200,1) )
            p = self.model.predict( [[pixels]])
            print(p)
            return p


