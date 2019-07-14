import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from keras.engine.training import collect_trainable_weights
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork

class AC_model:

    def __init__(self,simulator,state_size=6,action_size=3):

        #Tensorflow GPU optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        from keras import backend as K
        K.set_session(self.sess)

        self.state_size=state_size
        self.action_size=action_size  #Steering/Acceleration/Brake
        self.simulator=simulator
        self.GAMMA = 0.99
        self.BATCH_SIZE = 32
        self.build_model()
        self.load()
        self.save_batch_size = 200
        self.reward_tracker = reward_system.RewardTracker(self.actor,self.save_batch_size,70000)
        

    def build_model(self):

        BUFFER_SIZE = 100000
        TAU = 0.001     #Target Network HyperParameters
        LRA = 0.0001    #Learning rate for Actor
        LRC = 0.001  

        self.actor = ActorNetwork(self.sess, self.state_size, self.action_size, self.BATCH_SIZE, TAU, LRA)
        self.critic = CriticNetwork(self.sess, self.state_size, self.action_size, self.BATCH_SIZE, TAU, LRC)
        self.buff = ReplayBuffer(BUFFER_SIZE)
        #Create replay buffer

    def load(self):

        last_model,episode,epsilon =self.reward_tracker.get_previous()
        if last_model:
            self.actor.model.load_weights(os.path.join('save','models',last_model))
            self.critic.model.load_weights(os.path.join('save','models','critic_model',last_model))
            self.actor.target_model.load_weights(os.path.join('save','models','actor_target',last_model))
            self.critic.target_model.load_weights(os.path.join('save','models','critic_target',last_model))
            print("Weights load successfully")

            print("Loaded",last_model)
            self.epsilon = epsilon
            self.start = episode
            print("Last completed episode : ",self.start)

    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)

    def replay(self):
        #Do the batch update
    
        batch = self.buff.getBatch(self.BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[1] for e in batch])

        target_q_values = self.critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
        
        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + GAMMA*target_q_values[k]
        
        # if (train_indicator):
        loss += self.critic.model.train_on_batch([states,actions], y_t) 
        a_for_grad = self.actor.model.predict(states)
        grads = self.critic.gradients(states, a_for_grad)
        self.actor.train(states, grads)
        self.actor.target_train()
        self.critic.target_train()

        
    def train_model(self):

        EXPLORE = 100000.
        episode_count = 2000
        max_steps = 300
        reward = 0
        done = False
        step = 0
        epsilon = 1
        indicator = 0

        for i in range(self.start,episode_count):

            print(f"Line follow model: Episode : {i}")
            state = self.simulator.reset() #change to initial state
            state = np.reshape(state, [1, self.state_size])
            self.total_reward = 0.0

            for j in range(max_steps):

                loss = 0 
                epsilon -= 1.0 / EXPLORE
                self.actor.epsilon = epsilon
                a_t = np.zeros([1,self.action_size])
                noise_t = np.zeros([1,self.action_size])
                a_t_original = self.actor.model.predict(state)

                noise_t[0][0] = max(epsilon, 0) * function(a_t_original[0][0],  0.0 , 0.60, 0.30)
                noise_t[0][1] = max(epsilon, 0) * function(a_t_original[0][1],  0.5 , 1.00, 0.10)
                noise_t[0][2] = max(epsilon, 0) * function(a_t_original[0][2], -0.1 , 1.00, 0.05)

                #The following code do the stochastic brake
                #if random.random() <= 0.1:
                #    print("********Now we apply the brake***********")
                #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

                a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
                a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
                a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

                next_state,reward,done,_ = self.simulator.step(a_t[0],2)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.buff.add(state, a_t[0], reward, next_state, done)      #Add replay buffer
                self.replay()

                self.total_reward += reward
                state = next_state
            
                print("Episode", i, "Step", step, "Action", a_t[0], "Reward", reward) #, "Loss", loss)
                step += 1
                if done:
                    break

                if self.simulator.running==False:
                    self.running =False
                    break


            self.curr_episode=self.reward_tracker.curr_episode
            self.reward_tracker.end_episode(self.total_rewards)
            if not self.curr_episode%self.save_batch_size and self.curr_episode!=0:
                f_name = f'model{self.curr_episode}'
                self.critic.model.save_weights( os.path.join('save','models','critic_model',f_name+".data") )   
                self.actor.target_model.save_weights( os.path.join('save','models','actor_target',f_name+".data") )  
                self.critic.target_model.save_weights( os.path.join('save','models','critic_target',f_name+".data") )  

            if self.running==False:
                break   























        # if np.mod(i, 3) == 0:
        #     if (train_indicator):
        #         print("Now we save model")
        #         actor.model.save_weights("actormodel.h5", overwrite=True)
        #         with open("actormodel.json", "w") as outfile:
        #             json.dump(actor.model.to_json(), outfile)

        #         critic.model.save_weights("criticmodel.h5", overwrite=True)
        #         with open("criticmodel.json", "w") as outfile:
        #             json.dump(critic.model.to_json(), outfile)

        # print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        # print("Total Step: " + str(step))
        # print("")



