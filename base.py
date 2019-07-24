


import Simulator
import pygame
import numpy as np
from sklearn.preprocessing import StandardScaler
import ai_model
# import ac_model



def get_action():
    max_ = len(simulator.control_manager.controls)
    return np.random.randint(0,max_)


simulator = Simulator.Simulator('172.16.175.144')
# simulator = Simulator.Simulator()


model = ai_model.Model(simulator,6,len(simulator.control_manager.controls))
# ac_model=ac_model.AC_model(simulator,6,3)
# model=ac_model.actor.model

running = simulator.running
observation = simulator.get_observation()
prev = pygame.time.get_ticks()
curr_reward =0 
clock = pygame.time.Clock()


# while running:
#     clock.tick_busy_loop(60)
#     action = model.predict(observation)
#     # print(observation)
#     curr = pygame.time.get_ticks()
#     observation,reward,done,_ = simulator.step(action)
#     # observation,reward,done,_ = simulator.step(action,2)
#     curr_reward+=reward
#     if (curr-prev)>100:
#         # print("Reward: ",simulator.reward_system.curr_reward)
#         # print(observation, end='\n\n')
#         # print(simulator.vehicle_controller.control)
#         # print(simulator.vehicle_variables.vehicle_location,simulator.navigation_system.start.location)
#         prev =curr
#     # print(1000/(curr-prev))
#     # prev = curr
#     if done:
#         simulator.reset()
#         continue

#     simulator.render()
#     running = simulator.running



while running:
    # try:
        running = simulator.running
        # try:
        # model = ai_model.Model(simulator,4,len(simulator.control_manager.controls))
        model.train_model()
        # ac_model.train_model()
        running = simulator.running
            
    # except Exception as e:
    #     simulator.re_level()
    #     running = simulator.running
    #     print("local" + str(e))

simulator.stop()
pygame.quit()


