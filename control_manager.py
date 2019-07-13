# throttle = [0 , 0.3,0.7 ,-0.3]
# steer = [-0.5,0,0,5]
# brake =[0,0.5]
import carla

class ControlManager:

    def __init__(self,simulator,throttle_controls= [0,0.5, 0.8],steer_controls=[-0.5,-0.2,0,0.2,0.5],brake_controls=[0,0.5]):
        self.simulator = simulator
        self.throttle_controls = throttle_controls
        self.steer_controls = steer_controls
        self.brake_controls = brake_controls
        self.generate_states()
    
    def generate_states(self):
        self.controls= []
        for t in self.throttle_controls:
            if t==0:
                self.controls.append(carla.VehicleControl(throttle = 0,steer = 0,brake = self.brake_controls[0],reverse = False))
            else:
                for s in self.steer_controls:
                    reverse =False
                    if t<0:
                        reverse =True
                    
                    self.controls.append(carla.VehicleControl( throttle = abs(t),steer = s,brake = self.brake_controls[0],reverse = reverse))
        
        self.action_space = len(self.controls)

    def get_control(self,index):
        return self.controls[index]
    
    def print_all_controls(self):
        for control in self.controls:
            print(control)



c= ControlManager(None)
    

c.print_all_controls()





