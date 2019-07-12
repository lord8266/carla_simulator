import carla
import math
def draw_arrows(debug,location_list,life_time=3600,color=carla.Color(255,0,0)):
    
        for i in range(len(location_list)-1):
            debug.draw_arrow(location_list[i],location_list[i+1],life_time=life_time,color=color,arrow_size=5)
            

def draw_lines(debug,location_list,life_time=3600,color=carla.Color(255,0,0)):

    for i in range(len(location_list)-1):
        debug.draw_line(location_list[i], location_list[i+1],life_time=life_time,color=color)

def draw_waypoint(debug,waypoint,life_time=3600,color=carla.Color(255,0,0),length=1):
    l = waypoint.transform.location
    r = waypoint.transform.rotation.yaw
    debug.draw_arrow(l,l+carla.Location(x=math.cos(r)*length, y=math.sin(r)*length),color=color,life_time=0.5,thickness = 0.04,arrow_size=3)

def print_locations(debug,location_list,life_time=3600):

        for i in range(len(location_list)):
                debug.draw_string(location_list[i],f'({location_list[i].x}, {location_list[i].y})',life_time=life_time)