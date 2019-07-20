import carla

client  = carla.Client('172.16.175.144',2000)
world = client.get_world()

map_ = world.get_map()

spawn_points = map_.get_spawn_points()
blueprint_library = world.get_blueprint_library()
deb = world.debug
props = list(blueprint_library.filter('sensor.*'))
for i in props:
    print(i.id)
# for prop,s in zip(props,spawn_points):
#     loc = s.location
#     prop.set_attribute('role_name','front')
#     print(prop.tags)
    # aa=world.spawn_actor(prop,s)
    # deb.draw_string(loc,prop.id,life_time=3600)

