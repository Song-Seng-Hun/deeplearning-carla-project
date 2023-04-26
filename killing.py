import carla

# Connect to the Carla server
client = carla.Client('192.168.0.22', 2000)
client.set_timeout(2.0)

# Get the world
world = client.get_world()

actors = []
# Get all the pedestrian actors
actors += world.get_actors().filter('walker.*')
actors += world.get_actors().filter('sensor.*')
actors += world.get_actors().filter('vehicle.*')
# Destroy all the pedestrian actors
for actor in actors:
    actor.destroy()
