import carla
import pygame
import numpy as np

def handle_image(disp, image):
    #image.save_to_disk('output/%05d.png' % image.frame, 
    #   carla.ColorConverter.Raw)
    org_array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    array = np.reshape(org_array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:,:,::-1]
    array = array.swapaxes(0,1)
    surface = pygame.surfarray.make_surface(array)

    disp.blit(surface, (200,0))
    pygame.display.flip()

# Connect to the Carla server
client = carla.Client('192.168.0.4', 2000)
client.set_timeout(2.0)

display = pygame.display.set_mode(
    (1200, 600),
    pygame.HWSURFACE | pygame.DOUBLEBUF
)

# Get the world
world = client.get_world()
