"""
An example of client-side bounding boxes with basic car controls.
Controls:
    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake
    p            : autopilot scenario
    o            : stop scenario
    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
from ultralytics import YOLO
model = YOLO("/home/carla/PythonAPI/examples/src/best.pt")

import cv2
import time
import numpy as np

import tensorflow as tf
from PIL import Image
import colorsys


import glob
import os
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'
        ))[0])
except IndexError:
    pass
import carla
from carla import ColorConverter as cc

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import weakref
import random

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_r
    from pygame.locals import K_p
    from pygame.locals import K_o
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

scenario_flag = False

VIEW_WIDTH = 1280
VIEW_HEIGHT = 720
VIEW_FOV = 150
HOSTIP = 'localhost'

depth_array = []

# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================

        
class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.depth_camera = None
        self.car = None

        self.display = None
        self.image = None
        self.capture = True
        self.tm = None



    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """
        
        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True

        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        
        control.hand_brake = keys[K_SPACE]

        if keys[K_r] : 
            car.set_transform(self.spawn_points[50])
        elif keys[K_p]:
            car.set_autopilot(True, self.tm_port)
            self.tm.ignore_lights_percentage(car,100)
            self.tm.set_path(car, self.route)
            
        if keys[K_o]:
            car.set_autopilot(False, self.tm_port)
            control.brake = 1
        

        car.apply_control(control)
        return False

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """
        try :
            self.car = self.world.get_actors().filter('vehicle.tesla.model3')[0]
        except :
            car_bp = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
            location = random.choice(self.world.get_map().get_spawn_points())
            self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """
        
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=0))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.depth_camera_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        self.depth_camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        self.depth_camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        self.depth_camera_bp.set_attribute('fov', str(VIEW_FOV))
        self.depth_camera = self.world.spawn_actor(self.depth_camera_bp, 
        camera_transform,
        attach_to=self.car,
        attachment_type=carla.AttachmentType.Rigid)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))
        self.depth_camera.listen(lambda image: self.set_depth(image))

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            result = model(array)[0]
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

            font = pygame.font.SysFont(None, 48)


            boxes = result.boxes.boxes
            for box in boxes :
                x1, y1, x2, y2,c,idx = box
                width, height = abs(x2-x1), abs(y2-y1)
                text_surface = font.render(result.names[int(idx)]+", depth : "+str(depth_array[int(min(y1,y2)+height//2)][int(min(x1,x2)+width//2)]), True, (255, 0, 0))
                text_position = (min(x1,x2), min(y1,y2)-30)
                square_position = (min(x1,x2), min(y1,y2))
                pygame.draw.rect(display, (0, 255, 0), (square_position[0], square_position[1], width, height), 3)
                display.blit(text_surface, text_position)

    def set_depth(self, image):
        global depth_array
        image.convert(cc.LogarithmicDepth)
        array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        array = np.reshape(array, (image.height, image.width, 4))
        depth_array = array[:, :, 0]

    def game_loop(self):
        """
        Main program loop.
        """
        
        try:
            pygame.init()
            pygame.font.init()

            
            self.client = carla.Client("localhost", 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()
            self.tm = self.client.get_trafficmanager()
            self.tm.set_synchronous_mode(True)
            self.tm_port = self.tm.get_port()

            # 시나리오 실행을 위해 autopilot 루트 설정
            self.spawn_points = self.world.get_map().get_spawn_points()
            self.route_point = [52, 104, 115, 67, 140, 10, 71, 60, 143, 149, 92, 21, 105, 45, 47, 134, 50]
            self.route = []
            for ind in self.route_point:
                self.route.append(self.spawn_points[ind].location)

            self.setup_car()
            self.setup_camera()

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            
           
            while True:
                self.world.tick()

                self.capture = True
                pygame_clock.tick_busy_loop(20)

                self.render(self.display)
                
                pygame.display.flip()
                pygame.event.pump()
                if self.control(self.car) :
                    return
    

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.car.destroy()
            self.depth_camera.destroy()
            pygame.quit()
            pygame.font.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """
    
    try:
        client = BasicSynchronousClient()
        client.game_loop()
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()