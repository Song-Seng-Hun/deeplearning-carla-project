#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.
Controls:
    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake
    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
from ultralytics import YOLO
model = YOLO("yolov8m-pose.pt")


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
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90
HOSTIP = '192.168.0.4'
BB_COLOR = (248, 64, 24)

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
        self.car = None

        self.display = None
        self.image = None
        self.raw_image = None
        self.capture = True

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
            self.car = self.world.get_actors().filter('vehicle.*')[0]
        except :
            car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
            location = random.choice(self.world.get_map().get_spawn_points())
            self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """
        #camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        #First person view transform settings
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=0))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration

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
            array = array[:, :, ::-1]
            self.raw_image = cv2.cvtColor(array,cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def game_loop(self, num_classes, input_size, graph):
        """
        Main program loop.
        """
        
        try:
            pygame.init()
            
            self.client = carla.Client(HOSTIP, 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()

            self.setup_car()
            self.setup_camera()

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            
            with tf.compat.v1.Session(graph=graph) as sess:
                while True:
                    self.world.tick()
    
                    self.capture = True
                    pygame_clock.tick_busy_loop(20)
    
                    self.render(self.display)
                    
                    
                    self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
                    frame_size = self.raw_image.shape[:2]
                    
                    image_data = self.raw_image
                    image_data = image_data[np.newaxis, ...]
                    print(image_data.shape)
                    ##### 여기서 YOLO 처리해야
                    # model(image_data)
                    
                    pygame.display.flip()
                    pygame.event.pump()

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            # self.car.destroy()
            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """
    
    try:
        # video_path      = 0
        num_classes     = 80
        input_size      = 416
        graph           = tf.Graph()
        
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        
        client = BasicSynchronousClient()
        client.game_loop(num_classes, input_size, graph)
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()




