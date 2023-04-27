#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic ped controls.

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


import glob
import os
import sys
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
from carla import ColorConverter as cc

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

BB_COLOR = (248, 64, 24)

BBOX = (VIEW_WIDTH//2-120, VIEW_HEIGHT//2-100, 240, 200)
tracker = None

# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================

class myTracker :
    def __init__(self, frame, bbox) :
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, bbox)
        self.coords = [bbox]
    
    def update(self, frame) :
        success, bbox = self.tracker.update(frame)
        if success :
            self.coords.append(bbox)
        return success, bbox
    
    def get_coords(self) :
        return self.coords
        
class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.ped = None
        self.control = None
        self.display = None
        self.image = None
        self.capture = True
        self.rotation = None
        
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

    def setup_ped(self):
        """
        Spawns actor-human to be controled.
        """

        ped_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        location = random.choice(self.world.get_map().get_spawn_points())
        self.ped = self.world.spawn_actor(ped_bp, location)
        self.control = self.ped.get_control()
        self.rotation = self.ped.get_transform().rotation

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.ped)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration

    def controling(self, ped):
        """
        Applies control to main ped based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """
        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        self.control.speed = 0.0     
        
        if keys[K_s]:
            self.control.speed = 0.0
        if keys[K_a]:
            self.control.speed = .01
            self.rotation.yaw -= 1
        if keys[K_d]:
            self.control.speed = .01
            self.rotation.yaw += 1
        if keys[K_w]:
            self.control.speed = 3.713
        elif keys[K_ESCAPE] :
            self.set_synchronous_mode(False)
            self.camera.destroy()
            ped.destroy()
            pygame.quit()
        self.control.jump = keys[K_SPACE]
        # self.rotation.yaw = round(self.rotation.yaw, 1)
        self.control.direction = self.rotation.get_forward_vector()

        ped.apply_control(self.control)
        return False

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
        global tracker
        if self.image is not None:
            x1, y1 = 120, 130
            x2, y2 = 200, 300
            width, height = abs(x2-x1), abs(y2-y1)
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))
            
            if tracker is None :
                tracker = myTracker(array, BBOX)
            else :
                success, bbox = tracker.update(array)
                if success :
                    pygame.draw.rect(display, (0, 255, 0), (bbox[0], bbox[1], bbox[2], bbox[3]), 3)
                else :
                    del tracker
                    tracker = None
            
    def game_loop(self):
        """
        Main program loop.
        """

        try:
            pygame.init()

            self.client = carla.Client('192.168.0.4', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()

            self.setup_ped()
            self.setup_camera()

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            vehicles = self.world.get_actors().filter('vehicle.*')

            while True:
                self.world.tick()
                self.capture = True

                self.render(self.display)
                
                pygame.display.flip()
                pygame.event.pump()
                if self.controling(self.ped):
                    return

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.ped.destroy()
            pygame.quit()


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
