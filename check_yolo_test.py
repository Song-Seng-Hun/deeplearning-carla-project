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
import random, time

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
HOSTIP = 'localhost'
BB_COLOR = (248, 64, 24)

tracker_list = []
depth_array = []

TRACKER_LIMIT = 5
COORD_LIMIT = 10
W_THRES, H_THRES = 100, 100
# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================
class myTracker :
    global depth_array
    def __init__(self, frame, bbox, idx, depth) :
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, bbox)
        self.bbox = bbox
        self.coords = [bbox]
        self.idx = idx
        self.depth = depth
        self.color = (0, 255, 0)
        
    def update(self, frame) :
        success, bbox = self.tracker.update(frame)
        self.depth = depth_array[bbox[1]+bbox[3]//2][bbox[0]+bbox[2]//2]
        if success :
            self.coords.append(bbox)
            self.coords = self.coords[-COORD_LIMIT:]
        return success, bbox, self.idx, self.depth
    
    def set_idx(self, idx) :
        self.idx = idx
    
    def get_info(self) :
        print(self.coords)
        return self.coords, self.idx, self.depth
    
    def __del__(self) :
        try :
            with open('log.txt', 'a') as f :
                f.write(str(self.coords) + '\t' + str(self.depth) + '\t' + str(self.idx) + '\n')
        except :
            pass
        
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
        self.raw_image = None
        self.capture = True

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
    
    def depth_camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
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
        self.depth_camera.listen(lambda image: weak_self().set_depth(weak_self, image))

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
        global tracker_list
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """
        
        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            result = model(array)[0]
            array = array[:, :, ::-1]
            self.raw_image = cv2.cvtColor(array,cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))     
            display.blit(surface, (0, 0))
            font = pygame.font.SysFont(None, 48) 
            
            boxes = result.boxes.data
            do_create_box = True
            for box in boxes :
                x1, y1, x2, y2,c,idx = box
                center = (int((x1+x2)//2), int((y1+y2)//2))
                
                if len(tracker_list) > 0 :
                    for tracker in tracker_list :
                        bbox = tracker.bbox
                        tracker_center = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
                        if abs(tracker_center[0]-center[0])<W_THRES and abs(tracker_center[1]-center[1])<H_THRES:
                            if tracker.idx == int(idx) :
                                do_create_box = False
                                break
                            else :
                                tracker.idx = int(idx)
                                do_create_box = False
                                break
                        elif len(tracker_list) > TRACKER_LIMIT :
                            do_create_box = False
                            print([tracker.bbox for tracker in tracker_list])
                            
                if do_create_box is True :
                    tracker_list.append(myTracker(array, (int(x1), int(y1), int(abs(x2-x1)), int(abs(y2-y1))), idx, depth_array[center[1]][center[0]]))
                
            for i in range(len(tracker_list)) :
                if i >= len(tracker_list) :
                    break
                tracker = tracker_list[i]
                do_create_box = True
                success, bbox, idx, depth = tracker.update(array)
                if (success is False and bbox[2]<W_THRES//depth//depth and bbox[3]<H_THRES//depth//depth) or bbox[0]+bbox[2]<0 or bbox[0]>VIEW_WIDTH or bbox[1]+bbox[3]>VIEW_HEIGHT or bbox[1]<0 or bbox[2]<20 or bbox[3]<20 :
                    tracker_list.pop(i)
                    del tracker
                    i -= 1
                    
                else :
                    text_surface = font.render(result.names[int(idx)]+' depth : '+str(depth), True, tracker_list[i].color)
                    text_position = (bbox[0], bbox[1]-30)
                    pygame.draw.rect(display, tracker.color, (bbox[0], bbox[1], bbox[2], bbox[3]), 3)
                    display.blit(text_surface, text_position)
                    
    @staticmethod
    def set_depth(weak_self, image):
        global depth_array
        self = weak_self()
        image.convert(cc.LogarithmicDepth)
        array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        array = np.reshape(array, (image.height, image.width, 4))
        depth_array = array[:, :, 0]

    def game_loop(self, num_classes, input_size, graph):
        """
        Main program loop.
        """
        
        try:
            pygame.init()
            
            self.client = carla.Client("localhost", 2000)
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
                    time.sleep(0.01)
                    self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
                    
                    pygame.display.flip()
                    pygame.event.pump()
                    if self.control(self.car):
                        return

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.depth_camera.destroy()
            self.car.destroy()
            for tracker in tracker_list :
                del tracker
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
