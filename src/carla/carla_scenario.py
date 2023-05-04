"""
Controls:
    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake
    R            :
    P            : autopilot scenario
    O            : stop scenario
    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
import glob
import os
import sys

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

from ultralytics import YOLO
# yolov8 pt 모델 경로 입력
model = YOLO("/home/carla/PythonAPI/test/best_ver8.pt")

import cv2
import time
import numpy as np

import tensorflow as tf
from PIL import Image
import colorsys

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

# 카메라 시야 조정
VIEW_WIDTH = 1280
VIEW_HEIGHT = 720
VIEW_FOV = 130
HOSTIP = 'localhost'

        
class BasicSynchronousClient(object):


    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.depth_camera = None
        self.car = None
        self.walker = None
        self.scenario_flag = False
        self.bounding_boxes = None
        
        self.display = None
        self.image = None
        self.depth_image = None
        self.cctv_image = None
        self.tm = None
        
        self.cnt = -9999

# ===================================================================================
# -- 키보드 이벤트 --------------------------------------------------------------------
# ===================================================================================
    def control(self, car):
        """
        키보드 event
        차량 조종에 사용
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
            car.set_transform(self.spawn_points[104])
            # spawn 시 이전 운동에 대한 관성 방지(raycast 사용)
            self.modify_physics(car)
        elif keys[K_p]:
            self.scenario_flag = True
            
        if keys[K_o]:
            car.set_autopilot(False, self.tm_port)
            self.scenario_flag = False
            control.brake = 1

        if car.get_velocity().length() == 0 :
            control.brake = 0
        
        # 시나리오 실행시 우회전 로직
        if self.scenario_flag == True :
            # autopilot mode(신호등, 보행자 무시)
            car.set_autopilot(True, self.tm_port)
            self.tm.ignore_lights_percentage(car,100)
            self.tm.ignore_walkers_percentage(car,100)
            self.tm.random_left_lanechange_percentage(car, 0)
            self.tm.random_right_lanechange_percentage(car, 0)
            self.tm.keep_right_rule_percentage(car, 100)
            self.tm.vehicle_percentage_speed_difference(car, 30)
            self.tm.set_path(car, self.route)

            # 가장 큰 수를 초기값으로 설정
            red_min_depth = float('inf')
            ped_min_depth = float('inf')

            #YOLO 모델로 신호등, 보행자 감지
            for box in self.bounding_boxes:
                x1, y1, x2, y2, c, idx = box
                width, height = abs(x2-x1), abs(y2-y1)
                if idx == 1:
                    ped_depth = self.depth_image[int(min(y1, y2) + height // 2)][int(min(x1, x2) + width // 2)]
                    if ped_depth < ped_min_depth:
                        ped_min_depth = ped_depth

                if idx == 2:
                    red_depth = self.depth_image[int(min(y1, y2) + height // 2)][int(min(x1, x2) + width // 2)]
                    if red_depth < red_min_depth:
                        red_min_depth = red_depth
            
            # 빨간불 감지시 일시정지 후 주행
            if red_min_depth != float('inf') :
                if red_min_depth < 60 :
                    car.set_autopilot(False, self.tm_port)

                    if self.cnt < -40 :
                        self.cnt = 60
                        control.brake = self.cnt
                    elif self.cnt > 0 :
                        control.brake = self.cnt
                    else :
                        control.brake = 0
                        car.set_autopilot(True, self.tm_port)
            self.cnt -= 1
            if self.cnt < -9999 :
                self.cnt = -9999

                    
            print(self.cnt)

            # 보행자 감지시 brake
            if ped_min_depth != float('inf') :
                if ped_min_depth < 40 :
                    car.set_autopilot(False, self.tm_port)
                    control.brake = 1

        car.apply_control(control)
        return False
    
    def modify_physics(self, actor):
        try :
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except :
            pass

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

# ===================================================================================
# -- 차량 및 보행자 설정 ---------------------------------------------------------------
# ===================================================================================

    def setup_car(self):
        """
        조종할 차량 소환
        """
        try :
            self.car = self.world.get_actors().filter('vehicle.tesla.model3')[0]
        except :
            car_bp = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
            location = random.choice(self.world.get_map().get_spawn_points())
            self.car = self.world.spawn_actor(car_bp, location)

    def setup_walker(self) :
        """
        보행자 소환
        """      
        #walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.pedestrian.0042'))
        rotation = carla.Rotation(pitch=0, yaw=180.0, roll=0.0)
        transform = carla.Transform(self.crosswalks[16], rotation)
        self.walker = self.world.spawn_actor(walker_bp, transform)

        self.walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

        self.controller = self.world.spawn_actor(self.walker_controller_bp, carla.Transform(), self.walker)
        self.controller.start()
        self.controller.set_max_speed(2)
        self.controller.go_to_location(self.crosswalks[18])

    def walker_scenario(self) :
        """
        보행자 시나리오 루트 설정
        """    
        if (self.walker.get_location().distance(self.crosswalks[19]) < 2) or (self.walker.get_location().distance(self.crosswalks[16]) < 2)  :
            self.controller.go_to_location(self.crosswalks[18])
            # time.sleep(1.0)
        if (self.walker.get_location().distance(self.crosswalks[17]) < 2) or (self.walker.get_location().distance(self.crosswalks[18]) < 2)  :
            self.controller.go_to_location(self.crosswalks[19])
            # time.sleep(1.0)

# ===================================================================================
# -- rgb 카메라 와 depth 카메라 설정 ----------------------------------------------------
# ===================================================================================

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
        Returns depth_camera blueprint.
        """

        depth_camera_bp = self.world.get_blueprint_library().find('sensor.camera.depth')
        depth_camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        depth_camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        depth_camera_bp.set_attribute('fov', str(VIEW_FOV))

        return depth_camera_bp
    
    
    def setup_camera(self):
        """
        차량에 붙일 rgb camera 설정
        """
        
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=0))
        self.camera = self.world.spawn_actor(self.camera_blueprint(),
                                             camera_transform, 
                                             attach_to=self.car,
                                             attachment_type=carla.AttachmentType.Rigid)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration

    def setup_depth_camera(self) :
        """
        차량에 붙일 depth camera 설정
        """

        depth_camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=0))
        self.depth_camera = self.world.spawn_actor(self.depth_camera_blueprint(),
                                                   depth_camera_transform,
                                                   attach_to=self.car,
                                                   attachment_type=carla.AttachmentType.Rigid)
        weak_self = weakref.ref(self)
        self.depth_camera.listen(lambda image: weak_self().set_depth_image(weak_self,image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration


    @staticmethod
    def set_image(weak_self, img):

        self = weak_self()
        if not self:
            return

        self.image = img
  

    @staticmethod
    def set_depth_image(weak_self, image):

        self = weak_self()
        if not self:
            return
        
        image.convert(cc.LogarithmicDepth)
        array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        array = np.reshape(array, (image.height, image.width, 4))
        self.depth_image = array[:, :, 0]


    # 영상처리
    def render(self, display):
        """
        Transforms image from camera, depth sensor and blits it to main pygame display.
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

            self.bounding_boxes = result.boxes.boxes
            for box in self.bounding_boxes :
                x1, y1, x2, y2,c,idx = box
                width, height = abs(x2-x1), abs(y2-y1)
                text_surface = font.render(result.names[int(idx)]+", depth : "+str(self.depth_image[int(min(y1,y2)+height//2)][int(min(x1,x2)+width//2)]), True, (255, 0, 0))
                text_position = (min(x1,x2), min(y1,y2)-30)
                square_position = (min(x1,x2), min(y1,y2))
                pygame.draw.rect(display, (0, 255, 0), (square_position[0], square_position[1], width, height), 3)
                display.blit(text_surface, text_position)


    def game_loop(self):
        """
        Main 프로그램 루프.
        """
        
        try:
            pygame.init()
            pygame.font.init()

            # 클라이언트와 맵 설정
            self.client = carla.Client("localhost", 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()
            self.tm = self.client.get_trafficmanager()
            self.tm.set_synchronous_mode(True)
            self.tm_port = self.tm.get_port()

            # 날씨 설정
            self.world.set_weather(carla.WeatherParameters(
                sun_altitude_angle=-10.0,
                cloudiness=0.0,
                precipitation=0.0,
                precipitation_deposits=0.0,
                wind_intensity=0.0,
                sun_azimuth_angle=0.0,
                fog_density=0.0,
                fog_distance=0.0
            ))
            # 시나리오 실행을 위해 차량 autopilot 루트 설정
            self.spawn_points = self.world.get_map().get_spawn_points()
            self.route_point = [52, 104, 115, 67, 140, 10, 71, 60, 143, 149, 92, 21, 105, 45, 47, 134, 50]
            self.route = []
            for ind in self.route_point:
                self.route.append(self.spawn_points[ind].location)

            # 보행자 spawn 지점 설정
            self.crosswalks = self.world.get_map().get_crosswalks()
            for each in self.crosswalks :
                each.z += 1
            
            # 차량, 보행자, 카메라 spawn
            self.setup_car()
            self.setup_camera()
            self.setup_depth_camera()
            self.setup_walker()

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
                   
            while True:
                self.world.tick()

                pygame_clock.tick_busy_loop(20)

                # 영상처리
                self.render(self.display)
                pygame.display.flip()

                # 보행자 시나리오 실행
                self.walker_scenario()

                # 키보드 event 감지
                pygame.event.pump()               
                if self.control(self.car) :
                    return
    

        # 모든 object 해제 및 종료
        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.depth_camera.destroy()
            self.car.destroy()
            self.walker.destroy()

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