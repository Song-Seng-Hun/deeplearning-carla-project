import carla
import time
import pygame
import numpy as np
import argparse
import logging
import random

from ultralytics import YOLO
model = YOLO("/home/carla/PythonAPI/CARLA-project/test_scenario/best.pt")

VIEW_WIDTH = 1280
VIEW_HEIGHT = 720
VIEW_FOV = 120


def handle_image(disp, image):
    
    array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    array = np.reshape(array, (image.height, image.width, 4))

    yolo_array = array[:, :, :3]

    array = yolo_array[:,:,::-1]

    array = array.swapaxes(0,1)
    surface = pygame.surfarray.make_surface(array)
    disp.blit(surface, (0,0))

    result = model(yolo_array)[0]
    
    font = pygame.font.SysFont(None, 48)

    boxes = result.boxes.boxes
    for box in boxes :
        x1, y1, x2, y2,c,idx = box
        width, height = abs(x2-x1), abs(y2-y1)
        text_surface = font.render(result.names[int(idx)], True, (255, 0, 0))
        text_position = (min(x1,x2), min(y1,y2)-30)
        square_position = (min(x1,x2), min(y1,y2))
        pygame.draw.rect(disp, (0, 255, 0), (square_position[0], square_position[1], width, height), 3)
        disp.blit(text_surface, text_position)
        
    
    pygame.display.flip()



def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    argparser.add_argument(
        '--map',
        metavar='TOWN',
        default=None,
        help='start a new episode at the given TOWN')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    pygame.init()
    pygame.font.init()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)

        if args.map is None:
            world = client.get_world()
        else:
            world = client.load_world(args.map)


    except RuntimeError as ex:
        logging.error(ex)
        pygame.quit()

    actor_list = []

    world.set_weather(carla.WeatherParameters.ClearNoon)

    #===========================================================================================
    # pedestrian code

    world.set_pedestrians_seed(1235)
    ped_bp = random.choice(world.get_blueprint_library().filter("walker.pedestrian.*"))

    ped_start_location = carla.Location(x=-34.3, y=2.7, z=0.2)
    ped_start_rotation = carla.Rotation(pitch=-0.3, yaw=180.0, roll=0.0)
    start_trans = carla.Transform(ped_start_location, ped_start_rotation)

    ped_end_location = carla.Location(x=-60.3, y=2.7, z=0.2)
    ped_end_rotation = carla.Rotation(pitch=-0.3, yaw=-180.0, roll=0.0)
    dst_trans = carla.Transform(ped_end_location, ped_end_rotation)

    ped = world.spawn_actor(ped_bp, start_trans)
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    controller = world.spawn_actor(walker_controller_bp, carla.Transform(), ped)
    controller.start()
    controller.set_max_speed(2)

    # adding an actor to an actor list
    actor_list.append(ped)
    actor_list.append(controller)
    #===========================================================================================

    #시뮬레이션 내에서 사용 가능한 모든 차량 블루프린트를 가져오는 데 사용됩니다
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]

    transform = carla.Transform()

    transform.location.x = 0
    transform.location.y = 13
    transform.location.z = 0.5

    transform.rotation.yaw = 180
    transform.rotation.pitch = 0
    transform.rotation.roll = 0

    vehicle = world.spawn_actor(vehicle_bp, transform)

    #서버 카메라
    spectator = world.get_spectator()
    sp_transform = carla.Transform(transform.location + carla.Location(z=30, x=-25),
        carla.Rotation(yaw=90, pitch=-90))
    spectator.set_transform(sp_transform)

    control = carla.VehicleControl()
    control.throttle = 0.3
    vehicle.apply_control(control)

    # # 차량 위치 출력
    # vehicle_location = vehicle.get_location()
    # print( vehicle_location)

    # # 차량 x좌표 출력
    # vehicle_x = vehicle_location.x
    # print( vehicle_x)

    # vehicle_y = vehicle_location.y
    # print( vehicle_y)


    # 차량 카메라 위치
    rgb_camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    rgb_camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
    rgb_camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
    rgb_camera_bp.set_attribute('fov', str(VIEW_FOV))
    cam_transform = carla.Transform(carla.Location(x=0.3, z=1.7))
    camera = world.spawn_actor(rgb_camera_bp, 
        cam_transform,
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid)
    

    
    display = pygame.display.set_mode(
        (1280, 720),
        pygame.HWSURFACE | pygame.DOUBLEBUF
    )

    camera.listen(lambda image: handle_image(display, image))

    while True :

        x = ped.get_location().x
        y = ped.get_location().y        # restrict Unexpected movement
        
        if x < -62.0 or y < 0.5:
            controller.go_to_location(ped_start_location)
            # target_location = ped_start_location
        elif x > -32.0 or y < 0.5:
            controller.go_to_location(ped_end_location)

        if vehicle.get_location().x < -29.1 :
            # 차량 우회전
            control = carla.VehicleControl()
            control.throttle = 0.3  # 가속도 설정
            control.steer = 0.22  # 우회전을 위한 조향각 설정
            vehicle.apply_control(control)
            #print('start turn degree : ',vehicle.get_transform().rotation.yaw)
        if abs(vehicle.get_transform().rotation.yaw) <= 93.0 :
            control = carla.VehicleControl()
            control.throttle = 0.3  # 가속도 설정
            control.steer = 0  # 우회전을 위한 조향각 설정
            vehicle.apply_control(control)
            #print('stop turn degree : ',vehicle.get_transform().rotation.yaw)
            break

        world.tick()
            
    time.sleep(4)

    camera.destroy()
    vehicle.destroy()
    ped.destroy()
    pygame.quit()
    pygame.font.quit()


if __name__ == '__main__':

    main()
