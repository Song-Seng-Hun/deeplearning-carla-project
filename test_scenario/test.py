import carla
import time
import pygame
import numpy as np
import argparse
import logging


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

    world.set_weather(carla.WeatherParameters.ClearNoon)

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]

    transform = carla.Transform()

    transform.location.x = -110
    transform.location.y = 60
    transform.location.z = 20

    transform.rotation.yaw = 180
    transform.rotation.pitch = 0
    transform.rotation.roll = 0

    vehicle = world.spawn_actor(vehicle_bp, transform)

    spectator = world.get_spectator()
    sp_transform = carla.Transform(transform.location + carla.Location(z=30, x=-25),
        carla.Rotation(yaw=90, pitch=-90))
    spectator.set_transform(sp_transform)

    control = carla.VehicleControl()
    control.throttle = 0.3
    vehicle.apply_control(control)

    rgb_camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
    camera = world.spawn_actor(rgb_camera_bp, 
        cam_transform,
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid)
    
    display = pygame.display.set_mode(
        (1200, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF
    )

    camera.listen(lambda image: handle_image(display, image))

    time.sleep(15)

    camera.destroy()
    vehicle.destroy()


if __name__ == '__main__':

    main()
