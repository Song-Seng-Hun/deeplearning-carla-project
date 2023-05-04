import carla
import pygame
import cv2
import math
import time
import open3d as o3d
from matplotlib import cm
import numpy as np
import random

# 차량 생성위치 도로 ID(시뮬레이터 실행할때마다 ID는 바뀜)
# 상 0 1 하 51 52 좌 79 137 우 99 102

client = carla.Client("localhost",2000)
world = client.get_world()
spawn_points = world.get_map().get_spawn_points()

# Some parameters for text on screen
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,50)
fontScale              = 0.5
fontColor              = (255,255,255)
thickness              = 2
lineType               = 2

car_list = ['vehicle.volkswagen.t2_2021','vehicle.vespa.zx125','vehicle.mercedes.sprinter',
            'vehicle.carlamotors.carlacola','vehicle.tesla.model3','vehicle.mercedes.coupe_2020',
            'vehicle.audi.etron','vehicle.nissan.patrol_2021']
spawn_point_list = [0,1,51,52,79,137,99,102]
car_on_location_list = []
npc_car_list = []

def pygame_callback(disp, image):
    org_array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    array = np.reshape(org_array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:,:,::-1]
    array = array.swapaxes(0,1)
    surface = pygame.surfarray.make_surface(array)
    disp.blit(surface, (0,0))
    pygame.display.flip()

def remove():
    print("!!! destroyed !!!")
    cv2.destroyAllWindows()
    rgb_camera_1.destroy()
    rgb_camera_2.destroy()
    depth_camera.destroy()
    sem_camera.destroy()
    gnss_sensor.destroy()
    imu_sensor.destroy()
    vis.destroy_window()
    lidar.destroy()
    radar.destroy()
    vehicle.destroy()
    walker.destroy()
    # for actor in world.get_actors().filter('*sensor*'):
    #     actor.destroy()

# ============================== 센서 콜백 ============================== #
def rgb_callback(image, data_dict):
    data_dict['rgb_image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
def depth_callback(image, data_dict):
    image.convert(carla.ColorConverter.LogarithmicDepth)
    data_dict['depth_image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
def sem_callback(image, data_dict):
    image.convert(carla.ColorConverter.CityScapesPalette)
    data_dict['sem_image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
def gnss_callback(data, data_dict):
    data_dict['gnss'] = [data.latitude, data.longitude]
def imu_callback(data, data_dict):
    data_dict['imu'] = {'gyro': data.gyroscope,'accel': data.accelerometer,'compass': data.compass}

# Draw the compass data (in radians) as a line with cardinal directions as capitals
def draw_compass(img, theta):
    
    compass_center = (700, 100)
    compass_size = 50
    
    cardinal_directions = [
        ('N', [0,-1]),
        ('E', [1,0]),
        ('S', [0,1]),
        ('W', [-1,0])
    ]
    
    for car_dir in cardinal_directions:
        cv2.putText(rgb_data['rgb_image'], car_dir[0], 
        (int(compass_center[0] + 1.2 * compass_size * car_dir[1][0]), int(compass_center[1] + 1.2 * compass_size * car_dir[1][1])), 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
    
    compass_point = (int(compass_center[0] + compass_size * math.sin(theta)), int(compass_center[1] - compass_size * math.cos(theta)))
    cv2.line(img, compass_center, compass_point, (255, 255, 255), 3)

# LIDAR callback
def lidar_callback(point_cloud, point_list):
    """Prepares a point cloud with intensity
    colors ready to be consumed by Open3D"""
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # Isolate the intensity and compute a color for it
    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    points = data[:, :-1]
    points[:, :1] = -points[:, :1]
    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)
# RADAR callback
def radar_callback(data, point_list):
    radar_data = np.zeros((len(data), 4))
    
    for i, detection in enumerate(data):
        x = detection.depth * math.cos(detection.altitude) * math.cos(detection.azimuth)
        y = detection.depth * math.cos(detection.altitude) * math.sin(detection.azimuth)
        z = detection.depth * math.sin(detection.altitude)
        radar_data[i, :] = [x, y, z, detection.velocity]
        
    intensity = np.abs(radar_data[:, -1])
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, COOL_RANGE, COOL[:, 0]),
        np.interp(intensity_col, COOL_RANGE, COOL[:, 1]),
        np.interp(intensity_col, COOL_RANGE, COOL[:, 2])]
    
    points = radar_data[:, :-1]
    points[:, :1] = -points[:, :1]
    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)

# ============================== 웨이포인트 ============================== #
# # update required
# map = world.get_map()
# waypoint = world.get_map().get_waypoint

for i, spawn_point in enumerate(spawn_points):
    world.debug.draw_string(spawn_point.location, str(i), life_time=60)
bp_lib = world.get_blueprint_library()

# ============================== 차량 ============================== #
my_car_bp = bp_lib.filter('vehicle.tesla.model3')[0]
spawn_0 = carla.Transform(carla.Location(x=-5,y=12.7,z=1),carla.Rotation(pitch=0,yaw=180,roll=0))

for i in range(len(car_list)):
    vehicle_each = None
    vehicle_bp_each = bp_lib.find(car_list[i])
    spawn_each = spawn_points[spawn_point_list[i]]
    # spawn_each = spawn_each + carla.Transform(carla.Location(z=5))
    vehicle_each = world.spawn_actor(vehicle_bp_each, spawn_each)
    car_on_location_list.append(vehicle_each)

# ============================== 관전자 ============================== #
spectator = world.get_spectator()
spectator.set_transform(carla.Transform(carla.Location(x=-5,y=12.7,z=5),carla.Rotation(pitch=0,yaw=180,roll=0)))

# ============================== 카메라 ============================== #
cam_transform = carla.Transform(carla.Location(x=0.5, z=1.7))
rgb_camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
depth_camera_bp = bp_lib.find('sensor.camera.depth') 
sem_camera_bp = bp_lib.find('sensor.camera.semantic_segmentation')                  
image_w = rgb_camera_bp.get_attribute("image_size_x").as_int()
image_h = rgb_camera_bp.get_attribute("image_size_y").as_int()
rgb_data = {'rgb_image': np.zeros((image_h, image_w, 4))}
depth_data = {'depth_image': np.zeros((image_h, image_w, 4))}
sem_data = {'sem_image': np.zeros((image_h, image_w, 4))}

# ============================== GNSS, IMU ============================== #
gnss_bp = bp_lib.find('sensor.other.gnss')                                          # GNSS
imu_bp = bp_lib.find('sensor.other.imu')                                            # IMU 
gnss_data = {'gnss':[0,0]}
imu_data = {'imu':{'gyro': carla.Vector3D(), 'accel': carla.Vector3D(), 'compass': 0}}

# ============================== LiDAR, RADAR ============================== #
# Auxilliary code for colormaps and axes
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
COOL_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
COOL = np.array(cm.get_cmap('winter')(COOL_RANGE))
COOL = COOL[:,:3]

def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)

# Set up LIDAR and RADAR, parameters are to assisst visualisation
lidar_bp = bp_lib.find('sensor.lidar.ray_cast') 
lidar_bp.set_attribute('range', '100.0')
lidar_bp.set_attribute('noise_stddev', '0.1')
lidar_bp.set_attribute('upper_fov', '15.0')
lidar_bp.set_attribute('lower_fov', '-25.0')
lidar_bp.set_attribute('channels', '64.0')
lidar_bp.set_attribute('rotation_frequency', '20.0')
lidar_bp.set_attribute('points_per_second', '500000')
lidar_init_trans = carla.Transform(carla.Location(z=2))
radar_bp = bp_lib.find('sensor.other.radar') 
radar_bp.set_attribute('horizontal_fov', '30.0')
radar_bp.set_attribute('vertical_fov', '30.0')
radar_bp.set_attribute('points_per_second', '10000')
radar_init_trans = carla.Transform(carla.Location(z=2))

# Add auxilliary data structures
point_list = o3d.geometry.PointCloud()
radar_list = o3d.geometry.PointCloud()

# ============================== 보행자 ============================== #
spawn_location_walker = carla.Transform(carla.Location(x=-35, y=2.7, z=3.0),carla.Rotation(pitch=0, yaw=180, roll=0))
walker_bp = bp_lib.filter('walker.pedestrian.*')[3]

# ============================== 신호등 ============================== #
t_lights = world.get_actors().filter("*traffic_light*")
for i in range(len(t_lights)):
    t_lights[i].set_state(carla.TrafficLightState.Off)
    t_light_transform = t_lights[i].get_transform()
    location = t_light_transform.location
    world.debug.draw_string(location, str(t_lights[i].id), draw_shadow=False,
                             color=carla.Color(r=0, g=0, b=255), life_time=60.0,)
    loc = t_lights[i].get_location()
    if loc.x == -64.26419067382812:
        my_t_light = t_lights[i]
my_t_light.set_green_time(30.0)
my_t_light.set_yellow_time(0.5)
my_t_light.set_red_time(0.5)
my_t_light.set_state(carla.TrafficLightState.Green)

# ============================== 파이게임 화면 ============================== #
display = pygame.display.set_mode((800, 600),pygame.HWSURFACE | pygame.DOUBLEBUF)


frame = 0

objectExist = False
generate = False
trafficOn = False
autoPilotEnable = False
cameraOn = False
lidarRadarOn = False
quitGame = False

cv2.waitKey(1)

while True:
    keys = pygame.key.get_pressed()

    if generate == True:
        vehicle = world.spawn_actor(my_car_bp, spawn_0)
        walker = world.spawn_actor(walker_bp, spawn_location_walker)
        rgb_camera_1 = world.spawn_actor(rgb_camera_bp, cam_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
        rgb_camera_2 = world.spawn_actor(rgb_camera_bp, cam_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
        depth_camera = world.spawn_actor(depth_camera_bp, cam_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
        sem_camera = world.spawn_actor(sem_camera_bp, cam_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
        gnss_sensor = world.spawn_actor(gnss_bp, cam_transform, attach_to=vehicle)
        imu_sensor = world.spawn_actor(imu_bp, cam_transform, attach_to=vehicle)
        radar = world.spawn_actor(radar_bp, radar_init_trans, attach_to=vehicle)
        lidar = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=vehicle)

        rgb_camera_1.listen(lambda image: pygame_callback(display, image))           # rgb camera for pygame window
        rgb_camera_2.listen(lambda image: rgb_callback(image, rgb_data))            # rgb camera for cv2 window
        depth_camera.listen(lambda image: depth_callback(image, depth_data))
        sem_camera.listen(lambda image: sem_callback(image, sem_data))

        gnss_sensor.listen(lambda event: gnss_callback(event, gnss_data))
        imu_sensor.listen(lambda event: imu_callback(event, imu_data))
        lidar.listen(lambda data: lidar_callback(data, point_list))
        radar.listen(lambda data: radar_callback(data, radar_list))

        # Open3D visualiser for LIDAR and RADAR
        if lidarRadarOn == False:
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name='LiDAR and RADAR',
                width=960,
                height=540,
                left=480,
                top=270)
            vis.get_render_option().background_color = [0.05, 0.05, 0.05]
            vis.get_render_option().point_size = 1
            vis.get_render_option().show_coordinate_frame = True
            add_open3d_axis(vis)
            lidarRadarOn = True

        print("!!! initialized !!!")
        control = carla.VehicleControl()
        vehicle.apply_control(carla.VehicleControl(throttle=0.2,steer=0))
        generate = False
        objectExist = True
        cameraOn = True

    if my_t_light.get_state() == carla.libcarla.TrafficLightState.Green and objectExist == True:
        vehicle.apply_control(carla.VehicleControl(throttle=0.5,steer=0))

    if my_t_light.get_state() == carla.libcarla.TrafficLightState.Yellow and objectExist == True:
        vehicle.apply_control(carla.VehicleControl(throttle=0.4,steer=0))

    if my_t_light.get_state() == carla.libcarla.TrafficLightState.Red and objectExist == True:
        vehicle.apply_control(carla.VehicleControl(throttle=0.3,steer=0))

    if objectExist == True:                                                         # 직진 종료
        if vehicle.get_location().x < -64:
            print("x:%f " % vehicle.get_location().x +", " + "y:%f " % vehicle.get_location().y)
            remove()
            objectExist = False
            cameraOn = False
            quitGame = True
            break
        else:
            pass
    else:
        pass

    if objectExist == True:                                                         # 우회전 시작
        if vehicle.get_location().x <= -29.1 and \
            my_t_light.get_state() == carla.libcarla.TrafficLightState.Green:
            control.throttle = 0.5
            control.steer = 0.2
            vehicle.apply_control(control)
        else:
            pass
    else:
        pass

    if objectExist == True and abs(vehicle.get_transform().rotation.yaw) <= 93.0:   # 우회전 종료
        vehicle.apply_control(carla.VehicleControl(throttle=0.5,steer=-0.1))

    if objectExist == True:                                                         # 우회전 시나리오 종료, 차 제거
        if vehicle.get_location().y < -10:
            print("x:%f " % vehicle.get_location().x +", " + "y:%f " % vehicle.get_location().y)
            remove()
            objectExist = False
            cameraOn = False
            quitGame = True
            break
        else:
            pass
    else:
        pass

    if trafficOn == True:                                                           # 월드에 차량 랜덤 스폰
        for i in range(10):
            npc_car_bp = random.choice(bp_lib.filter('vehicle'))
            npc_car = world.try_spawn_actor(npc_car_bp, random.choice(spawn_points)) 
        for npc_car in world.get_actors().filter('*vehicle*'):
            npc_car.set_autopilot(True)
        trafficOn = False

    if autoPilotEnable == True:
        for i,car in enumerate(car_on_location_list):
            car.set_autopilot(True)
    elif autoPilotEnable == False:
        for i,car in enumerate(car_on_location_list):
            car.set_autopilot(False)

    for event in pygame.event.get() :
        if event.type == pygame.KEYDOWN:           
            if event.key == pygame.K_ESCAPE:                                        # 오브젝트 정리 및 세션 종료
                if objectExist == True:
                    remove()
                    objectExist = False
                    generate = False
                    quitGame = True
                elif objectExist == False:
                    quitGame = True
                if trafficOn == True:
                    for npc_car in world.get_actors().filter('*vehicle*'): 
                        npc_car.destroy()
                elif trafficOn == False:
                    npc_cars = world.get_actors().filter('*vehicle*')
                    if len(npc_cars)>=1:
                        for npc_car in world.get_actors().filter('*vehicle*'): 
                            npc_car.destroy()
            if event.key == pygame.K_c:                                             # c키는 차량 생성
                generate = True
            if event.key == pygame.K_m:                                             # m키는 지도 단순화
                world.unload_map_layer(carla.MapLayer.All)
            if event.key == pygame.K_n:                                             # n키는 지도 세팅 복구
                world.load_map_layer(carla.MapLayer.All)
            if event.key == pygame.K_g:                                             # g키는 월드에 차량 생성
                trafficOn = True
            if event.key == pygame.K_a:
                if autoPilotEnable == False:
                    autoPilotEnable = True
                    print("car_on_locatiob autopilot enabled")
                elif autoPilotEnable == True:
                    autoPilotEnable = False
                    print("car_on_locatiob autopilot disabled")

    if quitGame == True:
        pygame.quit()
        break

    if cv2.waitKey(1) == ord('q'):
        pass

    # Latitude from GNSS sensor
    cv2.putText(rgb_data['rgb_image'], 'Lat: ' + str(gnss_data['gnss'][0]), 
    (10,30), 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    
    # Longitude from GNSS sensor
    cv2.putText(rgb_data['rgb_image'], 'Long: ' + str(gnss_data['gnss'][1]), 
    (10,50), 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    
    # Calculate acceleration vector minus gravity
    accel = imu_data['imu']['accel'] - carla.Vector3D(x=0,y=0,z=9.81)
    
    # Display acceleration magnitude
    cv2.putText(rgb_data['rgb_image'], 'Accel: ' + str(accel.length()), 
    (10,70), 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    
    # Gyroscope output
    cv2.putText(rgb_data['rgb_image'], 'Gyro: ' + str(imu_data['imu']['gyro'].length()), 
    (10,100), 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    
    # Compass value in radians, North is 0 radians
    cv2.putText(rgb_data['rgb_image'], 'Compass: ' + str(imu_data['imu']['compass']), 
    (10,120), 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    
    # Draw the compass
    draw_compass(rgb_data['rgb_image'], imu_data['imu']['compass'])

    if lidarRadarOn == True:
        if frame == 2:
            vis.add_geometry(point_list)
            vis.add_geometry(radar_list)
        vis.update_geometry(point_list)
        vis.update_geometry(radar_list)
        vis.poll_events()
        vis.update_renderer()
        # # This can fix Open3D jittering issues:
        time.sleep(0.005)
        frame += 1

    if cameraOn == True:
        rds = np.concatenate((rgb_data['rgb_image'], depth_data['depth_image'], sem_data['sem_image']), axis=1)
        cv2.imshow("camera", rds)