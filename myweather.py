import carla
import time
import numpy as np
import pygame
import logging
import random

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

client = carla.Client('192.168.0.4', 2000)
client.set_timeout(20.0)
print('서버에 접속했습니다.')

world = client.get_world()
init_weather = world.get_weather()
weather = init_weather

try :
    while(True) :
        weather.cloudiness = random.uniform(0, 100) # 구름
        weather.precipitation = random.uniform(0, 100) # 비
        weather.precipitation_deposits = random.uniform(0, 100) # 물웅덩이
        weather.wind_intensity = random.uniform(0, 100) # 바람
        weather.sun_azimuth_angle = random.uniform(0, 360) # 태양
        weather.sun_altitude_angle = random.uniform(-90, 90) # 태양
        weather.fog_density = random.uniform(0, 100) # 안개
        weather.fog_distance = random.uniform(0, 30) # 안개
        weather.fog_falloff = random.uniform(0, 30) # 안개
        weather.scattering_intensity = random.uniform(0, 30) # 안개
        weather.mie_scattering_scale = random.uniform(0, 30) # 안개
        weather.rayleigh_scattering_scale = random.uniform(0, 30) # 안개
        weather.dust_storm = random.uniform(0, 100) # 모래먼지
        # weather.wetness = random.uniform(0, 100)
        
        print('날씨를 변경합니다')
        world.set_weather(weather)
        time.sleep(5)
finally :
    print('종료합니다.')
    world.set_weather(init_weather)
    pygame.quit()

