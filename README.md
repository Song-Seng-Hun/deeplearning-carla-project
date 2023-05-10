# CARLA-project  
![image](https://github.com/DL-project-team/CARLA-project/assets/69943723/b8c8f2d8-58ca-4bb9-90c9-a4148d1987fb)


[image.webm](https://github.com/DL-project-team/CARLA-project/assets/69943723/e06b6d12-480a-4276-a15e-a1ee2b3e7ad3)
## 조원 소개
- [성용호](https://github.com/DL-project-team/CARLA-project/tree/test_syh)
- [송승훈](https://github.com/DL-project-team/CARLA-project/tree/test_ssh)
- [신대준](https://github.com/DL-project-team/CARLA-project/tree/test_sdj)
- [이윤종](https://github.com/DL-project-team/CARLA-project/tree/test_lyj)
- [임용재]()
- [허동욱](https://github.com/DL-project-team/CARLA-project/tree/test-hdw)
![image](https://user-images.githubusercontent.com/69943723/236127279-186a3469-cff3-4ad4-8009-26bd7ee2f84b.png)

## 주제 선정 배경
![image](https://user-images.githubusercontent.com/69943723/236127436-4b56bd91-e93a-4fc0-ab4e-8469725d4482.png)

## 개발 환경 구축
![image](https://user-images.githubusercontent.com/69943723/236130224-4f7c69af-5f77-4750-8753-2e0a9b9a111b.png)
![image](https://user-images.githubusercontent.com/69943723/236135005-685ab991-d3ac-40cf-afc2-efff5c7fad95.png)
- 주의사항 : docker container의 bash에서 pip install ultralytics를 실행해야 정상 실행 가능 (Dockerfile에 포함 시 정상작동 안함)

## 프로젝트 설명 (carla)
![image](https://user-images.githubusercontent.com/69943723/236137777-ac37b2d0-b246-405d-9346-14fd7a4ae8a4.png)
![image](https://user-images.githubusercontent.com/69943723/236135729-c68f06ab-e4df-469b-bfb7-6ecf1feb61c7.png)
![image](https://user-images.githubusercontent.com/69943723/236135755-2d857e02-c0ad-43f1-a888-8f67509b65e3.png)

## 프로젝트 설명 (deeplearning)
![image](https://user-images.githubusercontent.com/69943723/236137325-191641f8-f7a7-4726-a288-1746696ffc07.png)
![image](https://user-images.githubusercontent.com/69943723/236137346-4f88a7be-e307-47c4-8f67-8469978420b3.png)
![image](https://user-images.githubusercontent.com/69943723/236137375-3fafc5e3-1daa-408b-9198-894095617aa9.png)
![yolo](https://user-images.githubusercontent.com/69943723/236143303-1449d493-7794-4347-87b2-d886f376bdeb.gif)
![image](https://user-images.githubusercontent.com/69943723/236143561-f3d84f55-1cfe-4f24-84c9-69ebbb87f136.png)
![image](https://user-images.githubusercontent.com/69943723/236143608-1b84e56e-44e5-4543-9234-009253f9aa71.png)

## 시연 영상
-[YouTube 링크](https://youtu.be/5GZHaqp7ENA)

## 그 외 시도
![image](https://user-images.githubusercontent.com/69943723/236137875-1eefc5db-875c-4aad-9d58-284599fdc1a5.png)
![image](https://user-images.githubusercontent.com/69943723/236137885-09d6c4da-42cb-458f-86f7-1ba71637b66d.png)

## 코드 설명
### docker
- Dockerfile
- requirements.txt
### carla
- carla_scenario.py :
### yolo
- yolo_train.ipynb :
- dataprocessing.ipynb : 여러 데이터셋들의 데이터 라벨들의 인덱스을 원하는 라벨의 인덱스로 바꿔서 적합한 비율로 합치는 코드 

## 데이터
### [roboflow](https://universe.roboflow.com/)
- [carla dataset1](https://universe.roboflow.com/carladataset-qxhsv/carla_dataset)
- [carla dataset2](https://universe.roboflow.com/alec-hantson-student-howest-be/carla-izloa)
- [횡단보도 dataset](https://universe.roboflow.com/john-gouej/crosswalk-v2/browse)
- [신호등 보강 dataset (제작)](https://universe.roboflow.com/amrws/dl_project_merge)
- [횡단보도 위 사람 보강 dataset (제작)](https://universe.roboflow.com/amrws/pedesatrian-crosswalks)
carla_dataset1, carla_dataset2의 데이터를 기반으로 횡단보도, 신호등, 보행자 dataset을 보강해서 사용
### 딥러닝 모델
#### [ultralytics yolov8](https://docs.ultralytics.com/)
- [학습 완료된 pt 파일들](https://drive.google.com/drive/folders/1WKXGFqITRNHN69JPrxvpcNPgI0d-Xtc6?usp=sharing)
- 기존에 학습시켰던 모든 모델.
- best_ver8.pt가 제일 최신이다.
### 기타
- [발표 PPT 파일](https://docs.google.com/presentation/d/1ndLPQ3ZGaDhEzL4bOqJjLXx3oib0LuxKsohafno7ztk/edit?usp=sharing)
## 참조
- [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [docker](https://docs.docker.com/engine/install/ubuntu/)
- [carla github](https://github.com/carla-simulator/carla)
- [ultralytics docs](https://docs.ultralytics.com/)
