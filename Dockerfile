FROM carlasim/carla:0.9.13
USER root
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/usr/lib/i386-linux-gnu/:$LD_LIBRARY_PATH
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && \
    apt-get update && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get install -y sudo wget curl build-essential cmake libvulkan1 mesa-vulkan-drivers xdg-user-dirs git software-properties-common gedit nano
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install --no-install-recommends -y python3.7 python3.7-dev python3.7-distutils && \
    apt-get install -y libtbb2 libtbb-dev libpng-dev libjpeg-dev libtiff-dev libglfw3-dev libglm-dev libglew-dev libgtk-3-dev libboost-all-dev && \
    apt-get install -y fonts-nanum fontconfig && \
    sudo ln -s /usr/lib/nvidia-525 /usr/lib/nvidia-current && \
    mkdir /var/run/sshd
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --force-reinstall 
    # rm get-pip.py
RUN sudo apt-get update && \
    apt-get upgrade -y && \
    sudo usermod -aG sudo carla && \
    sudo usermod -aG video carla && \
    sudo usermod -aG audio carla && \
    echo 'carla ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
USER carla
WORKDIR /home/carla/PythonAPI
COPY . /home/carla/PythonAPI
RUN pip3 install --upgrade setuptools && \
    pip3 install --upgrade pip && \
    pip3 install networkx future numpy distro Shapely thop pygame matplotlib seaborn open3d pillow psutil carla==0.9.13 easydict libopencv opencv-python tensorflow torch torchvision ultralytics
# server
CMD echo 'export DISPLAY=$DISPLAY' >> ~/.bashrc && /home/carla/CarlaUE4.sh -quality-level=Low
# client
# CMD tail -f /dev/null && sleep infinity
