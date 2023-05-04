FROM carlasim/carla:0.9.13

USER root
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/usr/lib/i386-linux-gnu/:$LD_LIBRARY_PATH
RUN apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get install -y libvulkan1 mesa-vulkan-drivers && \
    apt-get install -y fonts-nanum && \
    apt-get install -y fontconfig && \
    apt-get install -y sudo && \
    apt-get install -y git && \
    apt-get install -y curl && \
    apt-get upgrade -y 
# Upgrade installed packages
RUN apt update -y && apt upgrade -y

# install python 3.7.10 (or newer)
RUN apt update -y && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.7 python3.7-dev python3.7-distutils && \
    apt clean -y && rm -rf /var/lib/apt/lists/*

# Register the version in alternatives (and set higher priority to 3.7)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

# Upgrade pip to latest version
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --force-reinstall && \
    rm get-pip.py
    
WORKDIR /app
COPY . /app
RUN pip3 install --upgrade setuptools && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

USER carla
SHELL ["/bin/bash", "-c"]
CMD echo 'export DISPLAY=$DISPLAY' >> ~/.bashrc && /home/carla/CarlaUE4.sh -quality-level=Low
# ENTRYPOINT ["/bin/bash"]