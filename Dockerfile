from ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-rmw-zenoh-cpp \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    libjpeg-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy \
    opencv-python \
    pytest

# Setup workspace
WORKDIR /app
COPY . /app/

# Source ROS setup
ENV RMW_IMPLEMENTATION=rmw_zenoh_cpp
ENV PYTHONPATH=/app:$PYTHONPATH

RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

CMD ["bash"]
