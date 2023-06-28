# Use an official Python runtime as the base image
FROM nvidia/cuda:11.7.0-devel-ubuntu20.04 AS builder
ARG TZ=Asia/Taipei

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe
RUN apt-get -y update
RUN apt-get -y install python3.6
RUN apt-get -y install python3-pip libsm6 libxext6 libxrender-dev libegl1-mesa-dev libgl-dev wget unzip ffmpeg

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container, install the dependencies
COPY requirements.txt .
RUN pip install -U pip wheel cmake
RUN wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
RUN unzip ninja-linux.zip -d /usr/local/bin/
RUN update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask app and uWSGI configuration into the container
COPY . .

# Copy your custom Nginx configuration file
# COPY nginx.conf /etc/nginx/conf.d

# Expose the port that Nginx will listen on
EXPOSE 80

# Start uWSGI with the Flask app
CMD ["/bin/sh", "start_script.sh"]
