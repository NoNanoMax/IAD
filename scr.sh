#!/usr/bin/env bash

# Set these to get your notebooks running!
IMAGE_NAME="bebop-gpu:0.3.0"
# COMMON_STORAGE=/mnt/meteo-storage/common
DOCKER_JUPYTER_PORT=9080
DOCKER_TENSORBOARD_PORT=9081
WORKINGDIR=/mnt/meteo-storage/maximdanilov

if [ -z ${DOCKER_JUPYTER_PORT+UNSET} ]; then
	echo "Please set DOCKER_JUPYTER_PORT"
	echo $DOCKER_JUPYTER_PORT
	exit 1;
else
	echo "Will use port $DOCKER_JUPYTER_PORT for jupyter notebooks served from docker";
fi

if [ -z ${DOCKER_TENSORBOARD_PORT+UNSET} ]; then
	echo "Please set DOCKER_TENSORBOARD_PORT"
	echo $DOCKER_TENSORBOARD_PORT
	exit 1;
else
	echo "Will use port $DOCKER_TENSORBOARD_PORT for tensorboard served from docker";
fi

if [ -z ${WORKINGDIR+UNSET} ]; then
	echo "Please set WORKINGDIR variable"
	exit 1;
else
	echo "Will use $WORKINGDIR as docker /mnt";
fi

if [ ! -d "$WORKINGDIR/notebooks" ]; then
    echo "Creating notebooks in $WORKINGDIR/notebooks"
    mkdir -p "$WORKINGDIR/notebooks"
fi

COMMON_STORAGE=/mnt/meteo-storage/common
nvidia-docker run -d -t -i \
    --net=bridge \
	-e LOCAL_USER_ID=`id -u $USER` \
	-p $DOCKER_JUPYTER_PORT:8888 \
	-p $DOCKER_TENSORBOARD_PORT:6006 \
	-v $WORKINGDIR:/mnt \
	-v $COMMON_STORAGE:/mnt/common-storage \
	$IMAGE_NAME
