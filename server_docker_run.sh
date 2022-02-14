#!/bin/bash
image_name="mono_and_depth_image_attitude_estimator"
tag_name="docker"
script_dir=$(cd $(dirname $0); pwd)

docker run -it \
    --net="host" \
    --gpus all \
	--privileged \
    --shm-size=8g \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" \
    --name="mono_and_depth_image_attitude_estimator" \
    --volume="$script_dir/:/home/pycode/$image_name/" \
    --volume="/home/kawai/ssd_dir/:/home/ssd_dir/" \
    --volume="/share/kawai/:/home/strage/" \
    $image_name:$tag_name