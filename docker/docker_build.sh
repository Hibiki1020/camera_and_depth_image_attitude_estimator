#!/bin/bash

image_name='mono_and_depth_image_attitude_estimator'
image_tag='docker'

docker build -t $image_name:$image_tag .