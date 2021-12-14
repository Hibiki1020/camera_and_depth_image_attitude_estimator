FROM vistart/cuda:10.0-cudnn7-tensorrt6-devel-ubuntu18.04

########## nvidia-docker1 hooks ##########
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
########## non-interactive ##########
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \
	vim \
	wget \
	unzip \
	git \
	build-essential

RUN apt-get update && \
	apt-get install -y \
	python3-pip

RUN python3 -m pip install -U --force-reinstall pip

RUN pip install scikit-learn \
		tensorflow \
		tensorboardX==1.6 \
        opencv-python==4.2.0.34 \
        pbkdf2==1.3 \
        easydict==1.9 \
        protobuf==3.12.2 \
        scipy==1.5.0 \
        pbkdf2==1.3

RUN pip install \
	    torch==1.0.0 \
        torchvision==0.2.1

RUN pip install torchsummary==1.5.1 \
		tqdm \
		matplotlib

##### NO cache #####
ARG CACHEBUST=1

RUN mkdir /home/pycode && \
	cd /home/pycode && \
    git clone https://github.com/Hibiki1020/mono_and_depth_image_attitude_estimator.git && \
    cd /home/pycode/mono_and_depth_image_attitude_estimator

WORKDIR /home/pycode/mono_and_depth_image_attitude_estimator