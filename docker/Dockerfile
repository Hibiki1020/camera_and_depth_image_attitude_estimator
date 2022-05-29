FROM vistart/cuda:10.1-cudnn7-tensorrt6-devel-ubuntu18.04

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
	build-essential \
	libsm6 \
	libxrender1 \
	libxext-dev

RUN apt-get update && \
	apt-get install -y \
	python3-pip \
	python3-tk

RUN python3 -m pip install -U --force-reinstall pip

RUN pip3 install scikit-learn \
		tensorflow \
		tensorboardX==1.6 \
        opencv-python==4.2.0.34 \
        pbkdf2==1.3 \
        easydict==1.9 \
        protobuf==3.12.2 \
        scipy==1.5.0 \
        pbkdf2==1.3

RUN pip3 install \
	    torch==1.7.1 \
        torchvision==0.8.2 \
        torchaudio==0.7.2

RUN pip3 install torchsummary==1.5.1 \
		tqdm \
		matplotlib \
		pyyaml \
		ttach \
		grad-cam

##### NO cache #####
ARG CACHEBUST=1

RUN mkdir /home/pycode && \
	mkdir /home/log && \
	mkdir -p /home/log/SII2022_Grad_CAM && \
	mkdir -p /home/log/SII2023_Grad_CAM && \
	cd /home/pycode && \
    git clone https://github.com/Hibiki1020/mono_and_depth_image_attitude_estimator.git && \
    cd /home/pycode/mono_and_depth_image_attitude_estimator

#RUN apt-get update && apt-get upgrade -y && apt-get -y install gosu

#COPY entrypoint.sh /usr/local/bin/entrypoint.sh
#RUN chmod +x /usr/local/bin/entrypoint.sh
#ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

WORKDIR /home/pycode/mono_and_depth_image_attitude_estimator