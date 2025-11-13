FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

ARG DEBIAN_FRONTEND=noninteractive
# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    sox \
    libsox-dev \
    libsox-fmt-all \
    build-essential \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    ffmpeg \
    libsm6 \
    libxext6 \
    unzip \
    tmux \
    vim \
    fish \
    nvidia-cuda-toolkit \
    rclone \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

ARG DOCKER_NAME
ARG DOCKER_USER_ID
ARG DOCKER_GROUP_ID

USER root
RUN groupadd -g $DOCKER_GROUP_ID $DOCKER_NAME

RUN adduser --disabled-password --uid $DOCKER_USER_ID --gid $DOCKER_GROUP_ID --gecos '' --shell /bin/bash $DOCKER_NAME \
 && chown -R $DOCKER_NAME:$DOCKER_NAME /app
RUN echo "$DOCKER_NAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-$DOCKER_NAME
USER $DOCKER_NAME

ENV HOME=/home/$DOCKER_NAME

RUN mkdir -p $HOME/.cache $HOME/.config $HOME/tmp $HOME/opt/conda\
 && chmod -R 777 $HOME

# Set up the Conda environment
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh -O $HOME/opt/conda/miniconda.sh && chmod +x $HOME/opt/conda/miniconda.sh \
 && $HOME/opt/conda/miniconda.sh -b -p $HOME/opt/conda/miniconda 
# Install your environment.yaml deps into base env  
COPY --chown=$DOCKER_NAME:$DOCKER_NAME environment.yml $HOME/tmp
RUN . $HOME/opt/conda/miniconda/bin/activate && conda env update --name base --file $HOME/tmp/environment.yml
ENV PATH $HOME/opt/conda/miniconda/bin:$PATH

RUN . $HOME/opt/conda/miniconda/bin/activate && sudo apt remove cmake -y && pip install --upgrade cmake && sudo apt-get install gcc-8 \
 && sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 1 && sudo update-alternatives --config gcc
RUN echo "source activate base" > ~/.bashrc
RUN pip install comet_ml
