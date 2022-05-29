FROM public.ecr.aws/w6p6i9i7/aws-efa-nccl-rdma:base-cudnn8-cuda11.3-ubuntu20.04

ARG EFA_INSTALLER_VERSION=latest
ARG AWS_OFI_NCCL_VERSION=aws
ARG NCCL_TESTS_VERSION=master
ENV DEBIAN_FRONTEND=noninteractive
#execute

RUN apt-get install -y --no-install-recommends ca-certificates && \
rm -rf /var/lib/apt/lists/* \
&& update-ca-certificates

RUN apt-get update -y

#Install core packages
RUN apt-get install -y --allow-unauthenticated \
    git \
    gcc \
    vim \
    kmod \
    sudo \
    ssh \
    apt-utils \
    libncurses5 \
    bash \
    ca-certificates \
    build-essential \
    curl \
    autoconf \
    libtool \
    gdb \
    automake \
    python3-distutils \
    cmake \
    apt-utils \
    python3-dev \
    pdsh \
    nano \
    software-properties-common
    
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y

RUN apt install python3.9 python3.9-dev python3.9-venv
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN touch /var/run/sshd && \
    # Prevent user being kicked off after login
    sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd && \
    echo 'AuthorizedKeysFile     .ssh/authorized_keys' >> /etc/ssh/sshd_config && \
    # FIX SUDO BUG: https://github.com/sudo-project/sudo/issues/42
    sudo echo "Set disable_coredump false" >> /etc/sudo.conf

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH
ENV PATH /opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH

RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python /tmp/get-pip.py \
    && pip install awscli pynvml


#### User account
ARG USERNAME=mchorse
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Creating the user and usergroup
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USERNAME -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN chmod g+rw /home && \
    mkdir -p /home/mchorse && \
    mkdir -p /home/mchorse/.ssh && \
    chown -R $USERNAME:$USERNAME /home/mchorse && \
    chown -R $USERNAME:$USERNAME /home/mchorse/.ssh

USER $USERNAME

### SSH
# Create keys
RUN sudo chmod 700 /home/mchorse/.ssh
RUN ssh-keygen -t rsa -N "" -f /home/mchorse/.ssh/id_rsa && sudo chmod 600 /home/mchorse/.ssh/id_rsa && sudo chmod 600 /home/mchorse/.ssh/id_rsa.pub
RUN cp /home/mchorse/.ssh/id_rsa.pub /home/mchorse/.ssh/authorized_keys
RUN eval `ssh-agent -s` && ssh-add /home/mchorse/.ssh/id_rsa

USER root

## SSH config and bashrc
RUN mkdir -p /home/mchorse/.ssh /job && \
    echo 'Host *' > /home/mchorse/.ssh/config && \
    echo '    StrictHostKeyChecking no' >> /home/mchorse/.ssh/config && \
    echo 'export PDSH_RCMD_TYPE=ssh' >> /home/mchorse/.bashrc && \
    echo 'export PATH=/home/mchorse/.local/bin:$PATH' >> /home/mchorse/.bashrc && \
    echo 'export PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH' >> /home/mchorse/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH' >> /home/mchorse/.bashrc


RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install gpustat protobuf~=3.19.0

## Install APEX
## we use the latest git clone and edit the setup.py, to disable the check around line 102
RUN git clone https://github.com/NVIDIA/apex.git $HOME/apex \
    && cd $HOME/apex/ \
    && pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN git clone https://github.com/EleutherAI/gpt-neox.git $HOME/gpt-neox \
    && cd $HOME/gpt-neox/ \
    && chmod -R 777 $HOME/gpt-neox/ \
    && pip install -r requirements/requirements.txt && pip3 install -r requirements/requirements-onebitadam.txt && pip3 install -r requirements/requirements-sparseattention.txt && pip cache purge
COPY helpers/fused_kernels-0.0.1-cp38-cp38-linux_x86_64.whl $HOME/fused_kernels-0.0.1-cp38-cp38-linux_x86_64.whl
RUN pip install fused_kernels-0.0.1-cp38-cp38-linux_x86_64.whl

# mchorse
USER mchorse
WORKDIR /home/mchorse

# For intrapod ssh
EXPOSE 22

# Starting scripts
COPY helpers/entrypoint.sh ./entrypoint.sh
RUN sudo chmod +x ./entrypoint.sh
COPY helpers/.deepspeed_env ./.deepspeed_env
ENTRYPOINT [ "./entrypoint.sh" ]
USER root
