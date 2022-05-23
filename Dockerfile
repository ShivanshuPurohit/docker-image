FROM public.ecr.aws/w6p6i9i7/aws-efa-nccl-rdma:base-cudnn8-cuda11-ubuntu20.04

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
    nano

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

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

###################################################
## Install NCCL-tests
RUN git clone https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && cd /opt/nccl-tests \
    && git checkout ${NCCL_TESTS_VERSION} \
    && make MPI=1 \
       MPI_HOME=/opt/amazon/openmpi/ \
       CUDA_HOME=/usr/local/cuda \
       NCCL_HOME=/opt/nccl/build \
       NVCC_GENCODE="-gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_60,code=sm_60"
ENV NCCL_PROTO simple
RUN rm -rf /var/lib/apt/lists/*

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


RUN pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install gpustat
## Install APEX
## we use the latest git clone and edit the setup.py, to disable the check around line 102
#RUN git clone https://github.com/NVIDIA/apex.git $HOME/apex \
#    && cd $HOME/apex/ \
#    && pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN git clone https://github.com/EleutherAI/gpt-neox.git $HOME/gpt-neox \
    && cd $HOME/gpt-neox/ \
    && pip install -r requirements/requirements.txt && pip3 install -r requirements/requirements-onebitadam.txt && pip3 install -r requirements/requirements-sparseattention.txt && pip cache purge

# mchorse
USER mchorse
WORKDIR /home/mchorse

# For intrapod ssh
EXPOSE 22
