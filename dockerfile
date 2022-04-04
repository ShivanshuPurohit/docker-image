FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG EFA_INSTALLER_VERSION=latest
ARG AWS_OFI_NCCL_VERSION=aws
ARG NCCL_TESTS_VERSION=master

RUN apt-get install -y --no-install-recommends ca-certificates && \
rm -rf /var/lib/apt/lists/* \
&& update-ca-certificates

RUN apt-get update -y
RUN apt-get remove -y --allow-change-held-packages \
                      libmlx5-1 ibverbs-utils libibverbs-dev libibverbs1 \
                      libnccl2 libnccl-dev

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    git \
    gcc \
    vim \
    kmod \
    sudo \
    openssh-client \
    openssh-server \
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

RUN mkdir -p /var/run/sshd
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH
ENV PATH /opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH

RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python /tmp/get-pip.py \
    && pip install awscli pynvml

#################################################
## Install EFA installer
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf $HOME/aws-efa-installer

###################################################
## Install NCCL
RUN git clone https://github.com/NVIDIA/nccl /opt/nccl \
    && cd /opt/nccl \
    && git checkout v2.11.4-1 \
    && make -j src.build CUDA_HOME=/usr/local/cuda \
    NVCC_GENCODE="-gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_60,code=sm_60"

###################################################
## Install AWS-OFI-NCCL plugin
RUN git clone https://github.com/aws/aws-ofi-nccl.git /opt/aws-ofi-nccl \
    && cd /opt/aws-ofi-nccl \
    && git checkout ${AWS_OFI_NCCL_VERSION} \
    && ./autogen.sh \
    && ./configure --prefix=/opt/aws-ofi-nccl/install \
       --with-libfabric=/opt/amazon/efa/ \
       --with-cuda=/usr/local/cuda \
       --with-nccl=/opt/nccl/build \
       --with-mpi=/opt/amazon/openmpi/ \
    && make && make install

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
RUN useradd --create-home --uid 1000 --shell /bin/bash mchorse && \
    usermod -aG sudo mchorse && \
    echo "mchorse ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

### SSH
# Set password
RUN echo 'password' >> password.txt && \
    echo "root:`cat password.txt`" | chpasswd && \
    # Allow root login with password
    sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    # Prevent user being kicked off after login
    sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd && \
    echo 'AuthorizedKeysFile     .ssh/authorized_keys' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config && \
    # FIX SUDO BUG: https://github.com/sudo-project/sudo/issues/42
    echo "Set disable_coredump false" >> /etc/sudo.conf && \
    # Clean up
    rm password.txt

## SSH config and bashrc
RUN mkdir -p /home/mchorse/.ssh /job && \
    echo 'Host *' > /home/mchorse/.ssh/config && \
    echo '    StrictHostKeyChecking no' >> /home/mchorse/.ssh/config && \
    echo 'export PDSH_RCMD_TYPE=ssh' >> /home/mchorse/.bashrc && \
    echo 'export PATH=/home/mchorse/.local/bin:$PATH' >> /home/mchorse/.bashrc && \
    echo 'export PATH=/usr/local/mpi/bin:$PATH' >> /home/mchorse/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH' >> /home/mchorse/.bashrc


# Generate keys for intra-pod ssh coms
RUN ssh-keygen -b 2048 -t rsa -f /root/.ssh/id_rsa -q -N ""
RUN cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys
RUN touch /root/.ssh/config
RUN chmod 700 /root/.ssh/config

RUN pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
## Install APEX
## we use the latest git clone and edit the setup.py, to disable the check around line 102
#RUN git clone https://github.com/NVIDIA/apex.git $HOME/apex \
#    && cd $HOME/apex/ \
#    && pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN git clone https://github.com/EleutherAI/gpt-neox.git $HOME/gpt-neox \
    && cd $HOME/gpt-neox/ \
    && pip install -r requirements/requirements.txt && pip3 install -r requirements/requirements-onebitadam.txt && pip3 install -r requirements/requirements-sparseattention.txt && pip cache purge \
    && python megatron/fused_kernels/setup.py install


# mchorse
USER mchorse
WORKDIR /home/mchorse

# For intrapod ssh
EXPOSE 22
