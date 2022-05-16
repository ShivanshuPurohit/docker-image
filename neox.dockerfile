# syntax = docker/dockerfile:1.3.0-labs

################################################################################
#
#  GPT-NeoX 
#
################################################################################

ARG CUDA_VERSION=11.4.2
ARG CUDNN_VERSION=8
ARG LINUX_DISTRO=ubuntu
ARG DISTRO_VERSION=20.04
ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"

ARG BUILD_IMAGE=nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}
ARG TRAIN_IMAGE=nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}


###############################################################################
# Build Image
###############################################################################
FROM ${BUILD_IMAGE} AS build

# Change default settings to allow `apt` cache in Docker image.
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \

    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
    > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,id=apt-cache-build,target=/var/cache/apt \

    --mount=type=cache,id=apt-lib-build,target=/var/lib/apt \
    apt update && \
    apt upgrade -y && \
    apt install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        curl \
        wget \
        git && \
    rm -rf /var/lib/apt/lists/*

LABEL maintainer="eric@hallahans.name"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache && ccache --max-size 0


#######################################
# Python
#######################################
ARG PYTHON_VERSION

# Python won't try to write .pyc or .pyo files on the import of source modules.
ENV PYTHONDONTWRITEBYTECODE=1
# Force stdin, stdout and stderr to be totally unbuffered. Good for logging.
ENV PYTHONUNBUFFERED=1
# Allows UTF-8 characters as outputs in Docker.
ENV PYTHONIOENCODING=UTF-8

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \

    apt install -y \
        python3 \
        python3-dev && \
    rm -f /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    python3 -m pip install --upgrade pip


#######################################
# CMake
#######################################
RUN wget -q https://apt.kitware.com/kitware-archive.sh && \

    . ./kitware-archive.sh && \
    rm kitware-archive.sh && \
    apt update && \
    apt install -y cmake


###########################################################
# PyTorch Dependencies
###########################################################

ARG STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}


#######################################
# MKL
#######################################
RUN wget -q -O - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | apt-key add - && \

    echo "deb https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt update -y && \
    apt install -y --no-install-recommends \ 
        intel-oneapi-common-vars \
        intel-oneapi-mkl \
        intel-oneapi-mkl-devel
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:${LD_LIBRARY_PATH}


#######################################
# MAGMA
#######################################
ARG MAGMA_VERSION=2.6.1
ARG TORCH_CUDA_ARCH_LIST
RUN apt install gfortran -y && \

    cd ${STAGE_DIR} && \
    wget -q -O - http://icl.utk.edu/projectsfiles/magma/downloads/magma-${MAGMA_VERSION}.tar.gz | tar xzf - && \
    cd magma-${MAGMA_VERSION} && \
    sed -e "/^#GPU_TARGET.*/{s/^#//; s/?=.*/= ${TORCH_CUDA_ARCH_LIST}/; s/[0-9]\.[0-9]/sm_&/g; s/\.//g; s/\+PTX//}" \
        -e "/^FORT/d" \
        -e "/^#CUDADIR.*/{s/^#//}" \
        -e "/^#MKLROOT.*/{s/^#//; s/composerxe/oneapi/}" \
        ./make.inc-examples/make.inc.mkl-gcc > ./make.inc && \
    . /opt/intel/oneapi/setvars.sh && \
    make -j"$(nproc)" lib && \
    make install prefix=/usr/local/magma
ENV LD_LIBRARY_PATH=/usr/local/magma/lib:${LD_LIBRARY_PATH}


#######################################
# GDRCopy
#######################################
RUN apt install build-essential devscripts debhelper check libsubunit-dev fakeroot pkg-config dkms -y && \

    cd ${STAGE_DIR} && \
    git clone https://github.com/NVIDIA/gdrcopy.git
    cd gdrcopy && \
    git checkout tags/v${GDRCOPY_VERSION} && \
    cd packages && \
    CUDA=/usr/local/cuda ./build-deb-packages.sh && \
    dpkg -i gdrdrv-dkms_<version>_<arch>.<platform>.deb && \
    dpkg -i libgdrapi_<version>_<arch>.<platform>.deb && \
    dpkg -i gdrcopy-tests_<version>_<arch>.<platform>.deb && \
    dpkg -i gdrcopy_<version>_<arch>.<platform>.deb


#######################################
# UCX
#######################################
ARG UCX_VERSION=1.11.2
RUN apt install pkg-config -y && \

    cd ${STAGE_DIR} && \
    wget -q -O - https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/ucx-${UCX_VERSION}.tar.gz | tar xzf - && \
    cd ucx-${UCX_VERSION} && \
    mkdir build && \
    cd build && \
    ../contrib/configure-release --prefix=/usr/local/ucx --with-cuda=/usr/local/cuda && \
    make -j"$(nproc)" && \
    make install && \
    rm -r ${STAGE_DIR}/ucx-${UCX_VERSION}


#######################################
# Open MPI
#######################################
#ENV OPENMPI_BASEVERSION=4.1
#ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.1
#RUN cd ${STAGE_DIR} && \

    #wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION}.tar.gz | tar xzf - && \
    #cd openmpi-${OPENMPI_VERSION} && \
    #./configure --prefix=/usr/local/mpi/openmpi-${OPENMPI_VERSION} --with-cuda=/usr/local/cuda --with-ucx=/usr/local/ucx && \
    #make -j"$(nproc)" install && \
    #ln -s /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi && \
    # Sanity check:
    #test -f /usr/local/mpi/bin/mpic++ && \
    #cd ${STAGE_DIR} && \
    #rm -r ${STAGE_DIR}/openmpi-${OPENMPI_VERSION}
#ENV PATH=/usr/local/mpi/bin:${PATH} \
#    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}
# Create a wrapper for OpenMPI to allow running as root by default
#RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \

#    echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
#    echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
#    chmod a+x /usr/local/mpi/bin/mpirun

#################################################
## Install EFA installer
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH
ENV PATH /opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH

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

#######################################
# Python Packages
#######################################
RUN --mount=type=cache,id=pip-build,target=/root/.cache/pip \

    python3 -m pip install \
        astunparse \
        numpy \
        ninja \
        pyyaml \
        setuptools \
        cffi \
        typing_extensions \
        future \
        six \
        requests \
        pillow \
        pkgconfig

WORKDIR /opt

# Using --jobs 0 gives a reasonable default value for parallel recursion.
RUN git clone --recursive --jobs 0 https://github.com/pytorch/pytorch.git
#RUN git clone --recursive --jobs 0 https://github.com/pytorch/vision.git
#RUN git clone --recursive --jobs 0 https://github.com/pytorch/text.git
#RUN git clone --recursive --jobs 0 https://github.com/pytorch/audio.git


FROM build AS build-torch

ARG PYTORCH_VERSION_TAG=tags/v1.10.0

ARG TORCH_CUDA_ARCH_LIST
ARG TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Checkout to specific version and update submodules.
WORKDIR /opt/pytorch
RUN if [ -n ${PYTORCH_VERSION_TAG} ]; then \

    git fetch --all --tags --prune && \
    git checkout ${PYTORCH_VERSION_TAG} && \
    git submodule sync && \
    git submodule update --init --recursive --jobs 0; \
    fi

# Build PyTorch. `USE_CUDA` and `USE_CUDNN` are made explicit just in case.
RUN --mount=type=cache,target=/opt/ccache \

    . /opt/intel/oneapi/setvars.sh && \
    MAX_JOBS=$(nproc) USE_CUDA=1 USE_CUDNN=1 USE_SYSTEM_NCCL=1 \
    TORCH_NVCC_FLAGS="${TORCH_NVCC_FLAGS} -t $(nproc)" \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    python setup.py bdist_wheel -d /tmp/dist

# Install PyTorch for subsidiary libraries.
#RUN --mount=type=cache,target=/opt/ccache \
#    USE_CUDA=1 USE_CUDNN=1 \
#    TORCH_NVCC_FLAGS=${TORCH_NVCC_FLAGS} \
#    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
#    python setup.py install


#FROM build-torch AS build-vision
#
#ARG TORCHVISION_VERSION_TAG
#ARG TORCH_CUDA_ARCH_LIST
## Build TorchVision from source to satisfy PyTorch versioning requirements.
## Setting `FORCE_CUDA=1` creates bizarre errors unless CCs are specified explicitly.
## Fix this issue later if necessary by getting output from `torch.cuda.get_arch_list()`.
## Note that the `FORCE_CUDA` flag may be changed to `USE_CUDA` in later versions.
#WORKDIR /opt/vision
#RUN if [ -n ${TORCHVISION_VERSION_TAG} ]; then \
#    git checkout ${TORCHVISION_VERSION_TAG} && \
#    git submodule sync && \
#    git submodule update --init --recursive --jobs 0; \
#    fi
#
#RUN --mount=type=cache,target=/opt/ccache \
#    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
#    python setup.py bdist_wheel -d /tmp/dist
#
#FROM build-torch AS build-text
#
#ARG TORCHTEXT_VERSION_TAG
#
#WORKDIR /opt/text
#RUN if [ -n ${TORCHTEXT_VERSION_TAG} ]; then \
#    git checkout ${TORCHTEXT_VERSION_TAG} && \
#    git submodule sync && \
#    git submodule update --init --recursive --jobs 0; \
#    fi
#
## TorchText does not use CUDA.
#RUN --mount=type=cache,target=/opt/ccache \
#    python setup.py bdist_wheel -d /tmp/dist
#
#
#FROM build-torch AS build-audio
#
#ARG TORCHAUDIO_VERSION_TAG
#ARG TORCH_CUDA_ARCH_LIST
#
#WORKDIR /opt/audio
#RUN if [ -n ${TORCHAUDIO_VERSION_TAG} ]; then \
#    git checkout ${TORCHAUDIO_VERSION_TAG} && \
#    git submodule sync && \
#    git submodule update --init --recursive --jobs 0; \
#    fi
#
#RUN --mount=type=cache,target=/opt/ccache \
#    BUILD_SOX=1 USE_CUDA=1 \
#    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
#    python setup.py bdist_wheel -d /tmp/dist


###############################################################################
# Train/Development Image
###############################################################################
FROM ${TRAIN_IMAGE} AS train

LABEL maintainer="eric@hallahans.name"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=UTF-8

ARG PYTHON_VERSION
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

# Change default settings to allow `apt` cache in Docker image.
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \

    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' \
    > /etc/apt/apt.conf.d/keep-cache

ENV TZ=Etc/UTC
ARG DEBIAN_FRONTEND=noninteractive

#######################################
# System Packages
#######################################
RUN apt update && apt install -y wget software-properties-common && \

    wget -q https://apt.kitware.com/kitware-archive.sh && \
    . ./kitware-archive.sh && \
    rm kitware-archive.sh && \
    wget -q -O - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | apt-key add - && \

    echo "deb https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    add-apt-repository ppa:git-core/ppa -y && \

    apt update && \
    apt upgrade -y && \
    apt install -y --no-install-recommends \
        # Version Control
        git \
        # Development
        build-essential \
        gcc \
        g++ \
        autotools-dev \
        cmake \
        # Administration
        sudo \
        tmux \
        iotop \
        software-properties-common \
        # Text Editors
        vim \
        nano \
        # Web
        wget \
        curl \
        # Networking
        ssh \
        iputils-ping \
        net-tools \
        rsync \
        pdsh \
        iftop \
        net-tools \
            # InfiniBand
            ibverbs-providers \
            ibverbs-utils \
            ibutils \
            infiniband-diags \
            rdma-core \
            rdmacm-utils \
            perftest \
        # Python
        python3 \
        python3-dev \
        libpython3-dev \
        # Libraries
        tzdata \
        intel-oneapi-runtime-mkl \
        # Development Libraries
        libcupti-dev \
        # Miscellaneous
        ca-certificates \
        # Utilities
        htop \
        unzip \
        zstd && \
    rm -f /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    python3 -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

#######################################
# MAGMA & PyTorch
#######################################
COPY --from=build /usr/local/magma /usr/local/magma
COPY --from=build-torch /tmp/dist /tmp/dist
COPY --from=build /user/local/openmpi* /user/local/mpi

#######################################
# Python Packages
#######################################
RUN --mount=type=cache,id=pip-build,target=/root/.cache/pip \

    # PyTorch
    python3 -m pip install \
        /tmp/dist/torch-*.whl && \
    rm -rf /tmp/dist && \
    # Requirements
    DS_BUILD_OPS=1 python3 -m pip install \
        pybind11 \
        six \
        regex \
        numpy \
        git+https://github.com/EleutherAI/DeeperSpeed.git#egg=deepspeed \
        transformers \
        tokenizers \
        lm_dataformat \
        ftfy \
        lm_eval \
        wandb && \
    # APEX
    python3 -m pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \ 
        git+https://github.com/NVIDIA/apex.git

#######################################
# SSH
#######################################
RUN echo 'password' >> password.txt && \

    mkdir /var/run/sshd && \

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
# Expose SSH port
EXPOSE 22

#######################################
# User Account
#######################################
ARG GID
ARG UID=1000
ARG GRP=user
ARG USR=mchorse

RUN ((groupadd -g ${GID} ${GRP} && \

      useradd --shell /bin/bash --create-home -u ${UID} -g ${GRP} ${USR} && \

      echo "${GRP} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers) || \
     (useradd --shell /bin/bash --create-home -u ${UID} ${USR} && \

      echo "${USR} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers)) && \
    usermod -aG sudo ${USR} && \
    mkdir -p /home/${USR}/.ssh /job && \
    echo "Host *" > /home/${USR}/.ssh/config && \
    echo "    StrictHostKeyChecking no" >> /home/${USR}/.ssh/config && \
    echo "export PDSH_RCMD_TYPE=ssh" >> /home/${USR}/.bashrc && \
    echo "export PATH=/home/${USR}/.local/bin:\$PATH" >> /home/${USR}/.bashrc && \
    sed -i "s/#force_color_prompt=yes/force_color_prompt=yes/" /home/${USR}/.bashrc
    #echo "export PATH=/usr/local/mpi/bin:$PATH" >> /home/${USR}/.bashrc && \
    #echo "export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH" >> /home/${USR}/.bashrc

USER ${USR}
WORKDIR /home/${USR}
