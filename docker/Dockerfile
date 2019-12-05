FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.7
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
	 htop \
         git \
         curl \
	 imagemagick \
         emacs-nox \
         ca-certificates \
	 libcfitsio-dev	\
	 libopenslide-dev \
         libjpeg-dev \
	 pkg-config \
	 glib2.0-dev \
	 librsvg2-dev \
	 libexpat1-dev \
	 libtiff5-dev \
	 libjpeg-turbo8-dev \
         libgsf-1-dev \
	 libgl1-mesa-glx \
         rsync \
         gtk-doc-tools \
         autotools-dev \
         automake \
         swig \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

# install libvips
RUN cd /tmp && curl -L -O https://github.com/libvips/libvips/releases/download/v8.7.4/vips-8.7.4.tar.gz
RUN cd /tmp && tar xvf vips*.tar.gz && cd vips-8.7.4 && ./configure && make -j8 && make install

# install base conda
RUN curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Anaconda_37.sh && \
        sh ./Anaconda*.sh -b -p /opt/conda && rm ./Anaconda*.sh
ENV PATH /opt/conda/bin:$PATH
WORKDIR /opt/pytorch

# install conda deps
RUN /opt/conda/bin/conda install numpy pandas scipy scikit-image scikit-learn h5py matplotlib
#RUN /opt/conda/bin/conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
RUN /opt/conda/bin/conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
RUN /opt/conda/bin/conda install -y -c conda-forge opencv

# install misc conda / pip deps
RUN /opt/conda/bin/pip install torchvision tqdm pyvips msgpack joblib
RUN /opt/conda/bin/pip install visdom ray
RUN /opt/conda/bin/conda install -c defaults libprotobuf protobuf

# install apex
RUN git clone --recursive https://github.com/NVIDIA/apex /opt/apex
RUN cd /opt/apex && /opt/conda/bin/python setup.py install

#cleanup
RUN rm -rf /opt/pytorch && rm -rf /opt/apex

# fixup pyvips
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib
RUN /opt/conda/bin/pip uninstall -y pyvips && /opt/conda/bin/pip install --no-cache-dir pyvips

# install pyro-ppl, tensorboard and tensorflow-gpu (for tensorboard)
RUN /opt/conda/bin/pip install pyro-ppl tensorboardX tensorflow-gpu psutil

WORKDIR /workspace
RUN chmod -R a+w /workspace
