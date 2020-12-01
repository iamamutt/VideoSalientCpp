FROM ubuntu:20.04 AS OPENCV_BUILD
ENV DEBIAN_FRONTEND=noninteractive
LABEL maintainer="josephburling@gmail.com"
ENV cwd="/home/"
ARG OPENCV_VERSION="4.5.0"
WORKDIR $cwd

RUN echo "Installing linux opencv/ffmpeg dependencies..."
RUN apt-get -y update -qq --fix-missing && \
    apt-get install -y \
    build-essential \
    cmake \
    ffmpeg \
    git \
    opencv-data \
    pkg-config \
    unzip \
    v4l-utils \
    wget \
    x264 \
    yasm \
    libavcodec-dev  \
    libavformat-dev \
    libavresample-dev \
    libfaac-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    libopencore-amrnb-dev \
    libopencv-dev \
    libpng-dev \
    libpq-dev \
    libswscale-dev \
    libtbb-dev \
    libtbb2 \
    libtheora-dev \
    libtiff-dev \
    libtiff5-dev \
    libv4l-dev \
    libvorbis-dev \
    libx264-dev \
    libxvidcore-dev

RUN cd /usr/include/linux && \
    ln -s -f ../libv4l1-videodev.h videodev.h && \
    cd $cwd

RUN echo "Installing OpenCV..."

# download opencv source
RUN cd $cwd && \
    wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv.zip --progress=bar:force:noscroll --no-check-certificate

# create directories
RUN cd $cwd && \
    unzip -q opencv.zip && \
    mkdir -p opencv-${OPENCV_VERSION}/build

# install opencv with cmake
RUN cd ${cwd}/opencv-${OPENCV_VERSION}/build && \
    cmake \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUILD_DOCS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_java=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_PNG=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_TIFF=ON \
    -DENABLE_PRECOMPILED_HEADERS=OFF \
    -DFORCE_VTK=OFF \
    -DWITH_1394=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_EIGEN=OFF \
    -DWITH_FFMPEG=ON \
    -DWITH_GDAL=OFF \
    -DWITH_GSTREAMER=OFF \
    -DWITH_IPP=ON \
    -DWITH_OPENCL=ON \
    -DWITH_OPENEXR=OFF \
    -DWITH_OPENGL=ON \
    -DWITH_QT=OFF \
    -DWITH_TBB=ON \
    -DWITH_V4L=ON \
    -DWITH_XINE=OFF \
    .. && \
    make -j$(nproc) && \
    make install && \
    cd ..

RUN /bin/sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
RUN ldconfig
RUN rm opencv.zip
RUN rm -f -r opencv-${OPENCV_VERSION}
ENV DEBIAN_FRONTEND teletype

# docker build . -t opencv-build:1.0.0
# docker run --rm -it opencv-build:1.0.0
