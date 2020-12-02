# Saliency build with OpenCV/FFMPEG dependencies
#
# VERSION               1.0


# build from opencv/ffmpeg image, copying build files and building release
FROM opencv-build:v1.0.0 as saliency_build
LABEL maintainer="josephburling@gmail.com"

ARG saliency_build_dir=/opt/saliency
ENV SALIENCY_DIR_ROOT=$saliency_build_dir

WORKDIR $SALIENCY_DIR_ROOT

COPY CMakeLists.txt main.cpp ./
COPY source source
COPY tests tests

WORKDIR ${SALIENCY_DIR_ROOT}/build

RUN cmake --no-warn-unused-cli -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING=Release ..

RUN cmake --build . --config Release --target install -- -j$(nproc)

WORKDIR $SALIENCY_DIR_ROOT


# make runtime environment and copy over libraries and built binaries
FROM ubuntu:20.04 AS saliency_runtime
LABEL maintainer="josephburling@gmail.com"

ARG saliency_build_dir=/opt/saliency
WORKDIR /

# libraries needed at runtime
# NOTE: need to find a way to reduce final image size
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && \
    apt-get install -y \
    libjpeg-turbo8 \
    libtbb2 \
    libgtk2.0-0 \
    ffmpeg

RUN mkdir /home/user
RUN groupadd -r user -g 777 && \
    useradd -u 431 -r -g user -d /home/user -s /sbin/nologin -c "Ubuntu User is \"user\"" user
RUN chown -R user:user /home/user
USER user
ENV SALIENCY_PATH=/home/user/saliency
WORKDIR $SALIENCY_PATH

# copy OpenCV objects from OPENCV_BUILD
COPY --from=saliency_build /usr/local/lib /usr/local/lib
COPY --from=saliency_build /usr/local/include/opencv4/opencv2 /usr/local/include/opencv4/opencv2
COPY --from=saliency_build /usr/local/share/opencv4 /usr/local/share/opencv4

WORKDIR $SALIENCY_PATH/bin

# copy saliency objects from saliency_build
COPY --from=saliency_build ${saliency_build_dir}/saliency/bin ./

# # copy other data from host
WORKDIR $SALIENCY_PATH/internal
COPY saliency/share/parameters.yml ./
COPY saliency/share/samples ./samples

ENV PATH=${SALIENCY_PATH}:$PATH

# start in an empty directory
WORKDIR $SALIENCY_PATH/runtime

# show help from saliency program as default option
ENTRYPOINT ["../bin/saliency", "-alt_exit"]
CMD ["--help"]
