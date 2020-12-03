# Bottom-up Saliency Model

## Description

Detect visual saliency from video or images.

![saliency](media/saliency.gif)


*NOTES*:

- Need to implement inhibition of return for output saliency map
- USB camera option available but currently getting slow FPS.

## Program Usage

Running `./saliency --help` will show the following output:

```
Usage: saliency [params]

        -?, -h, --help
                print this help message
        --alt_exit
                sets program exit to right click on main image
        --cam
                usb camera index, use 0 for default
        --debug
                toggle visualization of feature parameters
        --dir
                full path to where root saliency output directory will be created
        --img
                full path to image file
        --par
                full path to the YAML parameters file
        --split
                output will be saved as a series of images instead of video
        --vid
                full path to video file
```

Assuming you are in the directory containing the `saliency` program.

### Using video as input

Point to a video called `my_video.avi`, use your own parameters settings from `my_settings.yml`, and export data to the `exported` folder.

```
saliency --vid=my_video.avi --par=my_settings.yml --dir=./exported
```

### Using an image as input

Export will be a video if `--dir` is specified, even though input is an image. The video will be the length ...

### Using a USB camera device as input

### Using custom saliency model parameters


## Building from source

Download repository contents to your user folder (you can download anywhere but the example below uses the user folder). If you already have git installed, you can do the following in a terminal.

```
cd ~
git clone https://github.com/iamamutt/VideoSalientCpp.git
```

Anytime there are updates to the source code, you can navigate to the `VideoSalientCpp` folder and pull the new changes with:

```
cd ~/VideoSalientCpp
git pull
```

### OSX

You will need some developer tools to build the program, open up the terminal app and run the following:

```
xcode-select --install
```

You'll also need Homebrew to grab the rest of the libraries and dependencies: https://brew.sh/

After Homebrew is installed, run:

```
brew update
brew install cmake opencv ffmpeg
```

After the above dependencies are installed, navigate to the repository folder, e.g., if you saved the contents to `~/VideoSalientCpp` then run `cd ~/VideoSalientCpp`. Once in the folder root, run the following to build the `saliency` binary.

```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release --target install -- -j 4
cd ..
```

The compiled binaries will be in `./saliency/bin`. Test using the samples data:

```
cd saliency
./bin/saliency --vid=share/samples/vtest.avi
```

Hold the "ESC" key down to quit (make sure the saliency output window is selected).


### Windows

Install dependencies:

- OpenCV (>=4.5.0) https://github.com/opencv/opencv/archive/4.5.0.zip
- CMake Binary distribution https://cmake.org/download/

TODO: OpenCV Build instructions

...

You'll also need to add the `opencv_videoio_ffmpeg*` library to the same folder as the program in order to read videos with non-standard codecs (e.g., on Windows, locate file `opencv_videoio_ffmpeg450_64.dll` and place in the folder that contains `saliency.exe`).

TODO: CMake build instructions

...


## Using the Docker image

If you don't want to build from source you can use the docker image to run the program. The image can be found in [Releases](https://github.com/iamamutt/VideoSalientCpp/releases).

### Setup

1. Install Docker Desktop: https://www.docker.com/get-started
2. Open the application after install. Allow privileged access if prompted. 
3. Check that Docker works from the command line. From a terminal type: `docker --version`.
4. Obtain the docker image `saliency-image.tar.gz` from "releases" on GitHub.
5. In a terminal, navigate to the directory containing the docker image and load it with `docker load -i saliency-image.tar.gz`
6. Run the image by entering the command `docker run -it --rm saliency-app:v0.1.0`. You should see the saliency program help documentation.

### Configure

#### Editing `docker-compose.yml`

Open up the file `docker-compose.yml` in any text editor. The fields that need to be changed are `environment`, `command`, and `source`.

- `source`: The keys `volumes: source:` maps a directory on your host machine to a directory inside the docker container. E.g., `source: /path/to/my/data`. If your data is located on your computer at `~/videos`, then use the full absolute path such as `source: $USERPROFILE/videos` on windows or `source: $HOME/videos` for unix-based systems. To use the samples from this repo, set the source mount as: `source: <saliency>/saliency/share`, where `<saliency>` is the full path to the `VideoSalientCpp` repo folder.

- `command`: These are the command-line arguments passed to the saliency program. If you want to specify a video to use with the `--vid=` option then use the relative path from the mapped volume. E.g., `--vid=my_video.avi` which may be located in `~/videos`.

- `environment`: Change `DISPLAY=#.#.#.#:0.0` to whatever your IP address is. If your IP address is `192.168.0.101` then the field would be `- DISPLAY=192.168.0.101:0.0`. This setting is required for displaying output windows. See the `Displaying windows` section below.

#### Running `docker-compose.yml`

After configuring `docker-compose.yml`, run the `saliency` service by entering this in the terminal:

```
docker-compose run --rm saliency
```

#### Displaying windows

When running from the docker container, the saliency program tries to show windows of the saliency output. These windows are generated by OpenCV and require access to the display. This access is operating system dependent, and without some way to map the display from the container to your own host machine you will generate an error such as `Can't initialize GTK backend in function 'cvInitSystem'`.

##### Windows

1. Download and install VcXsrv from here: https://sourceforge.net/projects/vcxsrv/
2. Run XLaunch and use all the default settings except for the last where it says "Disable access control." Make sure its selected.

![VcXsrv setting](media/vcxsrv_opt.png)

You should now be able to run the program and see the output windows.

##### OSX


```
brew install socat
socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\"
brew install xquartz
open -a Xquartz
```

XQuartz > preferences > Security > allow connections from network clients

```
ifconfig en0
docker run -e DISPLAY=10.0.0.134:0 --privileged saliency-app:v0.1.0
```



<!--
# cd opencv && docker build . -t opencv-build:v1.0.0
# docker run --rm -it opencv-build:v1.0.0

# docker build . --target saliency_build -t tmp-build:0.0.1
# docker run -it --rm tmp-build:0.0.1

# docker build . -t saliency-app:v0.1.0
# docker run -it --rm --entrypoint /bin/bash saliency-app:v0.1.0
# docker run -e DISPLAY=10.0.0.34:0.0 -p 5000:5000 -p 8888:8888 -it --rm saliency-app:v0.1.0
# docker run --device=/dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -p 5000:5000 -p 8888:8888 -it --rm saliency-app:v0.1.0

docker run -e DISPLAY=10.0.0.134:0 --privileged saliency-app:v0.1.0 -c --vid=../internal/samples/vtest.avi
-->
