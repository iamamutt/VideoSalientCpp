::: ------------------------------------------------------------------------------------------------
::: Joseph M. Burling <josephburling@gmail.com> 2020
::: ------------------------------------------------------------------------------------------------
@setlocal enabledelayedexpansion
@echo off
echo.
goto start_script
:: If error, may need to make changes to the source files
::   - in opencv/modules/videoio/src/cap_dshow.cpp add before #include <Dshow.h>
::      #define NO_DSHOW_STRSAFE
::
::   - in opencv\modules\python\src2\cv2.cpp add before #include <Python.h>
::      #define _hypot hypot
::
::  USAGE: build_opencv.bat [compiler] [CMake generator] [OpenCV Source] [OpenCV Install] "<CMake Options>" <toolchain path>
::
::  --- gcc example below
:: build_opencv.bat gcc "MinGW Makefiles" %USERPROFILE%\lib\opencv-4.5.0 %USERPROFILE%\lib\gcc\opencv "-DOPENCV_EXTRA_MODULES_PATH=%USERPROFILE%\lib\opencv\opencv_contrib\modules\xfeatures2d" ""
:start_script
set __pwd=%~dp0
set __cwd=%cd%
set __compiler=%~1
set __cmake_generator=%~2
set __opencv_source_dir=%~3
set __opencv_install_dir=%~4
set __add_cmake_opts=%~5
set __toolchain_path=%~6
set _FAILED=0

if "%__compiler%"=="" (
    echo.
    echo Invalid toolchain/compiler specification, use "gcc" or "msvc"
    set _FAILED=1
    goto end_build_script
)

if "%__cmake_generator%"=="" (
    echo.
    echo Invalid cmake generator. See `cmake --help` and the `-G` option.
    set _FAILED=1
    goto end_build_script
)

if "%__opencv_source_dir%"=="" (
    echo.
    echo Invalid OpenCV source directory
    set _FAILED=1
    goto end_build_script
)

set __opencv_build_dir=%__opencv_source_dir%\build
if "%__opencv_install_dir%"=="" set __opencv_install_dir=%__opencv_source_dir%\install

cd /d "%__opencv_source_dir%"
set __opencv_source_dir=%cd%

mkdir %__opencv_build_dir% > nul
cd /d "%__opencv_build_dir%"
echo Changing current directory to: "%__opencv_build_dir%"
set __opencv_build_dir=%cd%
if exist CMakeCache.txt del CMakeCache.txt

:: OpenCV build options
cmake -G "%__cmake_generator%" ^
-DCMAKE_INSTALL_PREFIX="%__opencv_install_dir%" ^
-DBUILD_DOCS=OFF ^
-DBUILD_EXAMPLES=OFF ^
-DBUILD_JAVA=OFF ^
-DBUILD_opencv_apps=OFF ^
-DBUILD_opencv_java_bindings_generator=OFF ^
-DBUILD_opencv_java=OFF ^
-DBUILD_opencv_python_bindings_generator=OFF ^
-DBUILD_opencv_python2=OFF ^
-DBUILD_opencv_python3=OFF ^
-DBUILD_PERF_TESTS=OFF ^
-DBUILD_SHARED_LIBS=OFF ^
-DBUILD_TESTS=OFF ^
-DENABLE_PRECOMPILED_HEADERS=OFF ^
-DFORCE_VTK=OFF ^
-DINSTALL_PYTHON_EXAMPLES=OFF ^
-DINSTALL_TESTS=OFF ^
-DOPENCV_ENABLE_NONFREE=ON ^
-DOPENCV_FORCE_PYTHON_LIBS=OFF ^
-DWITH_1394=OFF ^
-DWITH_CUDA=OFF ^
-DWITH_EIGEN=OFF ^
-DWITH_FFMPEG=ON ^
-DWITH_GDAL=OFF ^
-DWITH_GSTREAMER=OFF ^
-DWITH_OPENCL=ON ^
-DWITH_OPENEXR=OFF ^
-DWITH_OPENGL=ON ^
-DWITH_QT=OFF ^
-DWITH_QUIRC=OFF ^
-DWITH_VTK=OFF ^
-DWITH_XINE=OFF ^
%__add_cmake_opts% ^
--build "%__opencv_source_dir%"

::: ------------------------------------------------------------------------------------------------
if %__compiler%==msvc (
    call "%__toolchain_path%" x64
    cd "%__opencv_build_dir%"
    MSBuild.exe OpenCV.sln /p:Configuration=Debug /p:Platform=x64 /p:BuildProjectReferences=false /v:m /m
    MSBuild.exe OpenCV.sln /p:Configuration=Release /p:Platform=x64 /p:BuildProjectReferences=false /v:m /m
    goto end_build_script
)

if %__compiler%==gcc (
    mingw32-make -j3
    mingw32-make install
    goto end_build_script
)

:end_build_script

endlocal & (
    set _FAILED=%_FAILED%
    set __build_root=%__opencv_build_dir%
    set __install_root=%__opencv_install_dir%
    cd /d "%__cwd%"
)

ping -n 2 127.0.0.1 > nul
