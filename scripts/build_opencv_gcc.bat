::: ------------------------------------------------------------------------------------------------
::: Joseph M. Burling <josephburling@gmail.com> 2020
::: ------------------------------------------------------------------------------------------------
@echo off

cls
set pwd=%~dp0
set cwd=%cd%

:: Use GNU compiler
set toolchain=gcc

:: Path to compiler bin folder containing g++
set toolchain_path="%LOCALAPPDATA%\mingw-w64\bin"

:: Cmake generator name for gcc makefiles
set cmake_generator=MinGW Makefiles

:: Path where the OpenCV source files have been unzipped
set opencv_src_root="%USERPROFILE%\Downloads\opencv-4.5.0"

:: Path where OpenCV will be installed
set opencv_install_root="%USERPROFILE%\lib\gcc\opencv"

:: Additional cmake options
set cmake_options="-DCMAKE_CONFIGURATION_TYPES=Release"

cd /d "%pwd%"
echo.
echo Changing directory to scripts folder: %cd%

:: run this script from the scripts subfolder
call "..\opencv\build_opencv.bat" %toolchain% "%cmake_generator%" %opencv_src_root% %opencv_install_root% %cmake_options% %toolchain_path%

echo.
echo Changing directory working directory: %cwd%
cd /d "%cwd%"
echo.
if _FAILED==1 (echo Build did not complete!)
pause
