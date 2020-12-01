cmake_minimum_required(VERSION 3.15.0)
project(Saliency VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(PROJECT_SOURCE_DIR "${CMAKE_SOURCE_DIR}/source")

set(EXEC_NAME
        "saliency"
        CACHE STRING "Name of the main executable program")

set(OPENCV_INSTALL_DIR
        "${CMAKE_SOURCE_DIR}/opencv"
        CACHE PATH "Root path where the OpenCV library is installed")

find_package(OpenCV REQUIRED HINTS "${OPENCV_INSTALL_DIR}")
include_directories(${OpenCV_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})

add_executable(${EXEC_NAME} main.cpp)
add_executable(highgui_test tests/highgui_test.cpp)

target_link_libraries(${EXEC_NAME} ${OpenCV_LIBS})
target_link_libraries(highgui_test ${OpenCV_LIBS})

# try to find FFMPEG opencv library for video capture on windows
if (WIN32)
    find_file(OPENCV_FFMPEG_LIB
            NAMES opencv_videoio_ffmpeg.dll opencv_videoio_ffmpeg450_64.dll
            PATHS "${OPENCV_INSTALL_DIR}/../bin" "${CMAKE_SOURCE_DIR}/saliency/bin"
            HINTS "${OPENCV_INSTALL_DIR}/.." "${CMAKE_SOURCE_DIR}")
    if (NOT OPENCV_FFMPEG_LIB)
        file(GLOB_RECURSE OPENCV_FFMPEG_LIB
                LIST_DIRECTORIES false
                opencv_videoio_ffmpeg*)
        if (NOT OPENCV_FFMPEG_LIB)
            message("FFMPEG not found. Video capture may not work.")
        endif ()
    endif ()
    if (OPENCV_FFMPEG_LIB)
        message("FFMPEG found. Copy to path from: ${OPENCV_FFMPEG_LIB}")
    endif ()
endif ()

install(TARGETS ${EXEC_NAME} highgui_test RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/saliency/bin")
