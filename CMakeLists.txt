cmake_minimum_required(VERSION 3.5.1)
project(opencv_ssd)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(cartoon ${SOURCE_FILES})

set(OpenCV_LIBS
        "/usr/local/lib/libopencv_core.so"
        "/usr/local/lib/libopencv_dnn.so"
        "/usr/local/lib/libopencv_imgproc.so"
        "/usr/local/lib/libopencv_video.so"
        "/usr/local/lib/libopencv_videoio.so"
        "/usr/local/lib/libopencv_imgcodecs.so"
        "/usr/local/lib/libopencv_highgui.so"
        )

target_link_libraries(cartoon ${OpenCV_LIBS})
