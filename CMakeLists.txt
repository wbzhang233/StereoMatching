cmake_minimum_required(VERSION 3.15)
project(StereoMatch)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
find_package(OpenCV REQUIRED)
message("${OpenCV_DIR}")
INCLUDE_DIRECTORIES("/usr/include/eigen3/")

# 1-立体匹配
add_executable(StereoMatch main.cpp stereoMatch.cpp stereoMatch.h )
target_link_libraries(StereoMatch ${OpenCV_LIBS})

# 2-立体匹配案例
add_executable(StereoMatchExp stereoExample.cpp stereoMatch.cpp stereoMatch.h )
target_link_libraries(StereoMatchExp ${OpenCV_LIBS})

# 3-立体匹配序列图测试
add_executable(stereo_seq stereo_forseq.cpp stereoMatch.cpp stereoMatch.h )
target_link_libraries(stereo_seq ${OpenCV_LIBS})