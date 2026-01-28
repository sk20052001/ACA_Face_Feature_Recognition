cmake_minimum_required(VERSION 3.16)
project(face_feature_baseline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# If dlib is installed system-wide with CMake config:
find_package(dlib REQUIRED)

add_executable(baseline_serial
    src/baseline_serial.cpp
)

target_include_directories(baseline_serial PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(baseline_serial PRIVATE ${OpenCV_LIBS} dlib::dlib)
