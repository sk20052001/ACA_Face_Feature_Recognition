cmake_minimum_required(VERSION 3.16)
project(face_feature_baseline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

find_package(OpenMP)

add_executable(baseline_omp_preload
    src/baseline_omp_preload.cpp
)
target_link_libraries(baseline_omp_preload PRIVATE ${OpenCV_LIBS} dlib::dlib)

if(OpenMP_CXX_FOUND)
    target_link_libraries(baseline_omp_preload PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(baseline_omp_preload PRIVATE USE_OPENMP=1)
endif()


add_executable(baseline_serial
    src/baseline_serial.cpp
)
target_link_libraries(baseline_serial PRIVATE ${OpenCV_LIBS} dlib::dlib)

add_executable(baseline_omp
    src/baseline_omp.cpp
)
target_link_libraries(baseline_omp PRIVATE ${OpenCV_LIBS} dlib::dlib)

if(OpenMP_CXX_FOUND)
    target_link_libraries(baseline_omp PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(baseline_omp PRIVATE USE_OPENMP=1)
endif()