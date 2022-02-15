cmake_minimum_required(VERSION 3.16)
project(bundle_adjustment)

set(CMAKE_CXX_STANDARD 17)


include("${CMAKE_CURRENT_SOURCE_DIR}/build/conanbuildinfo.cmake")

set(CONAN_DISABLE_CHECK_COMPILER OFF INTERNAL "Disable compiler compatibility check (for Visual Studio).")
conan_basic_setup()

set(OpenCV_DIR ".../lib/build_opencv") # SET TO <your_path>/lib/build_opencv
set(Pangolin_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/build-pangolin/src")

message(STATUS "===Are these directories correct?===")
message(STATUS "opencv_DIR:${OpenCV_DIR}")
message(STATUS "=========")


find_package(OpenCV REQUIRED)
include_directories(headers ${OpenCV_INCLUDE_DIRS}) # uncomment if OpenCV includes are causing errors (missing from PATH).
find_package(Pangolin REQUIRED CONFIG NO_DEFAULT_PATH)

set(HEADER_FILES
        "headers/Eigen.h" # we don't need it, right?
        "headers/VirtualSensor.h"
        "headers/Detection.h"
        "headers/Matching.h"
        "headers/Map3D.h"
        "headers/CommonTypes.h"
        "headers/BundleAdjustmentConfig.h"
        "headers/OptimizationUtils.h"
        "headers/helpers.h"
        )

set(SOURCE_FILES
        src/main.cpp
        src/Detection.cpp
        src/Matching.cpp
        src/Map3D.cpp
        src/OptimizationUtils.cpp)
include_directories("headers/")
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${CONAN_LIBS} ${OpenCV_LIBS})

# Set files to be compiled for testing target
set(TEST_SOURCE_FILES
        test/main.cpp
        test/Exercise5LibsIncludeTest.cpp
        src/main.cpp
        )

# Create testing target with the same dependencies
add_executable(${PROJECT_NAME}_tests ${HEADER_FILES} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}_tests ${CONAN_LIBS} ${OpenCV_LIBS} pangolin)

add_executable(${PROJECT_NAME}_surf_flann_test test/SurfFlannMatchingTest.cpp)
target_link_libraries(${PROJECT_NAME}_surf_flann_test ${CONAN_LIBS} ${OpenCV_LIBS})

add_executable(${PROJECT_NAME}_main src/main.cpp)
target_link_libraries(${PROJECT_NAME}_main ${CONAN_LIBS} ${OpenCV_LIBS} pangolin)
