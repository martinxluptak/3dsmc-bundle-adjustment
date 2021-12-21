cmake_minimum_required(VERSION 3.16)
project(bundle_adjustment)

set(CMAKE_CXX_STANDARD 17)


include("${CMAKE_CURRENT_SOURCE_DIR}/build/conanbuildinfo.cmake")

set(CONAN_DISABLE_CHECK_COMPILER OFF INTERNAL "Disable compiler compatibility check (for Visual Studio).")
conan_basic_setup()

set(OpenCV_DIR ".../build_opencv") # SET TO <your_path>/lib/build_opencv
set(OpenGV_DIR ".../build_opengv") # SET TO <your_path>/lib/build_opengv

message(STATUS "===Are these directories correct?===")
message(STATUS "opencv_DIR:${OpenCV_DIR}")
message(STATUS "opengv_DIR:${OpenGV_DIR}")
message(STATUS "=========")


find_package(OpenCV REQUIRED)
find_package(OpenGV REQUIRED)

set(HEADER_FILES
        "headers/Eigen.h"
        "headers/FreeImageHelper.h"
        "headers/VirtualSensor.h"
        )

set(SOURCE_FILES
        src/main.cpp
        src/FreeImageHelper.cpp
        )

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${CONAN_LIBS} ${OpenCV_LIBS} opengv freeimage)

# Set files to be compiled for testing target
set(TEST_SOURCE_FILES
        test/main.cpp
        test/Exercise5LibsIncludeTest.cpp
        )

# Create testing target with the same dependencies
add_executable(${PROJECT_NAME}_tests ${HEADER_FILES} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}_tests ${CONAN_LIBS} ${OpenCV_LIBS} opengv)

add_executable(${PROJECT_NAME}_surf_flann_test ${HEADER_FILES} test/SurfFlannMatchingTest.cpp)
