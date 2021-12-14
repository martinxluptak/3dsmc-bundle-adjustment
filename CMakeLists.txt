cmake_minimum_required(VERSION 3.16)
project(bundle_adjustment)

set(CMAKE_CXX_STANDARD 14)


add_definitions("-std=c++11")

include(${CMAKE_BINARY_DIR}/../build/conanbuildinfo.cmake)
conan_basic_setup()

set(HEADER_FILES
        ""
        )

set(SOURCE_FILES
        src/main.cpp
        )

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${CONAN_LIBS})

# Set files to be compiled for testing target
set(TEST_SOURCE_FILES
        test/main.cpp
        test/Exercise5LibsIncludeTest.cpp
        test/OpenCVIncludeTest.cpp
        )

# Create testing target with the same dependencies
add_executable(${PROJECT_NAME}_tests ${HEADER_FILES} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}_tests ${CONAN_LIBS})