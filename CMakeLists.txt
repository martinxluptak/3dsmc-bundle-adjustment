cmake_minimum_required(VERSION 3.16)
project(bundle_adjustment)
include(FetchContent)

set(CMAKE_CXX_STANDARD 14)


include(${CMAKE_BINARY_DIR}/../build/conanbuildinfo.cmake)

set(CONAN_DISABLE_CHECK_COMPILER OFF INTERNAL "Disable compiler compatibility check (for Visual Studio).")
conan_basic_setup()

FetchContent_Declare(
        opengv # sub-project name
        GIT_REPOSITORY git@github.com:laurentkneip/opengv.git # repository URL
        GIT_TAG master # Git commit ID
        GIT_SHALLOW TRUE # only pull latest commits (faster)
        GIT_PROGRESS TRUE # show progress
)
FetchContent_MakeAvailable(opengv)

set(HEADER_FILES
        ""
        )

set(SOURCE_FILES
        src/main.cpp
        )

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${CONAN_LIBS} opengv)

# Set files to be compiled for testing target
set(TEST_SOURCE_FILES
        test/main.cpp
        test/Exercise5LibsIncludeTest.cpp
        test/OpenCVIncludeTest.cpp
        )

# Create testing target with the same dependencies
add_executable(${PROJECT_NAME}_tests ${HEADER_FILES} ${TEST_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}_tests ${CONAN_LIBS})