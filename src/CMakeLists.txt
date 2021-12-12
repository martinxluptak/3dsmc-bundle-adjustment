cmake_minimum_required(VERSION 3.16)
project(bundle_adjustment)

set(CMAKE_CXX_STANDARD 14)

include(FetchContent)

# Declare dependency
FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG master
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
)


# Set options for building the Eigen sub-project.
option(EIGEN_BUILD_DOC OFF)
option(EIGEN_BUILD_TESTING OFF)
option(EIGEN_BUILD_PKGCONFIG OFF)

# Make dependency available
FetchContent_MakeAvailable(Eigen)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
