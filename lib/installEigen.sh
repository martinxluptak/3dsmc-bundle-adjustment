#!/bin/bash -e
myRepo=$(pwd)
CMAKE_GENERATOR_OPTIONS=(-G"Visual Studio 16 2019" -A x64)  # CMake 3.14+ is required
if [  ! -d "$myRepo/eigen"  ]; then
    echo "cloning eigen"
    git clone https://gitlab.com/libeigen/eigen.git
else
    cd eigen
    git pull --rebase
    cd ..
fi

RepoSource=eigen
mkdir -p build_eigen
pushd build_eigen
set -x
cmake "${CMAKE_GENERATOR_OPTIONS[@]}" -DCMAKE_INSTALL_PREFIX="$myRepo/install/$RepoSource" "$myRepo/$RepoSource"
echo "************************* $Source_DIR -->debug"
cmake --build .  --config debug
echo "************************* $Source_DIR -->release"
cmake --build .  --config release
cmake --build .  --target install --config release
cmake --build .  --target install --config debug
popd
