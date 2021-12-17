#!/bin/bash -e
myRepo=$(pwd)
CMAKE_GENERATOR_OPTIONS=(-G"Visual Studio 16 2019" -A x64)  # CMake 3.14+ is required
if [  ! -d "$myRepo/opengv"  ]; then
    echo "cloning opengv"
    git clone https://github.com/laurentkneip/opengv.git
else
    cd opencv
    git pull --rebase
    cd ..
fi
RepoSource=opengv
mkdir -p build_opengv
pushd build_opengv
set -x
CMAKE_OPTIONS=(-DEIGEN_INCLUDE_DIR="../eigen/" -DBUILD_TESTS:BOOL=OFF)
cmake "${CMAKE_GENERATOR_OPTIONS[@]}" "${CMAKE_OPTIONS[@]}" -DCMAKE_INSTALL_PREFIX="$myRepo/install/$RepoSource" "$myRepo/$RepoSource"
echo "************************* $Source_DIR -->debug"
cmake --build .  --config debug
echo "************************* $Source_DIR -->release"
cmake --build .  --config release
cmake --build .  --target install --config release
cmake --build .  --target install --config debug
popd
