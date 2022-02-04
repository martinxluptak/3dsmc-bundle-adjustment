BUILD_PANGOLIN=build-pangolin

./pangolin_prerequisites.sh recommended

mkdir -p "$BUILD_PANGOLIN"
pushd "$BUILD_PANGOLIN"
cmake ../Pangolin \
    -DCMAKE_FIND_FRAMEWORK=LAST \
    -DEXPORT_Pangolin=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PANGOLIN_PYTHON=OFF
make -j1
