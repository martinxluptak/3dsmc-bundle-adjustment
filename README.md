# 3D Scanning & Motion capture Bundle adjustment project

## Setup instructions

Required on system: `python3` and `git`.\
The project uses Conan to simplify dependency management:

* to download binaries of each library so compiling them from sources on your machine is needed.
* to download the correct versions for your architecture and compiler.

### Linux Conan instructions

1. Execute the following:

```shell
# setup virtualenv
$ python -m venv 3dsenv
$ source 3dsenv\bin\activate
$ python -m pip install conan

# Switch to project directory
$ cd <PROJECT_DIRECTORY>
$ mkdir build
$ cd build

# Create a new conan profile and setup
$ conan profile new default --detect
$ conan profile update settings.compiler.libcxx=libstdc++11 default
$ conan install ..
# If the installation fails on this step, install any missing libraries with apt-get / dnf.
```

### Windows Conan instructions

1. start Developer Powershell for VS as administrator
2. Execute the following:

```shell
# Allow execution of scripts in powershell
Set-ExecutionPolicy RemoteSigned

# Setup virtualenv
Python -m venv 3dsenv
.\3dsenv\Scripts\activate
Python -m pip install conan

# Switch to project directory
cd <PROJECT_DIRECTORY>
mkdir build
cd build

# Create a new conan profile and setup
conan profile new default --detect
conan profile update settings.compiler.cppstd=14 default
conan profile update settings.compiler="Visual Studio" default
conan profile update settings.compiler.runtime=MD default
conan profile update settings.compiler.version=15 default
conan install ..
```

Your installed C++ MSVC compiler must be at least version 14 to ensure backwards compatibility.

### OpenCV and OpenGV
These libraries need to be built from source. Run the bash scripts found in `lib/` directory (use [git-bash](https://git-scm.com/downloads) on Windows) in the following order:
```
./installEigen.sh
./installOCV.sh
./installOGV.sh
```
After the compilation is complete, open `CMakeLists.txt` and set the `OpenCV_DIR` and `OpenGV_DIR` variables to navigate to
`opencv_build` and `opengv_build` directories respectively.

### Further setup instructions

1. After installing the libraries with Conan, import the project into Visual Studio / CLion and execute
the `CMakeLists.txt`.
2. Switch your build profile from Debug to Release, as the bundled binaries are also in Release. Attempting to build and
run with the Debug build profile results in an error.
3. If a missing OpenCV DLL error appears on launch on Windows, check CMake output for a path to append to Windows PATH
environment variable and append it. ([instructions](https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/))
4. With or without OpenGV, you should be able to build and run the `bundle_adjustment_tests` target, which imports
Eigen and Ceres.
5. With the OpenCV dependency oyu should be able to build and run the `bundle_adjustment_surf_flann_test` target to verify
you have the non-free algorithms available.

If you encounter issues with build architecture mismatch `x86 != x64`, make sure you are building an
executable for the same architecture as your libraries.\
To display the library architecture Conan downloaded, type `conan profile show default`.

## Source sets

Currently, there are two source sets: `src/` and `test/`, for project code and project unit tests respectively.\
These correspond to two build targets: `bundle_adjustment` and `bundle_adjustment_tests`. GoogleTest is chosen as the testing framework.