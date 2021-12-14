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

### Further setup instructions

1. After installing the libraries with Conan, import the project into Visual Studio / CLion and execute
the `CMakeLists.txt`.
2. Switch your build profile from Debug to Release, as the bundled binaries are also in Release. Attempting to build and
run with the Debug build profile results in an error.
3. Dependencies should become available without further setup. OpenGV is an exception, as it is not distributed for Windows
in Conan and must be built from source. This should not take more than 5 minutes on a modern system.
4. With or without OpenGV, you should be able to build and run the `bundle_adjustment_tests` target, which imports
Eigen and Ceres.

If you encounter issues with build architecture mismatch `x86 != x64`, make sure you are building an
executable for the same architecture as your libraries.\
To display the library architecture Conan downloaded, type `conan profile show default`.

## Source sets

Currently, there are two source sets: `src/` and `test/`, for project code and project unit tests respectively.\
These correspond to two build targets: `bundle_adjustment` and `bundle_adjustment_tests`. GoogleTest is chosen as the testing framework.