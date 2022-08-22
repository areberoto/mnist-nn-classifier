cmake_minimum_required(VERSION 3.10)

# set the project name
project(mnist_classifier)

#Use OpenCV libraries
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# This fixes the clangd include path issue in vscode
# If not using vscode + clangd, this can be removed
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add the executable
add_executable(mnist_classifier src/main.cpp src/matrix.cpp src/mnist.cpp src/network.cpp)

target_link_libraries(mnist_classifier ${OpenCV_LIBS})