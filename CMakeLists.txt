cmake_minimum_required(VERSION 3.19)
project(libtorch-examples)

option(INSTALL_TORCH "Automatically download and install libtorch." ON)
option(CUDA "Download GPU Version" ON)
option(INSTALL_PYTHON_DEPENDENCIES "Install Python Dependencies during build" ON)
option(DOWNLOAD_MNIST "Download the MNIST Dataset during build" ON)
#option(DOWNLOAD_CIFAR10 "Download the CIFAR10 Dataset during build" ON)

message("-- Installing Python Dependencies")
if(INSTALL_PYTHON_DEPENDENCIES)
    execute_process (
        COMMAND bash -c "pip install --upgrade pip gdown --no-cache-dir"
        OUTPUT_VARIABLE outVar
    )
endif()

if(INSTALL_TORCH)
    if(APPLE)
        message("-- MacOSX Detected")
        message("-- Downloading LibTorch")
        file(DOWNLOAD
                https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.8.1.zip
                ${CMAKE_SOURCE_DIR}/libtorch.zip)
        message("-- Extracting LibTorch")
    else()
        message("-- Linux Detected")
        message("-- Downloading LibTorch")
        if(CUDA)
            file(DOWNLOAD
                    https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcu111.zip
                ${CMAKE_CURRENT_SOURCE_DIR}/libtorch.zip)
            message("-- Extracting LibTorch")
        else()
            file(DOWNLOAD
                    https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip
                ${CMAKE_CURRENT_SOURCE_DIR}/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcpu.zip)
            message("-- Extracting LibTorch")
        endif()
    endif()
    execute_process (
            COMMAND bash -c "unzip ${CMAKE_SOURCE_DIR}/libtorch.zip -d ${CMAKE_SOURCE_DIR}"
            OUTPUT_VARIABLE outVar
    )
    execute_process (
            COMMAND bash -c "rm ${CMAKE_SOURCE_DIR}/libtorch.zip"
            OUTPUT_VARIABLE outVar
    )
endif()

set(CMAKE_CXX_STANDARD 14)
set(EXECUTABLE_NAME main)
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/libtorch)

include_directories(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/libtorch-examples/include)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${EXECUTABLE_NAME} main.cpp)
target_link_libraries(${EXECUTABLE_NAME} "${TORCH_LIBRARIES}")

set(DATA_DIR ${CMAKE_SOURCE_DIR}/data CACHE PATH "Dataset download directory")
file(MAKE_DIRECTORY ${DATA_DIR})

if(DOWNLOAD_MNIST)
    message("-- Downloading MNIST Dataset")
    execute_process (
        COMMAND bash -c "gdown https://drive.google.com/uc?id=1aCeLWoq8dAT-_AXocbvWPj5T4JE3jdgS"
        OUTPUT_VARIABLE outVar
    )
    message("-- Expanding MNIST Dataset")
    execute_process (
        COMMAND bash -c "unzip mnist.zip -d ${DATA_DIR}"
        OUTPUT_VARIABLE outVar
    )
    execute_process (
        COMMAND bash -c "rm ./mnist.zip"
        OUTPUT_VARIABLE outVar
    )
endif()

#if(DOWNLOAD_CIFAR10)
#    message("-- Downloading CIFAR10 Dataset")
#    execute_process (
#            COMMAND bash -c "gdown https://drive.google.com/uc?id=1A45Gz3v6cEAvl-7THyK0LFRhndac16xj"
#            OUTPUT_VARIABLE outVar
#    )
#    execute_process (
#            COMMAND bash -c "gdown https://drive.google.com/uc?id=1A45Gz3v6cEAvl-7THyK0LFRhndac16xj"
#            OUTPUT_VARIABLE outVar
#    )
#endif()

# Linear Regression
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libtorch-examples/linear-regression)
add_custom_target(linear_regression)
add_dependencies(linear_regression linear-regression)

# Logistic Regression
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libtorch-examples/logistic-regression)
add_custom_target(logistic_regression)
add_dependencies(logistic_regression logistic-regression)

# Multi-Layered Perceptron
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libtorch-examples/multi-layered-perceptron)
add_custom_target(mlp_example)
add_dependencies(mlp_example multi-layered-perceptron)

# Convnet-Classification
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/libtorch-examples/convnet)
add_custom_target(convnet_example)
add_dependencies(convnet_example convnet)
