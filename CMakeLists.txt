cmake_minimum_required(VERSION 3.19)
project(libtorch-examples)

option(DOWNLOAD_DATASETS "Automatically download required datasets at build-time." ON)
option(INSTALL_TORCH "Automatically download and install libtorch." ON)
option(CUDA "Download GPU Version" ON)
option(DOWNLOAD_MNIST, "Download the MNIST Dataset during build" ON)

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
set(EXECUTABLE_NAME libtorch-examples)
set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/libtorch)

include_directories(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/libtorch-examples/include)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${EXECUTABLE_NAME} main.cpp)
target_link_libraries(${EXECUTABLE_NAME} "${TORCH_LIBRARIES}")

if(DOWNLOAD_DATASETS)

    set(DATA_DIR ${CMAKE_SOURCE_DIR}/data CACHE PATH "Dataset download directory")
    file(MAKE_DIRECTORY ${DATA_DIR})

    if(DOWNLOAD_MNIST)
        file(MAKE_DIRECTORY ${DATA_DIR}/mnist)
        set(MNIST_URL "http://yann.lecun.com/exdb/mnist")
        set(MNIST_ARCHIVES
            "train-images-idx3-ubyte.gz"
            "train-labels-idx1-ubyte.gz"
            "t10k-images-idx3-ubyte.gz"
            "t10k-labels-idx1-ubyte.gz")
        foreach(ARCHIVE ${MNIST_ARCHIVES})
            message("-- Downloading from ${MNIST_URL}/${ARCHIVE}")
            file(DOWNLOAD ${MNIST_URL}/${ARCHIVE} ${DATA_DIR}/mnist/${ARCHIVE})
            execute_process (
                COMMAND bash -c "gunzip ${DATA_DIR}/mnist/${ARCHIVE}"
                OUTPUT_VARIABLE outVar
            )
            execute_process (
                COMMAND bash -c "rm ${DATA_DIR}/mnist/${ARCHIVE}"
                OUTPUT_VARIABLE outVar
            )
        endforeach()
    endif()
endif()
