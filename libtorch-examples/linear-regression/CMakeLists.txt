cmake_minimum_required(VERSION 3.19)
project(linear-regression)

set(CMAKE_CXX_STANDARD 14)
set(EXECUTABLE_NAME linear-regression)

add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE main.cpp include/linear_regression.hpp)
target_include_directories(${EXECUTABLE_NAME} PRIVATE include)
target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES})
