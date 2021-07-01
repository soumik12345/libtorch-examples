cmake_minimum_required(VERSION 3.19)
project(project1)

set(CMAKE_CXX_STANDARD 14)
set(EXECUTABLE_NAME project1)

add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE main.cpp include/sample.hpp)
target_include_directories(${EXECUTABLE_NAME} PRIVATE include)
target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES})
