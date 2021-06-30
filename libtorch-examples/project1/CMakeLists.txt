cmake_minimum_required(VERSION 3.19)
project(project1)

set(CMAKE_CXX_STANDARD 14)
set(EXECUTABLE_NAME project1)

add_executable(${EXECUTABLE_NAME} main.cpp)
