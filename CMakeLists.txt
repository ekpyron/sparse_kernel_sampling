cmake_minimum_required(VERSION 3.0)
project(sparse_kernel_sampling)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

include_directories(contrib/mpreal)

add_subdirectory(src)
