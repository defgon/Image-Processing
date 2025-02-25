cmake_minimum_required(VERSION 3.22)
project(ImageProcessing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SRC_FILES src/*.cpp)
file(GLOB_RECURSE HEADER_FILES src/*.h)

include_directories(dft img smpls src)

add_executable(ImageProcessing ${SRC_FILES} ${HEADER_FILES})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
