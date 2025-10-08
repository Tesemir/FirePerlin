cmake_minimum_required(VERSION 3.26.0)
project(perlin_noise VERSION 1.0.0)

cmake_policy(SET CMP0072 NEW)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(perlin_noise
    src/test.cpp
    src/glad.c  
)

target_include_directories(perlin_noise PRIVATE
    ${CMAKE_SOURCE_DIR}/dependencies  
)

target_link_libraries(perlin_noise
    ${GLFW3_LIBRARY}
    OpenGL::GL
)