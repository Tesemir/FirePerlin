cmake_minimum_required(VERSION 3.26.0)
project(perlin_noise VERSION 1.0.0)

cmake_policy(SET CMP0072 NEW)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

set(IMGUI_SRC
        ${CMAKE_SOURCE_DIR}/dependencies/imgui/imgui.cpp
        ${CMAKE_SOURCE_DIR}/dependencies/imgui/imgui_draw.cpp
        ${CMAKE_SOURCE_DIR}/dependencies/imgui/imgui_tables.cpp
        ${CMAKE_SOURCE_DIR}/dependencies/imgui/imgui_widgets.cpp
        ${CMAKE_SOURCE_DIR}/dependencies/imgui/backends/imgui_impl_glfw.cpp
        ${CMAKE_SOURCE_DIR}/dependencies/imgui/backends/imgui_impl_opengl3.cpp
)

add_executable(perlin_noise
    src/main.cpp
    src/glad.c
    ${IMGUI_SRC}  
)

target_include_directories(perlin_noise PRIVATE
    ${CMAKE_SOURCE_DIR}/dependencies  
    ${CMAKE_SOURCE_DIR}/dependencies/imgui
    ${CMAKE_SOURCE_DIR}/dependencies/imgui/backends
)

target_link_libraries(perlin_noise
    ${GLFW3_LIBRARY}
    OpenGL::GL
)