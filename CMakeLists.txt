cmake_minimum_required(VERSION 3.10)
project(NN_cpp)

set(CMAKE_CXX_STANDARD 17)

# add_executable(myproject
#     main.cpp
#     # helper.cpp
#     # math_utils.cpp
# )

file(GLOB SOURCES *.cpp)
add_executable(NN_cpp ${SOURCES})


add_custom_target(run
    COMMAND NN_cpp
    DEPENDS NN_cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)