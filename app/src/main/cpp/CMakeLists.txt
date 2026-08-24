cmake_minimum_required(VERSION 3.22.1)
project(facerobot_llama)

include(FetchContent)
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(llama_cpp)

add_library(facerobot_llama SHARED llama_bridge.cpp)
target_link_libraries(facerobot_llama llama common ${log-lib} log)
