cmake_minimum_required(VERSION 3.22.1)
project(facerobot_llama)

# Ang Android Gradle Plugin ay nagpapasa ng CMAKE_BUILD_TYPE=Debug papunta sa buong
# native build kapag "assembleDebug" ang ginagawa (nakita natin ito sa gradle log mo:
# "CMAKE_BUILD_TYPE=Debug"). Ibig sabihin, TUMATAKBO ANG LLAMA.CPP NANG WALANG
# OPTIMIZATION (-O0) - kaya umaabot ng ilang minuto ang isang maikling sagot na dapat
# ilang segundo lang. I-force natin dito ang -O3 nang direkta bilang compiler flags,
# kahit "Debug" pa rin ang variant ng buong Android app (Kotlin/Java side lang
# apektado noon - hindi kailangang mabagal din ang native na Llama inference).
# Kailangang ilagay ito BAGO ang FetchContent_MakeAvailable() para ma-inherit ito
# ng mga target ng llama.cpp mismo.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")

include(FetchContent)
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    # Naka-pin sa isang partikular na release TAG (b10336, na-release Agosto 10, 2026,
    # commit f401bb1 sa totoong ggml-org/llama.cpp repo) imbis na "master" - para hindi
    # na basta-basta magbago ang API/behavior sa susunod mong rebuild kapag nag-update
    # ang upstream llama.cpp. Ginamit ang PANGALAN ng tag (hindi raw commit hash) dahil
    # mas simple at maaasahan itong i-checkout kahit sa shallow clone. Kung gusto mong
    # palitan ito ng ibang bersyon, tingnan ang https://github.com/ggml-org/llama.cpp/tags
    # para sa listahan ng mga totoong release tag.
    GIT_TAG        b10336
    GIT_SHALLOW    TRUE
)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
# Kailangan i-force ON ito dahil ang default value niya ay LLAMA_STANDALONE,
# at OFF ang LLAMA_STANDALONE kapag FetchContent subproject tayo (hindi top-level build).
# Kapag OFF ito, hindi tinatawag ang add_subdirectory(common), kaya walang "common" target
# na mabubuo -> "unable to find library -lcommon" sa linker.
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(llama_cpp)

add_library(facerobot_llama SHARED llama_bridge.cpp)
target_link_libraries(facerobot_llama llama llama-common ${log-lib} log)
