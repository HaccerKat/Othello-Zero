cmake_minimum_required(VERSION 3.20)
project(OthelloZero LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wshadow)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_CXX_FLAGS_DEBUG "-O1 -g -D_GLIBCXX_ASSERTIONS -fsanitize=address,undefined")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address,undefined")
endif()

add_library(othello_engine STATIC engine/board.cpp engine/nnue.cpp engine/search.cpp)
target_include_directories(othello_engine PUBLIC engine)

if(EMSCRIPTEN)
    # Browser build: emcmake cmake -B build-web && cmake --build build-web
    # writes web/engine.js and web/engine.wasm next to the page.
    add_executable(engine engine/wasm.cpp)
    target_link_libraries(engine PRIVATE othello_engine)
    target_compile_options(othello_engine PRIVATE -msimd128)
    target_link_options(engine PRIVATE
        -sENVIRONMENT=worker -sMODULARIZE -sEXPORT_NAME=createEngine
        -sEXPORTED_FUNCTIONS=_best_move -sEXPORTED_RUNTIME_METHODS=cwrap
        -sALLOW_MEMORY_GROWTH -sINITIAL_MEMORY=64MB)
    set_target_properties(engine PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/web)
    return()
endif()

add_executable(othello-engine engine/cli.cpp)
target_link_libraries(othello-engine PRIVATE othello_engine)

enable_testing()
add_executable(engine_tests engine/tests/test_engine.cpp)
target_link_libraries(engine_tests PRIVATE othello_engine)
add_test(NAME engine_tests COMMAND engine_tests ${CMAKE_CURRENT_SOURCE_DIR}/engine/tests/data/positions.txt)
