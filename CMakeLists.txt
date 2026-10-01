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
    # Browser build, see web/README.md
    return()
endif()

add_executable(othello-engine engine/cli.cpp)
target_link_libraries(othello-engine PRIVATE othello_engine)

enable_testing()
add_executable(engine_tests engine/tests/test_engine.cpp)
target_link_libraries(engine_tests PRIVATE othello_engine)
add_test(NAME engine_tests COMMAND engine_tests ${CMAKE_CURRENT_SOURCE_DIR}/engine/tests/data/positions.txt)
