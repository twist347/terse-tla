cmake_minimum_required(VERSION 3.20)
project(terse-tla
        VERSION 0.1.0
        DESCRIPTION ""
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(tla INTERFACE)
add_library(tla::tla ALIAS tla)
target_include_directories(tla INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

option(TLA_BUILD_TEST "Build unit tests" ${PROJECT_IS_TOP_LEVEL})

if (TLA_BUILD_TEST)
    add_subdirectory(test)
endif ()
