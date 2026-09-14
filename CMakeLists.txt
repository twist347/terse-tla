cmake_minimum_required(VERSION 3.23)
project(terse-tla
        VERSION 0.1.0
        DESCRIPTION "Terse linear algebra in C++23"
        LANGUAGES CXX
)

if (PROJECT_IS_TOP_LEVEL)
    if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    endif ()
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif ()

set(TLA_WARNINGS
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>
)

option(TLA_BUILD_TEST "Build unit tests" ${PROJECT_IS_TOP_LEVEL})

add_library(tla INTERFACE)
add_library(tla::tla ALIAS tla)

target_compile_features(tla INTERFACE cxx_std_23)
target_sources(tla INTERFACE
    FILE_SET HEADERS
        BASE_DIRS include
        FILES include/tla/tla.h
)

if (TLA_BUILD_TEST)
    enable_testing()
    add_subdirectory(test)
endif ()
