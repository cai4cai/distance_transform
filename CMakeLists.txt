cmake_minimum_required(VERSION 3.25)
project(distance_transform)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Warnings compiler flags
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Optionally enable clang-tidy
find_program(CLANG_TIDY_EXE NAMES clang-tidy PATHS /opt/homebrew/opt/llvm/bin/)
if(NOT CLANG_TIDY_EXE)
  message(STATUS "clang-tidy not found. Skipping corresponding checks.")
else()
  set(CMAKE_CXX_CLANG_TIDY2 
    ${CLANG_TIDY_EXE};
    -header-filter=.*;
    -checks=-*,portability-*,bugprone-*,readability-,clang-analyzer-*,performance-*,cppcoreguidelines-*;
    )
  message(STATUS "Found clang-tidy: ${CLANG_TIDY_EXE}.")
endif()

add_library(distance_transform INTERFACE)
target_include_directories(distance_transform INTERFACE "include")

add_subdirectory(example)