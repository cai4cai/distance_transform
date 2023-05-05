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
  set(CMAKE_CXX_CLANG_TIDY 
    ${CLANG_TIDY_EXE};
    -header-filter=.*;
    -checks=-*,portability-*,bugprone-*,-bugprone-easily-swappable-parameters,readability-,clang-analyzer-*,performance-*;
    )
    # ,cppcoreguidelines-*,-cppcoreguidelines-avoid-magic-numbers
  message(STATUS "Found clang-tidy: ${CLANG_TIDY_EXE}.")
endif()

add_library(distance_transform INTERFACE)
target_include_directories(distance_transform INTERFACE "include")

option(DOPE_USE_EIGEN "Build DopeVector with Index<Dimension> as Eigen Matrix (if present) or not." OFF)
if(DOPE_USE_EIGEN)
  message(STATUS "Using Eigen in dope_vector")

  include(FetchContent)
  FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    SYSTEM
    )
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set(EIGEN_BUILD_DOC OFF)
  set(EIGEN_BUILD_PKGCONFIG OFF)
  FetchContent_MakeAvailable(Eigen)

  target_compile_definitions(distance_transform INTERFACE DOPE_USE_EIGEN)
  target_link_libraries(distance_transform INTERFACE Eigen3::Eigen)
endif()



enable_testing()
add_subdirectory(example)