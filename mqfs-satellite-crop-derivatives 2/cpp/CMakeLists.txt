# ====================================================================
# MQFS C++ pricing engine — CMake build
# --------------------------------------------------------------------
# Targets:
#   mqfs_cli        standalone demo (cli.cpp)
#   mqfs_capi       C-ABI shared lib for ctypes (libmqfs_cpp.so)
#   mqfs_cpp        pybind11 extension module (built only if pybind11 found)
#
#   mkdir build && cd build && cmake .. && cmake --build . -j
#
# (c) Nicolo Angileri / MQFS — MIT Licence.
# ====================================================================
cmake_minimum_required(VERSION 3.15)
project(mqfs_pricer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

find_package(OpenMP)

# --- standalone CLI -------------------------------------------------
add_executable(mqfs_cli cli.cpp)
if(OpenMP_CXX_FOUND)
  target_link_libraries(mqfs_cli PRIVATE OpenMP::OpenMP_CXX)
endif()

# --- C-ABI shared library (ctypes) ---------------------------------
add_library(mqfs_capi SHARED capi.cpp)
set_target_properties(mqfs_capi PROPERTIES OUTPUT_NAME mqfs_cpp POSITION_INDEPENDENT_CODE ON)
if(OpenMP_CXX_FOUND)
  target_link_libraries(mqfs_capi PRIVATE OpenMP::OpenMP_CXX)
endif()

# --- optional pybind11 module --------------------------------------
find_package(pybind11 QUIET)
if(pybind11_FOUND)
  pybind11_add_module(mqfs_cpp bindings.cpp)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(mqfs_cpp PRIVATE OpenMP::OpenMP_CXX)
  endif()
  message(STATUS "pybind11 found — building native Python module mqfs_cpp")
else()
  message(STATUS "pybind11 not found — skipping Python module (ctypes ABI still built)")
endif()
