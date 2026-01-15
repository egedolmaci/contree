cmake_minimum_required(VERSION 3.9)
project(ConTree)

# 1. Global C++ Standard (Host code için)
set(CMAKE_CXX_STANDARD 17)

# 2. CUDA Setup
option(ENABLE_CUDA "Enable CUDA GPU solver" OFF)
if (ENABLE_CUDA)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)

# Set default build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "Build type not specified: Use Release by default")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Compiling in release mode")
endif()

# Now include FetchContent
include(FetchContent)

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(WARNING "OpenMP not found - building without parallel support")
endif()

option(BUILD_BENCHMARKS "Build benchmarks" ON)

if (BUILD_BENCHMARKS)
    find_package(benchmark QUIET)

    if (NOT benchmark_FOUND)
        FetchContent_Declare(
                googlebenchmark
                GIT_REPOSITORY https://github.com/google/benchmark.git
                GIT_TAG v1.8.3
        )
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googlebenchmark)
    endif()

    add_executable(contree_benchmarks benchmarks/benchmark_main.cpp)
    target_link_libraries(contree_benchmarks
            contree_lib
            benchmark::benchmark
    )
endif()

# Gather source files
file(GLOB_RECURSE SOURCES "Engine/src/*.cpp")
if (ENABLE_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES "Engine/src/*.cu")
    list(APPEND SOURCES ${CUDA_SOURCES})
endif()

file(GLOB LIB_SRC_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/DataStructures/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/src/*.cpp
        ${SOURCES}
)

# Define include directories
set(INCLUDE_DIRS
        ${CMAKE_SOURCE_DIR}/Utilities/include
        ${CMAKE_SOURCE_DIR}/DataStructures/include
        ${CMAKE_SOURCE_DIR}/Engine/include
)

include_directories(${INCLUDE_DIRS})

# Create library
add_library(contree_lib STATIC ${LIB_SRC_FILES})
target_include_directories(contree_lib PUBLIC ${INCLUDE_DIRS})

# ---------------------------------------------------------
# CUDA AYARLARI (DÜZELTİLDİ)
# ---------------------------------------------------------
if (ENABLE_CUDA)
    # Separable Compilation aktif et
    set_property(TARGET contree_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    # HATA VEREN SATIRLARI SİLDİM (set_target_properties ... CUDA_STANDARD "")
    # CMake'in otomatik standart ayarını kullanmayacağız.
    # Bunun yerine aşağıda manuel olarak -std=c++17 bayrağını ekliyoruz.
    
    # NVCC'ye C++17 kullanmasını manuel olarak söylüyoruz:
    target_compile_options(contree_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>)
    
    # CUDA optimizasyonları
    target_compile_options(contree_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endif()
# ---------------------------------------------------------

# Global Optimization flags for C++ (Release mode)
if (CMAKE_BUILD_TYPE STREQUAL Release)
    if (MSVC)
        message(STATUS "Using MSVC compiler")
        add_compile_options(/Ot)
    else()
        # Sadece C++ dosyaları için (CUDA dosyalarına karışmasın diye ayırdık)
        target_compile_options(contree_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
        target_compile_options(contree_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
        target_compile_options(contree_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>)
    endif()
endif()

# Link OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(contree_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# Create Executable
add_executable(ConTree main.cpp)
target_link_libraries(ConTree contree_lib)

# IPO / LTO
if (CMAKE_BUILD_TYPE STREQUAL Release)
    if( ipo_supported )
        message(STATUS "IPO / LTO enabled")
        set_property(TARGET ConTree PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "IPO / LTO not supported: <${ipo_error}>")
    endif()
else()
    if (MSVC)
        message(STATUS "Enabling profiler")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /PROFILE")
    else ()
        message(STATUS "No profiler")
    endif()
endif()