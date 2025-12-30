  cmake_minimum_required(VERSION 3.9)
  project(ConTree)

  set(CMAKE_CXX_STANDARD 17)

  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)

  # Set default build type to Release if not specified (BEFORE FetchContent!)
  if (NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE Release)
      message(STATUS "Build type not specified: Use Release by default")
  endif()

  if (CMAKE_BUILD_TYPE STREQUAL "Release")
      message(STATUS "Compiling in release mode")
  else()
      message(STATUS "Compiling in debug mode")
      add_compile_definitions(DEBUG=1)
  endif()

  # Now include FetchContent - CMAKE_BUILD_TYPE is already set
  include(FetchContent)

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

  # Optimization flags for Release mode
  if (CMAKE_BUILD_TYPE STREQUAL Release)

      if (MSVC)
          message(STATUS "Using MSVC compiler")
          add_compile_options(/Ot)
      else()
          add_compile_options(-O3)
          add_compile_options(-march=native)
          add_compile_options(-funroll-loops)
      endif()

  endif()

  # Gather source files (excluding main.cpp)
  file(GLOB LIB_SRC_FILES
          ${CMAKE_CURRENT_SOURCE_DIR}/DataStructures/src/*.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/Utilities/src/*.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/Engine/src/*.cpp
  )

  # Define include directories
  set(INCLUDE_DIRS
          ${CMAKE_SOURCE_DIR}/Utilities/include
          ${CMAKE_SOURCE_DIR}/DataStructures/include
          ${CMAKE_SOURCE_DIR}/Engine/include
  )

  # Specify include directories
  include_directories(${INCLUDE_DIRS})

  # Create library from sources
  add_library(contree_lib STATIC ${LIB_SRC_FILES})
  target_include_directories(contree_lib PUBLIC ${INCLUDE_DIRS})

  # Create the executable
  add_executable(ConTree main.cpp)
  target_link_libraries(ConTree contree_lib)

  # Set target properties based on build type
  if (CMAKE_BUILD_TYPE STREQUAL Release)
      if( ipo_supported )
          message(STATUS "IPO / LTO enabled")
          set_property(TARGET ConTree PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
      else()
          message(STATUS "IPO / LTO not supported: <${ipo_error}>")
      endif()
  else()

      # enable runtime profiling (with MSVC)
      if (MSVC)
          message(STATUS "Enabling profiler")
          set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /PROFILE")
      else ()
          message(STATUS "No profiler")
      endif()
  endif()