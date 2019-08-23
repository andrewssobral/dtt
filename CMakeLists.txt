cmake_minimum_required(VERSION 3.1)
project(dtt)

include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

set(output_dir ".")
# First for the generic no-config case (e.g. with mingw)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${output_dir})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${output_dir})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${output_dir})
# Second, for multi-config builds (e.g. msvc)
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${output_dir})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${output_dir})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${output_dir})
endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Armadillo 9 REQUIRED)
find_package(ArrayFire REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}\n")

add_executable(dtt_test test/dtt_test.cpp test/dtt_test.h include/dtt.h)
target_link_libraries(dtt_test Eigen3::Eigen ArrayFire::af ${ARMADILLO_LIBRARIES} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
