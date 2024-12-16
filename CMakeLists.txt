cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(cinfer VERSION 0.0.1)
add_library(${PROJECT_NAME})

enable_language(C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.")
endif()
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(cmake/CPM.cmake)
set(ENABLE_CJSON_TEST OFF)
CPMAddPackage(
    NAME cJSON
    GITHUB_REPOSITORY DaveGamble/cJSON
    GIT_TAG  v1.7.18
)

set(BLA_VENDOR "OpenBLAS" CACHE STRING "BLAS/LAPACK Vendors")
find_package(BLAS REQUIRED)
if("${BLAS_INCLUDE_DIRS}" STREQUAL "")
    get_filename_component(BLAS_INCLUDE_DIRS ${BLAS_LIBRARIES} DIRECTORY)
    get_filename_component(BLAS_INCLUDE_DIRS ${BLAS_INCLUDE_DIRS} DIRECTORY)
    string(APPEND BLAS_INCLUDE_DIRS "/include")
endif()

target_include_directories(${PROJECT_NAME}
    PUBLIC include ${BLAS_INCLUDE_DIRS} ${cJSON_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME}
    PUBLIC BLAS::BLAS cjson)
add_subdirectory(src)

enable_language(CXX)
include(CTest)
if(BUILD_TESTING)
    CPMAddPackage(
        NAME googletest
        GITHUB_REPOSITORY google/googletest
        GIT_TAG v1.15.2
    )
    enable_testing()
    set(PROJECT_NAME_TEST "${PROJECT_NAME}_test" )
    add_executable(${PROJECT_NAME_TEST})
    add_test(NAME ${PROJECT_NAME_TEST} COMMAND ${PROJECT_NAME_TEST})
    add_subdirectory(test)
endif()
