cmake_minimum_required(VERSION 3.10)
project(hatch_detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV (minimal dependency)
find_package(OpenCV REQUIRED)
find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} EXACT REQUIRED)


add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ZED_INCLUDE_DIRS}
    include_directories(${CUDA_INCLUDE_DIRS})
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

file(GLOB SRC_FILES src/*.cpp)
file(GLOB HDR_FILES include/*.hpp)

add_executable(${PROJECT_NAME} ${SRC_FILES} ${HDR_FILES})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${ZED_LIBRARIES}
    pthread
)

