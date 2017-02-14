#Project
cmake_minimum_required(VERSION 2.8)
project(sgmflow)

# Flags
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -msse4.2")
if (APPLE)
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -msse4.2 -stdlib=libstdc++")
endif (APPLE)
set(CMAKE_BUILD_TYPE Release)

# SPS-Stereo
find_package( OpenCV REQUIRED )
file(GLOB SGMFLOW_SRC_FILES "*.cpp")
add_executable(sgmflow ${SGMFLOW_SRC_FILES})
target_link_libraries(sgmflow png)
target_link_libraries(sgmflow ${OpenCV_LIBS})
