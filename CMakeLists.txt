CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12.1)

PROJECT(sicf_demo)


find_package(OpenCV REQUIRED)
find_package(PCL 1.7 REQUIRED)

include_directories(
               ${PCL_INCLUDE_DIRS}
               ${OpenCV_INCLUDE_DIR}
)

add_definitions(${PCL_DEFINITIONS})

ADD_EXECUTABLE(sicf_demo
      main.cpp
      spatialinvariantcolorfeature.cpp
)

TARGET_LINK_LIBRARIES(sicf_demo
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
)

