cmake_minimum_required(VERSION 3.5.0)
project(MACHINE_LEARNING_IN_CPP)

set(EIGEN3_INCLUDE_DIR "E:/Vs Code Scripts/cpp/libraries/eigen-3.4.0")
include_directories(${EIGEN3_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${PROJECT_SOURCE_DIR}/include/data
                  ${PROJECT_SOURCE_DIR}/include/linear_reggresion
                  ${PROJECT_SOURCE_DIR}/include/knn
                  )

add_executable(linearreg src/linear_reggresion/main.cpp src/linear_reggresion/linear.cpp src/data/data_utils.cpp src/data/csv_handler.cpp)
add_executable(knn src/knn/main.cpp src/knn/knn.cpp src/data/data_utils.cpp src/data/csv_handler.cpp)


