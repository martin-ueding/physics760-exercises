# Copyright © 2014 Martin Ueding <dev@martin-ueding.de>

cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)
 
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_11,code=sm_11)
 
include_directories(
    kernels
    utility
    )

cuda_add_executable(
    random-walk

    main.cu
    )
