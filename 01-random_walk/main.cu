// Copyright © 2014 Martin Ueding <dev@martin-ueding.de>

#include <stdio.h>
#include <stdlib.h>

void handle_error(cudaError_t err, int line) {
    if (err != cudaSuccess) {
        printf("%s at line %d\n", cudaGetErrorString(err), line);
        exit(EXIT_FAILURE);
    }
}

__global__
void reset_kernel(struct walker *walkers_dev, int walker_count, float *distances_dev) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < walker_count) {
        walkers_dev[idx].x = 0.0;
        walkers_dev[idx].y = 0.0;
        distances_dev[idx] = 0.0;
    }
}

__global__
void random_walk_kernel(struct walker *walkers_dev, int walker_count, float *distances_dev) {
}

struct walker {
    int x;
    int y;
};

int main(int argc, char **argv) {
    int walker_count = 10000;
    cudaError_t err;

    // Get memory to store positions in.
    struct walker *walkers_dev;
    err = cudaMalloc(&walkers_dev, sizeof(*walkers_dev) * walker_count);
    handle_error(err, __LINE__);

    // Get memory to store distance in.
    float *distances_dev;
    err = cudaMalloc(&distances_dev, sizeof(*distances_dev) * walker_count);
    handle_error(err, __LINE__);


    return 0;
}
