// Copyright © 2014 Martin Ueding <dev@martin-ueding.de>

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void handle_error(cudaError_t err, int line) {
    if (err != cudaSuccess) {
        printf("%s at line %d\n", cudaGetErrorString(err), line);
        exit(EXIT_FAILURE);
    }
}

/**
  Performs a random walk.

  @param[in] walker_count Number of walkers in the array.
  @param[in] steps Number of steps.
  @param[out] distances_dev Distances the walkers have travelled.
  */
__global__
void random_walk_kernel(int walker_count, int steps, float *distances_dev) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= walker_count) {
        return;
    }

    // Copy variables into a local register to avoid costly global memory
    // accesses.
    int x = 0;
    int y = 0;

    for (int step = 0; step != steps; step++) {
        int random = 0; // TODO

        // XXX This is probably a bad implementation since a lot of branching will
        // slow it down.
        if (random == 0) {
            x++;
        }
        else if (random == 1) {
            x--;
        }
        else if (random == 2) {
            y++;
        }
        else if (random == 3) {
            y--;
        }
    }

    float square_distance = x*x + y*y;

    //distances_dev[idx] = sqrt(square_distance);
    distances_dev[idx] = square_distance;
}

int main(int argc, char **argv) {
    int walker_count = 30;
    int steps = 1;
    cudaError_t err;


    // Get memory to store distance in.
    float *distances_host;
    size_t distances_size = sizeof(*distances_host) * walker_count;
    distances_host = (float *) malloc(distances_size);
    assert(distances_host);

    float *distances_dev;
    err = cudaMalloc(&distances_dev, distances_size);
    handle_error(err, __LINE__);

    int block_size = 256;

    //random_walk_kernel<<< walker_count/block_size, block_size >>>(
    random_walk_kernel<<< 1, walker_count >>>(
            walker_count,
            steps,
            distances_dev
            );

    // Copy the results back to the host.
    err = cudaMemcpy(distances_host, distances_dev, distances_size, cudaMemcpyDeviceToHost);
    handle_error(err, __LINE__);

    for (int walker = 0; walker != walker_count; walker++) {
        printf("%g\n", distances_host[walker]);
    }

    return 0;
}
