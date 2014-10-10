// Copyright © 2014 Martin Ueding <dev@martin-ueding.de>

#include "random_walk.h"

#include <curand_kernel.h>

#include <math.h>

__global__
void random_walk_kernel(int walker_count, int steps, float *distances_dev) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= walker_count) {
        return;
    }

    curandState_t state;

    curand_init(0, idx, 0, &state);

    // Copy variables into a local register to avoid costly global memory
    // accesses.
    int x = 0;
    int y = 0;

    for (int step = 0; step != steps; step++) {
        int random = curand(&state) % 4;

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

    distances_dev[idx] = sqrt(square_distance);
}
