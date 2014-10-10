// Copyright © 2014 Martin Ueding <dev@martin-ueding.de>

#include "random_walk.h"

#include <math.h>

__global__
void init_kernel(int walker_count, int *walkers, float *distances_dev, curandState_t * curand_states) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= walker_count) {
        return;
    }

    walkers[idx] = 0.0;
    walkers[idx + 1] = 0.0;

    curand_init(0, idx, 0, curand_states + idx);
}

__global__
void random_walk_kernel(int walker_count, int steps, int *walkers, float *distances_dev, curandState_t * curand_states) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= walker_count) {
        return;
    }

    // Copy variables into a local register to avoid costly global device
    // memory accesses.
    int x = walkers[idx];
    int y = walkers[idx + 1];
    curandState_t state = curand_states[idx];

    for (int step = 0; step != steps; step++) {
        int random = curand(&state) % 4;

        if (random < 0) {
            random = -random;
        }

        // XXX This is probably a bad implementation since a lot of branching
        // will slow it down.
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

    // Store variables in global device memory again.
    walkers[idx] = x;
    walkers[idx + 1] = y;
    curand_states[idx] = state;

    // Compute the distance and store that in global device memory.
    float square_distance = x*x + y*y;
    distances_dev[idx] = sqrt(square_distance);
}
