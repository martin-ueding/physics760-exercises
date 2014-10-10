// Copyright © 2014 Martin Ueding <dev@martin-ueding.de>

#pragma once

#include <curand_kernel.h>

__global__
void init_kernel(int walker_count, int *walkers, float *distances_dev, curandState_t * curand_states);

/**
  Performs a random walk.

  @param[in] walker_count Number of walkers in the array.
  @param[in] steps Number of steps.
  @param[out] distances_dev Distances the walkers have travelled.
  */
__global__
void random_walk_kernel(int walker_count, int steps, int *walkers, float *distances_dev, curandState_t * curand_states);

