// Copyright © 2014 Martin Ueding <dev@martin-ueding.de>

#pragma once

/**
  Performs a random walk.

  @param[in] walker_count Number of walkers in the array.
  @param[in] steps Number of steps.
  @param[out] distances_dev Distances the walkers have travelled.
  */
__global__
void random_walk_kernel(int walker_count, int steps, float *distances_dev);
