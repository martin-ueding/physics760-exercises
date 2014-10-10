// Copyright © 2014 Martin Ueding <dev@martin-ueding.de>

#include "random_walk.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void handle_error(cudaError_t err, int line) {
    if (err != cudaSuccess) {
        printf("%s at line %d\n", cudaGetErrorString(err), line);
        exit(EXIT_FAILURE);
    }
}


void plot_histogram(int bin_count, int walker_count, float *distances) {
    int *bins = (int *) malloc(sizeof(int) * bin_count);

    for (int bin_idx = 0; bin_idx != bin_count; bin_idx++) {
        bins[bin_idx] = 0;
    }

    float max = 0.0;

    for (int walker = 0; walker != walker_count; walker++) {
        if (distances[walker] > max) {
            max = distances[walker];
        }
    }

    for (int walker = 0; walker != walker_count; walker++) {
        int bin_idx = distances[walker] * bin_count / max;
        if (bin_idx >= bin_count) {
            bin_idx = bin_count - 1;
        }
        bins[bin_idx]++;
    }

    free(distances);

    int slot_max = 0;

    for (int bin_idx = 0; bin_idx != bin_count; bin_idx++) {
        if (bins[bin_idx] > slot_max) {
            slot_max = bins[bin_idx];
        }
    }

    for (int bin_idx = 0; bin_idx != bin_count; bin_idx++) {
        printf("%10f ", bin_idx * max / bin_count);

        for (int slot_idx = 0; slot_idx != bins[bin_idx] * 50 / slot_max; slot_idx++) {
            printf("#");
        }
        printf("\n");
    }

    free(bins);
}

int main(int argc, char **argv) {
    int walker_count = 100000;
    int steps = 2000;
    int bin_count = 30;
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

    clock_t start = clock();

    random_walk_kernel<<< (walker_count-1)/block_size + 1, block_size >>>(
            walker_count,
            steps,
            distances_dev
            );

    // Copy the results back to the host.
    err = cudaMemcpy(distances_host, distances_dev, distances_size, cudaMemcpyDeviceToHost);
    handle_error(err, __LINE__);
    cudaFree(distances_dev);

    clock_t end = clock();

    printf("The part on the GPU for %d walkers for %d steps took %g seconds.\n", walker_count, steps, (end-start) / (float) CLOCKS_PER_SEC);

    /*
    for (int walker = 0; walker != walker_count; walker++) {
        printf("%g\n", distances_host[walker]);
    }
    */

    plot_histogram(bin_count, walker_count, distances_host);

    return 0;
}
