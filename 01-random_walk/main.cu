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


    int slot_max = 0;

    for (int bin_idx = 0; bin_idx != bin_count; bin_idx++) {
        if (bins[bin_idx] > slot_max) {
            slot_max = bins[bin_idx];
        }
    }

    for (int bin_idx = 0; bin_idx != bin_count; bin_idx++) {
        printf("%10.1f ", bin_idx * max / bin_count);

        for (int slot_idx = 0; slot_idx != bins[bin_idx] * 50 / slot_max; slot_idx++) {
            printf("#");
        }
        printf("\n");
    }

    free(bins);
}

double compute_average(float *data, int count) {
    double sum = 0.0;
    for (int idx = 0; idx != count; idx++) {
        sum += data[idx];
    }

    return sum / count;
}

double compute_rms(float *data, int count) {
    double sum = 0.0;
    for (int idx = 0; idx != count; idx++) {
        sum += data[idx] * data[idx];
    }

    return sqrt(sum / count);
}

int main(int argc, char **argv) {
    int walker_count = 1000;
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

    int *walkers_dev;
    err = cudaMalloc(&walkers_dev, sizeof(*walkers_dev) * 2 * walker_count);
    handle_error(err, __LINE__);

    curandState_t *curand_states_dev;
    err = cudaMalloc(&curand_states_dev, sizeof(*curand_states_dev) * walker_count);
    handle_error(err, __LINE__);

    int block_size = 256;


    init_kernel<<< (walker_count-1)/block_size + 1, block_size >>>(
            walker_count, walkers_dev, distances_dev, curand_states_dev);

    int steps_per_iter = 100;

    FILE *averages_stream = fopen("averages.csv", "w");
    FILE *rms_stream = fopen("rms.csv", "w");

    int iter_count = 100;

    for (int iter = 0; iter != iter_count; iter++) {
        clock_t start = clock();

        random_walk_kernel<<< (walker_count-1)/block_size + 1, block_size >>>(
                walker_count,
                steps_per_iter,
                walkers_dev,
                distances_dev,
                curand_states_dev
                );

        // Copy the results back to the host.
        err = cudaMemcpy(distances_host, distances_dev, distances_size, cudaMemcpyDeviceToHost);
        handle_error(err, __LINE__);

        clock_t end = clock();

        double average = compute_average(distances_host, walker_count);
        double rms = compute_rms(distances_host, walker_count);

        if (iter == iter_count - 1) {
            printf("The part on the GPU for %d walkers for %d steps took %g seconds.\n", walker_count, iter * steps_per_iter, (end-start) / (float) CLOCKS_PER_SEC);

            plot_histogram(bin_count, walker_count, distances_host);
            printf("\n");
        }

        fprintf(averages_stream, "%d %f\n", iter * steps_per_iter, average);
        fprintf(rms_stream, "%d %f\n", iter * steps_per_iter, rms);
    }


    fclose(averages_stream);
    fclose(rms_stream);

    free(distances_host);
    cudaFree(distances_dev);
    cudaFree(curand_states_dev);
    cudaFree(walkers_dev);

    return 0;
}
