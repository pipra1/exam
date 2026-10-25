/* Problem 18: Parallel histogram with MPI_Allreduce.
 * Root scatters the data; each process builds a local histogram;
 * MPI_Allreduce adds them so EVERY process gets the global histogram.
 * Compile: mpicc problem18_histogram.c -o p18
 * Run    : mpirun -np 4 ./p18            */
#include <mpi.h>
#include <stdio.h>
#define TOTAL 16
#define BINS 10

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (TOTAL % size != 0) { if (rank == 0) printf("16 must be divisible by #processes\n"); MPI_Finalize(); return 1; }

    int data[TOTAL] = {1,2,2,3,  2,3,4,4,  1,4,5,5,  0,2,5,9};   /* P0,P1,P2,P3 chunks */
    int local_n = TOTAL / size;
    int local[TOTAL];

    MPI_Scatter(data, local_n, MPI_INT, local, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    int local_hist[BINS] = {0}, global_hist[BINS];
    for (int i = 0; i < local_n; i++) local_hist[local[i]]++;    /* value = bin index */

    MPI_Allreduce(local_hist, global_hist, BINS, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        int total = 0;
        printf("Final histogram after MPI_Allreduce:\nValue    Frequency\n");
        for (int v = 0; v < BINS; v++) { printf("%-8d %d\n", v, global_hist[v]); total += global_hist[v]; }
        printf("Total number of elements = %d\n", total);
    }
    MPI_Finalize();
    return 0;
}
