/* Problem 17: MPI_Alltoall demo + communication cost measurement.
 * Process i sends element j of its send buffer to process j.
 * Compile: mpicc problem17_alltoall.c -o p17
 * Run    : mpirun -np 4 ./p17
 * Repeat with -np 2, 4, 8, 16 (add --oversubscribe if you have fewer cores). */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---------- Part A: one int per destination ---------- */
    int *sendbuf = malloc(size * sizeof(int));
    int *recvbuf = malloc(size * sizeof(int));
    for (int j = 0; j < size; j++) sendbuf[j] = rank * 10 + j;   /* P0: 0 1 2 3, P1: 10 11 12 13 ... */

    MPI_Alltoall(sendbuf, 1, MPI_INT, recvbuf, 1, MPI_INT, MPI_COMM_WORLD);

    int *all = NULL;                          /* gather so the root prints in order */
    if (rank == 0) all = malloc(size * size * sizeof(int));
    MPI_Gather(recvbuf, size, MPI_INT, all, size, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("After MPI_Alltoall:\n");
        for (int p = 0; p < size; p++) {
            printf("P%d received:", p);
            for (int j = 0; j < size; j++) printf(" %d", all[p * size + j]);
            printf("\n");
        }
        free(all);
    }
    free(sendbuf); free(recvbuf);

    /* ---------- Part B: timing for different message sizes ---------- */
    int counts[] = {1, 64, 1024, 16384};      /* ints sent to EACH process */
    const int ITER = 20;
    if (rank == 0) printf("\nP = %d\n%18s %18s\n", size, "Ints per process", "Time (s)");
    for (int c = 0; c < 4; c++) {
        int cnt = counts[c];
        int *sb = calloc((size_t)size * cnt, sizeof(int));
        int *rb = calloc((size_t)size * cnt, sizeof(int));
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        for (int it = 0; it < ITER; it++)
            MPI_Alltoall(sb, cnt, MPI_INT, rb, cnt, MPI_INT, MPI_COMM_WORLD);
        double t = (MPI_Wtime() - t0) / ITER, tmax;
        MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);  /* slowest process */
        if (rank == 0) printf("%18d %18.9f\n", cnt, tmax);
        free(sb); free(rb);
    }
    MPI_Finalize();
    return 0;
}
